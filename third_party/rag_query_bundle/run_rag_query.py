"""
run_rag_query.py — Parlant Server 模式单条 RAG 查询入口（后台执行，无 UI）

本目录是自包含的（含 plan_a/、model_client/、utils/），
从本目录直接运行即可，无需额外设置 PYTHONPATH：

    cd rag_query_bundle
    python run_rag_query.py --query "你的问题"

执行流程与阶段划分
------------------
RAG 的核心职责是**检索**，LLM 生成回答属于后处理，两者严格分离：

  1. 初始化  — Parlant Server 启动 + LiteLLM 服务初始化 + 知识库 Embedding 缓存
               （一次性，服务存续期内不会重复）
  2. RAG 检索 — Agent RAG Tool 向量检索（不含 LLM 调用）
  3. 后处理  — 组装输出；加 ``--answer`` 时额外调 LLM 生成回答

各阶段耗时均会打印到终端（格式：``⏱  <阶段>  took=XXXms``）。

输出 JSON 结构
--------------
::

    {
      "query":     "<查询文本>",
      "retrieval": {
        "top_k":       <实际使用的检索片段上限>,
        "chunk_count": <命中的片段数>,
        "took_ms":     <RAG Tool 内部向量检索耗时（毫秒）>,
        "chunks": [
          { "content": "<片段正文>", "source": "<来源>", "score": <相似度> },
          ...
        ]
      },
      "answer": "<LLM 生成的回答（--answer 开启时）或空字符串>",
      "timing": {
        "init_ms":        <初始化阶段耗时（毫秒）>,
        "rag_ms":         <RAG 检索阶段耗时（毫秒）>,
        "postprocess_ms": <后处理阶段耗时（毫秒）>,
        "total_ms":       <总耗时（毫秒）>
      }
    }

详细字段说明见同目录 README.md。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# 将本脚本所在目录（rag_query_bundle/）插入 sys.path，使 plan_a/model_client/utils 可直接导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()

from plan_a import PlanAConfig, RAGPipeline
from plan_a.runner import ParlantServerContext
from utils.logging import setup_logging


logger = setup_logging("rag_query", level="INFO")
_DEFAULT_QUERY = "what is one of the stars of The Newcomers known for?"


def _ms(start: float) -> int:
    """返回从 start 到现在的毫秒数（整数）。"""
    return int((time.monotonic() - start) * 1000)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parlant 模式 RAG 单条查询（后台执行，无 UI）",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--query", type=str, default=_DEFAULT_QUERY, help="检索查询文本")
    parser.add_argument("--top-k", type=int, default=None, metavar="K", help="检索返回的最大片段数（默认使用 pipeline 配置）")
    parser.add_argument("--include-metadata", action="store_true", help="在检索结果中包含文档元数据")
    parser.add_argument("--answer", action="store_true", help="后处理阶段调用 LLM 生成回答（默认关闭，开启后耗时增加）")
    parser.add_argument("--output-format", choices=["text", "json"], default="text", help="输出格式")
    parser.add_argument("--output-file", type=str, default=None, metavar="PATH", help="输出 JSON 文件路径")
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    t_total = time.monotonic()

    # ------------------------------------------------------------------
    # 阶段 1：初始化（Parlant Server 启动 + LiteLLM 服务初始化 + 知识库 Embedding 缓存）
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    # 从环境变量加载配置，构建 RAGPipeline 并初始化向量索引
    config = PlanAConfig.from_env()
    pipeline = RAGPipeline(config)
    await pipeline.initialize()

    # 若开启 --answer，提前创建 LLMClient 供后处理阶段使用
    llm = None
    if args.answer:
        from model_client.api.llm import LLMClient
        llm = LLMClient()

    # 启动 Parlant Server，创建 Agent 和 Guidelines（哨兵模式，不弹出 UI）
    ctx = ParlantServerContext(
        pipeline,
        top_k=args.top_k,
        include_metadata=args.include_metadata,
    )
    await ctx.__aenter__()
    init_ms = _ms(t0)
    print(f"⏱  初始化完成  took={init_ms}ms")

    try:
        # ------------------------------------------------------------------
        # 阶段 2：RAG 检索（Agent RAG Tool 向量检索，不含 LLM 调用）
        # ------------------------------------------------------------------
        t0 = time.monotonic()
        # 调用 RAG Tool 执行向量检索，llm=None 确保此阶段不触发 LLM
        results = await ctx.run_queries([args.query], llm=None)
        rag_ms = _ms(t0)
        result = results[0] if results else {}
        resp = result.get("response", {})
        print(f"⏱  RAG 检索完成  took={rag_ms}ms  chunks={resp.get('chunk_count', 0)}")

        # ------------------------------------------------------------------
        # 阶段 3：后处理（可选 LLM 生成回答 + 组装输出 dict）
        # ------------------------------------------------------------------
        t0 = time.monotonic()
        answer = ""
        if llm is not None:
            # 用检索到的 formatted 上下文构建 prompt，调用 LLM 生成回答
            formatted: str = resp.get("formatted") or ""
            if formatted:
                answer_prompt = (
                    "请根据以下知识库内容简洁回答用户问题。若内容不足以回答，请如实说明。\n\n"
                    f"用户问题：{args.query}\n\n"
                    f"知识库内容：\n{formatted}\n\n回答："
                )
                try:
                    answer = (await llm.chat_async([{"role": "user", "content": answer_prompt}])).strip()
                except Exception as exc:
                    logger.error("[run_rag_query] LLM 调用失败: %s", exc)
                    answer = f"LLM 调用失败：{exc}"
            else:
                answer = "知识库中未找到与该问题相关的内容，无法回答。"

        postprocess_ms = _ms(t0)
        total_ms = _ms(t_total)
        print(f"⏱  后处理完成   took={postprocess_ms}ms  |  总耗时={total_ms}ms")

    finally:
        # 退出 Parlant Server 上下文，释放资源
        await ctx.__aexit__(None, None, None)

    # 组装最终输出结构
    output = {
        "query": args.query,
        "retrieval": {
            "top_k": resp.get("top_k"),
            "chunk_count": resp.get("chunk_count", 0),
            "took_ms": resp.get("took_ms"),
            "chunks": resp.get("chunks", []),
        },
        "answer": answer,
        "timing": {
            "init_ms": init_ms,
            "rag_ms": rag_ms,
            "postprocess_ms": postprocess_ms,
            "total_ms": total_ms,
        },
    }

    # 按输出格式写入终端或文件
    use_json = args.output_format == "json" or args.output_file is not None
    if use_json:
        text = json.dumps(output, ensure_ascii=False, indent=2)
        if args.output_file:
            out_path = Path(args.output_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text, encoding="utf-8")
            print(f"✅ 结果已保存至：{out_path}")
        else:
            print(text)
    else:
        chunks = output["retrieval"]["chunk_count"]
        answer_text = output["answer"]
        print(f"✅ 查询完成  chunks={chunks}")
        if answer_text:
            print(f"   回答：{answer_text[:300]}")


if __name__ == "__main__":
    asyncio.run(main(_parse_args()))
