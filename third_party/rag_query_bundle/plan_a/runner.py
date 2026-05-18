"""
runner.py — Plan A 两种 RAG 模式的批量 query-answer runner。

将 Standalone 和 Parlant 两种 RAG 模式的计算/设置逻辑集中在此，
供 ``examples/`` 脚本直接调用；examples 仅负责参数解析与结果输出。

接口一览
--------
* :func:`load_test_queries`      — 从 JSON 数据集抽样测试问题
* :func:`generate_answer`        — 用 LLM 根据检索片段生成回答（公共辅助函数）
* :func:`run_standalone`         — Standalone 模式：Pipeline 检索 + LLM 回答（无需 Parlant）
* :class:`ParlantServerContext`  — Parlant Server 上下文（**启动时初始化一次**，多轮对话复用）
* :func:`run_parlant_server`     — Parlant Server 模式便捷入口（单次批量查询）
* :func:`run_parlant`            — 轻量 Parlant 模式（无 Server/Agent/Guidelines，仅用于工具调试）
"""
from __future__ import annotations

import json
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Any, Optional

from plan_a.core.types import DocChunk, RetrievalRequest
from plan_a.rag.pipeline import RAGPipeline
from utils.logging import get_logger

if TYPE_CHECKING:
    from model_client.api.llm import LLMClient

logger = get_logger("plan_a.runner")


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_test_queries(path: str | Path, limit: int = 3, seed: int = 0) -> list[str]:
    """
    从 JSON 数据集文件中抽样测试问题。

    Args:
        path:  JSON 文件路径（格式：``list of {"question": str, ...}``）。
        limit: 返回的问题数量上限。
        seed:  随机抽样的固定种子，保证结果可复现。

    Returns:
        问题文本列表（长度 ``<= limit``）。

    Raises:
        ValueError: 若文件中没有可用问题。
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        samples = json.load(f)
    questions = [
        item["question"].strip()
        for item in samples
        if isinstance(item, dict) and isinstance(item.get("question"), str) and item["question"].strip()
    ]
    if not questions:
        raise ValueError(f"未从数据集读取到可用问题：{path}")
    if len(questions) <= limit:
        return questions
    return Random(seed).sample(questions, limit)


# ---------------------------------------------------------------------------
# 公共辅助：LLM 回答生成
# ---------------------------------------------------------------------------

async def generate_answer(llm: "LLMClient", question: str, chunks: list[DocChunk]) -> str:
    """
    根据检索到的文档片段用 LLM 生成回答。

    Args:
        llm:      :class:`~model_client.api.llm.LLMClient` 实例（运行时必须可用）。
        question: 用户问题。
        chunks:   RAGPipeline 返回的文档片段列表。

    Returns:
        LLM 生成的回答文本；无相关内容时返回固定提示语。
    """
    if not chunks:
        return "知识库中未找到与该问题相关的内容，无法回答。"
    context_text = "\n\n".join(
        f"[来源: {c.metadata.get('title') or c.source}]\n{c.content}"
        for c in chunks
    )
    prompt = (
        "请根据以下知识库内容简洁回答用户问题。若内容不足以回答，请如实说明。\n\n"
        f"用户问题：{question}\n\n"
        f"知识库内容：\n{context_text}\n\n回答："
    )
    try:
        return await llm.chat_async([{"role": "user", "content": prompt}])
    except Exception as exc:
        logger.error("[Runner] LLM 调用失败: %s", exc)
        return f"LLM 调用失败，无法生成回答：{exc}"


# ---------------------------------------------------------------------------
# 内部辅助：通过 Parlant RAG Tool 执行一组查询（供两种 Parlant 模式复用）
# ---------------------------------------------------------------------------

async def _run_tool_queries(
    search_knowledge,
    queries: list[str],
    llm: Optional["LLMClient"] = None,
) -> list[dict]:
    """
    内部辅助：通过 Parlant RAG Tool 执行一组查询。

    直接以 ``(None, query)`` 调用 ``@p.tool`` 函数（``None`` 表示无 Parlant 会话上下文），
    等同于 Parlant Agent 内部触发 Tool 调用时的检索路径，但无需启动 Server。

    Args:
        search_knowledge: :func:`~plan_a.tools.rag.make_rag_tool` 返回的 Parlant Tool 函数。
        queries:          查询文本列表。
        llm:              可选 LLMClient；提供时基于检索结果生成回答，省略时只返回检索结果。

    Returns:
        每条查询对应一个 dict：``{"request": {...}, "response": {...}}``，
        提供 llm 时还包含 ``"answer"`` 键。
    """
    results: list[dict] = []
    for query in queries:
        tool_result = await search_knowledge(None, query)
        tool_data: dict[str, Any] = getattr(tool_result, "data", {}) if tool_result is not None else {}
        tool_resp: dict[str, Any] = tool_data.get("response") or {}
        chunk_count: int = tool_resp.get("chunk_count", 0)

        logger.info("[Runner] Tool 检索完成  query=%r  chunk_count=%d", query, chunk_count)

        row: dict[str, Any] = {
            "request": tool_data.get("request") or {"query": query},
            "response": tool_resp,
        }

        if llm is not None:
            formatted: str = tool_resp.get("formatted") or ""
            if formatted:
                answer_prompt = (
                    "请根据以下知识库内容简洁回答用户问题。若内容不足以回答，请如实说明。\n\n"
                    f"用户问题：{query}\n\n"
                    f"知识库内容：\n{formatted}\n\n回答："
                )
                try:
                    answer = await llm.chat_async([{"role": "user", "content": answer_prompt}])
                    row["answer"] = answer.strip()
                except Exception as exc:
                    logger.error("[Runner] LLM 调用失败: %s", exc)
                    row["answer"] = f"LLM 调用失败：{exc}"
            else:
                row["answer"] = "知识库中未找到与该问题相关的内容，无法回答。"

        results.append(row)
    return results


# ---------------------------------------------------------------------------
# Standalone 模式
# ---------------------------------------------------------------------------

async def run_standalone(
    pipeline: RAGPipeline,
    queries: list[str],
    llm: "LLMClient",
    top_k: Optional[int] = None,
    include_metadata: bool = False,
) -> list[dict]:
    """
    Standalone 模式 Q&A：RAGPipeline 直接检索 → LLMClient 生成回答。

    无需安装 Parlant，适合脚本测试与批量评估。

    Args:
        pipeline:         已初始化的 :class:`~plan_a.rag.pipeline.RAGPipeline`。
        queries:          查询文本列表。
        llm:              :class:`~model_client.api.llm.LLMClient` 实例（运行时必须可用）。
        top_k:            最大返回片段数；``None`` 使用 pipeline 默认值。
        include_metadata: 是否在检索结果中包含文档元数据（默认 False）。

    Returns:
        每条查询对应一个 dict：``{"retrieval": {...}, "answer": str}``。
    """
    results: list[dict] = []
    for query in queries:
        logger.info("[Runner.Standalone] query=%r  top_k=%s", query, top_k)
        response = await pipeline.retrieve_standard(
            RetrievalRequest(query=query, top_k=top_k, include_metadata=include_metadata)
        )
        answer = await generate_answer(llm, query, response.chunks)
        results.append({
            "retrieval": response.to_dict(include_metadata=include_metadata),
            "answer": answer.strip(),
        })
        logger.info("[Runner.Standalone] 完成  chunks=%d", len(response.chunks))
    return results


# ---------------------------------------------------------------------------
# Parlant 模式（无 Server，后台执行，不弹 UI）
# ---------------------------------------------------------------------------

async def run_parlant(
    pipeline: RAGPipeline,
    queries: list[str],
    llm: Optional["LLMClient"] = None,
    top_k: Optional[int] = None,
    include_metadata: bool = False,
) -> list[dict]:
    """
    Parlant 模式 Q&A（**无需启动 Parlant Server，后台执行，不弹 UI**）。

    通过 :func:`~plan_a.tools.rag.make_rag_tool` 获取 ``@p.tool`` 检索函数，
    在后台程序化调用，等同于 Parlant Agent 内部触发 Tool 时的检索路径，
    但不启动任何 HTTP Server 或 Web 界面。

    Args:
        pipeline:         已初始化的 :class:`~plan_a.rag.pipeline.RAGPipeline`。
        queries:          查询文本列表。
        llm:              可选 :class:`~model_client.api.llm.LLMClient`（运行时必须可用）；
                          提供时在检索后生成 LLM 回答，省略时只返回检索结果。
        top_k:            最大返回片段数；``None`` 使用 pipeline 默认值。
        include_metadata: 是否在检索结果中包含文档元数据（默认 False）。

    Returns:
        每条查询对应一个 dict：
        ``{"request": {...}, "response": {...}}``，提供 llm 时还包含 ``"answer"``。

    Raises:
        ImportError: 若未安装 ``parlant``（pip install parlant）。
        RuntimeError: 若未安装 ``torch``（Parlant LiteLLM 适配器依赖）。
    """
    from plan_a.tools import make_rag_tool

    search_knowledge = make_rag_tool(pipeline, top_k=top_k, include_metadata=include_metadata)
    logger.info(
        "[Runner.Parlant] 开始（无 Server，后台执行）  query_count=%d  top_k=%s",
        len(queries), top_k,
    )
    results = await _run_tool_queries(search_knowledge, queries, llm)
    logger.info("[Runner.Parlant] 完成  result_count=%d", len(results))
    return results


# ---------------------------------------------------------------------------
# 内部哨兵：跳过 Parlant HTTP Server 启动
# ---------------------------------------------------------------------------

class _ParlantSkipServerStart(Exception):
    """
    哨兵异常：以程序化方式退出 p.Server 上下文时传入，
    令 p.Server.__aexit__ 走"有异常"分支（仅清理资源），
    跳过实体评估（~20s）、uvicorn HTTP Server 启动和 UI 链接打印。
    """


# ---------------------------------------------------------------------------
# Parlant Server 上下文（启动时初始化一次，多轮对话复用）
# ---------------------------------------------------------------------------

class ParlantServerContext:
    """
    Parlant Server 上下文管理器：**启动时初始化一次**，多轮对话复用。

    与 :func:`run_parlant_server` 相比，此类将"启动初始化"与"查询执行"分离：

    * ``__aenter__``：启动 Parlant Server、创建 Agent、配置 Guidelines（只执行一次）。
    * :meth:`run_queries`：使用已初始化的 Server 执行查询（可多次调用）。
    * ``__aexit__``：以哨兵异常方式退出，跳过 uvicorn HTTP Server 启动，只做资源清理。

    适用于需要长期运行的服务场景——Server 和 Guidelines 只在进入上下文时创建一次，
    后续每轮对话直接调用 :meth:`run_queries`，不会重复初始化。

    **注意**：通过哨兵异常绕过 Parlant 内置的 HTTP Server 启动，
    程序不会主动打开浏览器或弹出 UI 界面，也不会打印 Sandbox UI 链接。

    使用方式::

        async with ParlantServerContext(pipeline) as ctx:
            # Server + Agent + Guidelines 已就绪，以下可多次调用
            results1 = await ctx.run_queries(["问题一"], llm=llm)
            results2 = await ctx.run_queries(["问题二", "问题三"], llm=llm)

    Args:
        pipeline:          已初始化的 :class:`~plan_a.rag.pipeline.RAGPipeline`。
        agent_name:        Parlant Agent 名称。
        agent_description: Parlant Agent 描述。
        top_k:             最大返回片段数；``None`` 使用 pipeline 默认值。
        include_metadata:  是否在检索结果中包含文档元数据（默认 False）。
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        agent_name: str = "plan-a-rag-agent",
        agent_description: str = "Plan A RAG Agent：知识库问答助手",
        top_k: Optional[int] = None,
        include_metadata: bool = False,
    ) -> None:
        self.pipeline = pipeline
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.top_k = top_k
        self.include_metadata = include_metadata

        self._server_ctx: Any = None
        self._server: Any = None
        self._search_knowledge: Any = None
        self.agent: Any = None
        self.agent_info: dict[str, Any] = {}

    async def __aenter__(self) -> "ParlantServerContext":
        try:
            import parlant.sdk as p
        except ModuleNotFoundError as exc:
            if getattr(exc, "name", None) == "torch" or "No module named 'torch'" in str(exc):
                raise RuntimeError("Parlant 依赖 torch，请先安装：pip install torch") from exc
            raise RuntimeError("未安装 parlant，请先安装：pip install parlant") from exc

        from model_client._utils import bridge_parlant_litellm_env
        from plan_a.guidelines import setup_guidelines
        from plan_a.tools import make_rag_tool

        bridge_parlant_litellm_env()
        logger.info(
            "[ParlantServerContext] 启动 Parlant Server（NLP: LiteLLM）  top_k=%s",
            self.top_k,
        )

        # 启动 Server
        self._server_ctx = p.Server(nlp_service=p.NLPServices.litellm)
        self._server = await self._server_ctx.__aenter__()

        # 创建 Agent + Guidelines
        self.agent = await self._server.create_agent(
            name=self.agent_name, description=self.agent_description
        )
        scoped_gl, tool_gl = await setup_guidelines(
            self.agent, self.pipeline, top_k=self.top_k, include_metadata=self.include_metadata
        )
        self._search_knowledge = make_rag_tool(
            self.pipeline, top_k=self.top_k, include_metadata=self.include_metadata
        )

        self.agent_info = {
            "id": self.agent.id,
            "name": self.agent_name,
            "guidelines": {
                "scoped_retriever": scoped_gl.id,
                "rag_tool": tool_gl.id,
            },
        }
        logger.info("[ParlantServerContext] Agent 就绪  id=%s", self.agent.id)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if self._server_ctx is not None:
            server_ctx = self._server_ctx
            self._server_ctx = None
            self._server = None
            # 传入哨兵异常：令 p.Server.__aexit__ 走"有异常"分支（只做资源清理），
            # 跳过 _process_evaluations()、uvicorn 启动和 UI 链接打印。
            _sentinel = _ParlantSkipServerStart()
            await server_ctx.__aexit__(type(_sentinel), _sentinel, None)
        return False

    async def run_queries(
        self,
        queries: list[str],
        llm: Optional["LLMClient"] = None,
    ) -> list[dict]:
        """
        使用已初始化的 Server 执行一组查询。

        Server、Agent 和 Guidelines **不会重新创建**，可多次调用以支持多轮对话。

        Args:
            queries: 查询文本列表。
            llm:     可选 LLMClient；提供时在检索后生成 LLM 回答，省略时只返回检索结果。

        Returns:
            每条查询对应一个 dict：``{"request": {...}, "response": {...}}``，
            提供 llm 时还包含 ``"answer"`` 键。
        """
        logger.info("[ParlantServerContext] 执行查询  query_count=%d", len(queries))
        results = await _run_tool_queries(self._search_knowledge, queries, llm)
        logger.info("[ParlantServerContext] 查询完成  result_count=%d", len(results))
        return results


# ---------------------------------------------------------------------------
# Parlant Server 模式便捷入口（单次批量查询）
# ---------------------------------------------------------------------------

async def run_parlant_server(
    pipeline: RAGPipeline,
    queries: list[str],
    agent_name: str = "plan-a-rag-agent",
    agent_description: str = "Plan A RAG Agent：知识库问答助手",
    llm: Optional["LLMClient"] = None,
    top_k: Optional[int] = None,
    include_metadata: bool = False,
) -> tuple[dict, list[dict]]:
    """
    Parlant Server 模式便捷入口：启动 Server、创建 Agent、配置 Guidelines，
    执行一批查询后返回结果，全程后台运行，不弹出 UI。

    Server 和 Guidelines 在本次调用中**只初始化一次**，所有 queries 共用同一 Server 实例。
    若需跨多次调用复用 Server（服务场景），请改用 :class:`ParlantServerContext`。

    NLP 服务使用 ``NLPServices.litellm``，模型来自 ``.env`` 配置（``QWEN3_*``/``LLM_*``），
    由 :func:`~model_client._utils.bridge_parlant_litellm_env` 自动桥接。

    **注意**：通过哨兵异常绕过 Parlant 内置的 HTTP Server 启动，
    程序不会主动打开浏览器、弹出 UI 界面或打印 Sandbox UI 链接。

    Args:
        pipeline:          已初始化的 :class:`~plan_a.rag.pipeline.RAGPipeline`。
        queries:           查询文本列表。
        agent_name:        Parlant Agent 名称。
        agent_description: Parlant Agent 描述。
        llm:               可选 LLMClient（运行时必须可用，用于生成回答）。
        top_k:             最大返回片段数；``None`` 使用 pipeline 默认值。
        include_metadata:  是否在检索结果中包含文档元数据（默认 False）。

    Returns:
        ``(agent_info, results)``：
        - ``agent_info`` — 包含 Agent ID 和 Guideline ID 的元数据字典；
        - ``results``    — 每条查询对应一个结果字典。

    Raises:
        ImportError: 若未安装 ``parlant``（pip install parlant）。
        RuntimeError: 若未安装 ``torch``（Parlant LiteLLM 适配器依赖）。
    """
    async with ParlantServerContext(
        pipeline,
        agent_name=agent_name,
        agent_description=agent_description,
        top_k=top_k,
        include_metadata=include_metadata,
    ) as ctx:
        results = await ctx.run_queries(queries, llm)

    logger.info("[Runner.ParlantServer] 完成  result_count=%d", len(results))
    return ctx.agent_info, results
