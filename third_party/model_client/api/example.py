"""
model_client/api/example.py — API 模式调用示例

演示如何通过 OpenAI 兼容 API 调用远程 LLM 和 Embedding 服务。

前置条件：
    1. 复制并填写 .env.example → .env，配置以下变量：
         QWEN3_API_KEY=sk-xxxx（或 EMPTY）
         QWEN3_BASE_URL=http://10.x.x.x:18023/v1
         QWEN3_MODEL=Qwen2.5-7B-Instruct
         EMBEDDING_BASE_URL=http://10.x.x.x:8018/v1
         EMBEDDING_MODEL=bge-large-zh-v1.5
    2. 启动 vLLM LLM 服务和 Embedding 服务
    3. pip install litellm openai python-dotenv llama-index

运行方式（从项目根目录）：
    python model_client/api/example.py
"""
from __future__ import annotations

import asyncio
import os
import sys

# 确保从任意目录运行时都能找到 model_client
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from model_client.api import LLMClient, LLMConfig, OpenAILikeEmbedding


async def main() -> None:
    # ── 1. LLM 调用示例 ──────────────────────────────────────────────────────
    print("=" * 60)
    print("【LLM API 模式】")
    print("=" * 60)

    # 从 .env 自动加载配置（QWEN3_* > LLM_* > OPENAI_*）
    config = LLMConfig.from_env()
    print(f"  模型：{config.model}")
    print(f"  地址：{config.base_url}")

    client = LLMClient(config=config)
    reply = await client.chat_async(
        messages=[{"role": "user", "content": "用一句话介绍你自己"}]
    )
    print(f"  回复：{reply}\n")

    # ── 2. Embedding API 调用示例 ─────────────────────────────────────────────
    print("=" * 60)
    print("【Embedding API 模式】")
    print("=" * 60)

    # 从 .env 自动加载（EMBEDDING_BASE_URL / EMBEDDING_MODEL）
    embed_model = OpenAILikeEmbedding.from_env()
    print(f"  模型：{embed_model.model_name}")
    print(f"  地址：{embed_model.base_url}")

    texts = ["什么是向量数据库？", "RAG 检索增强生成的原理"]
    embeddings = embed_model._get_text_embeddings(texts)
    print(f"  文本数量：{len(embeddings)}")
    print(f"  向量维度：{len(embeddings[0])}")
    print(f"  第一个向量前5维：{embeddings[0][:5]}")


if __name__ == "__main__":
    asyncio.run(main())
