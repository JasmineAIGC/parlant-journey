"""
model_client/local/example.py — 本地模型调用示例

演示如何使用 sentence-transformers 在本地进行 Embedding 推理，无需任何外部 API 服务。

前置条件：
    1. 下载好本地 Embedding 模型，例如 BAAI/bge-large-zh-v1.5
    2. 在 .env 中配置（或直接修改下方 MODEL_PATH）：
         EMBEDDING_MODEL_PATH=/path/to/models/BAAI/bge-large-zh-v1.5
         EMBEDDING_MODEL=bge-large-zh-v1.5（可选，用于显示）
    3. pip install sentence-transformers llama-index python-dotenv

运行方式（从项目根目录）：
    python model_client/local/example.py
"""
from __future__ import annotations

import asyncio
import os
import sys

# 确保从任意目录运行时都能找到 model_client
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from model_client.local import SentenceTransformerEmbedding

# 若未配置 .env，可直接修改此路径
MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "/path/to/models/BAAI/bge-large-zh-v1.5")


async def main() -> None:
    print("=" * 60)
    print("【本地 Embedding 模式】")
    print("=" * 60)
    print(f"  模型路径：{MODEL_PATH}\n")

    print("⏳ 正在加载本地模型...")
    embed = SentenceTransformerEmbedding(model_path=MODEL_PATH)
    print(f"✅ 模型加载完成（模型名：{embed.model_name}）\n")

    # ── 同步调用 ──────────────────────────────────────────────────────────────
    print("── 同步 Embedding ──────────────────────────────────────")
    texts = ["什么是向量数据库？", "RAG 检索增强生成的原理", "Sentence Transformers 本地推理"]
    embeddings = embed._get_text_embeddings(texts)
    print(f"  文本数量：{len(embeddings)}")
    print(f"  向量维度：{len(embeddings[0])}")
    for i, (text, vec) in enumerate(zip(texts, embeddings)):
        print(f"  [{i+1}] \"{text[:20]}\" → 前3维: {[round(v, 4) for v in vec[:3]]}")

    # ── 异步调用 ──────────────────────────────────────────────────────────────
    print("\n── 异步 Embedding ──────────────────────────────────────")
    query = "如何提高检索精度？"
    query_vec = await embed._aget_query_embedding(query)
    print(f"  查询：\"{query}\"")
    print(f"  向量维度：{len(query_vec)}")
    print(f"  前5维：{[round(v, 4) for v in query_vec[:5]]}")

    print("\n✅ 本地 Embedding 调用成功")


if __name__ == "__main__":
    asyncio.run(main())
