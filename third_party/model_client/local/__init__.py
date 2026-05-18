"""
model_client.local — 本地模型推理（无需 API 服务）。

提供：
  - :class:`SentenceTransformerEmbedding` — 基于 sentence-transformers 的本地 Embedding

适用场景：不依赖外部 API 服务，直接在本地加载并运行模型文件。
"""
from model_client.local.embedding import SentenceTransformerEmbedding

__all__ = [
    "SentenceTransformerEmbedding",
]
