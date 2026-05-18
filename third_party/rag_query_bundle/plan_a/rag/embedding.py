"""
embedding.py — 文本向量化组件（Embedder）。

对应架构图中 RAG 服务的 Embedding 步骤。
底层统一复用 ``model_client/``：

- 本地模型：``SentenceTransformerEmbedding``（``EMBEDDING_MODEL_PATH``）
- API 模式：``OpenAILikeEmbedding``（``EMBEDDING_BASE_URL`` 等）

所有配置均从 ``.env`` / 环境变量读取。
"""
from __future__ import annotations

import os
from typing import Any, List, Optional

from model_client import OpenAILikeEmbedding, SentenceTransformerEmbedding
from utils.logging import get_logger


logger = get_logger("plan_a.rag.embedding")


class Embedder:
    """
    文本向量化器。

    封装 ``model_client.OpenAILikeEmbedding``，提供简洁的异步接口：

    - :meth:`embed_query`  — 向量化单条查询文本
    - :meth:`embed_texts`  — 批量向量化文本列表

    客户端按需懒初始化（首次调用时从环境变量读取配置）。

    Examples
    --------
    ::

        embedder = Embedder()
        vector = await embedder.embed_query("退货政策是什么？")
    """

    def __init__(self, client: Optional[Any] = None) -> None:
        """
        Args:
            client: 预先创建的 model_client embedding 客户端；
                    为 ``None`` 时首次使用时从环境变量自动创建。
        """
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is None:
            model_path = os.getenv("EMBEDDING_MODEL_PATH")
            if model_path:
                logger.info("[Embedder] 从环境变量初始化本地 SentenceTransformerEmbedding")
                self._client = SentenceTransformerEmbedding(
                    model_path=model_path,
                    model_name=os.getenv("EMBEDDING_MODEL") or None,
                )
            else:
                logger.info("[Embedder] 从环境变量初始化 OpenAILikeEmbedding")
                self._client = OpenAILikeEmbedding.from_env()
        return self._client

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量向量化文本。

        Args:
            texts: 要向量化的文本列表。

        Returns:
            每条文本对应的向量（与索引维度一致）。
        """
        client = self._ensure_client()
        return await client._aget_text_embeddings(texts)

    async def embed_query(self, query: str) -> List[float]:
        """
        向量化单条查询文本。

        Args:
            query: 查询文本。

        Returns:
            查询向量。
        """
        vectors = await self.embed_texts([query])
        return vectors[0]
