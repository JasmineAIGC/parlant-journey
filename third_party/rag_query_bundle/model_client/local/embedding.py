"""
本地 Embedding 模型封装 — 使用 sentence-transformers 在本地推理。

适用于不依赖外部 Embedding API 服务、直接加载本地模型文件的场景。
与 ``OpenAILikeEmbedding`` 接口完全一致，可无缝替换传入 ``FlexRAGRetriever``。

依赖::

    pip install llama-index sentence-transformers
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, List, Optional

# guard：llama-index + pydantic
try:
    from pydantic import PrivateAttr
    from llama_index.core.embeddings import BaseEmbedding as _BaseEmbedding
    _LLAMA_AVAILABLE = True
except ImportError:
    _LLAMA_AVAILABLE = False
    _BaseEmbedding = object  # type: ignore[assignment,misc]
    PrivateAttr = lambda **kw: None  # type: ignore[assignment]  # noqa: E731


class SentenceTransformerEmbedding(_BaseEmbedding):
    """
    LlamaIndex 兼容的本地 Embedding 模型封装，使用 sentence-transformers 库在本地推理。

    Args:
        model_path:  本地模型目录路径（传给 ``SentenceTransformer()``），
                     也可以是 HuggingFace Hub 模型名称（如 ``BAAI/bge-large-zh-v1.5``）。
        model_name:  可选的可读模型名称（用于日志/属性展示），默认取 ``model_path`` 的最后一段。

    Usage::

        from model_client.local.embedding import SentenceTransformerEmbedding

        # 从 .env 自动加载（EMBEDDING_MODEL_PATH）
        embed = SentenceTransformerEmbedding.from_env()

        # 手动指定路径
        embed = SentenceTransformerEmbedding(
            model_path="/path/to/models/BAAI/bge-large-zh-v1.5"
        )
        vectors = embed._get_text_embeddings(["你好", "world"])
    """

    model_path: str
    model_name: str = "local"
    embed_batch_size: int = 32

    _model: Any = PrivateAttr()

    def __init__(
        self,
        model_path: str,
        model_name: Optional[str] = None,
        embed_batch_size: int = 32,
        **kwargs: Any,
    ) -> None:
        # 先检查依赖，避免因 _BaseEmbedding=object 而产生误导性的 TypeError
        if not _LLAMA_AVAILABLE:
            raise ImportError(
                "SentenceTransformerEmbedding 需要安装 llama-index：\n"
                "pip install llama-index sentence-transformers"
            )
        _name = model_name or os.path.basename(model_path.rstrip("/\\")) or "local"
        super().__init__(
            model_path=model_path,
            model_name=_name,
            embed_batch_size=embed_batch_size,
            **kwargs,
        )
        try:
            from sentence_transformers import SentenceTransformer as _ST
        except ImportError:
            raise ImportError(
                "SentenceTransformerEmbedding 需要安装 sentence-transformers：\n"
                "pip install sentence-transformers"
            )
        self._model = _ST(model_path)

    @classmethod
    def from_env(cls) -> "SentenceTransformerEmbedding":
        """从 ``EMBEDDING_MODEL_PATH`` 环境变量创建实例。

        Raises:
            ValueError: 若 ``EMBEDDING_MODEL_PATH`` 未设置。
        """
        model_path = os.getenv("EMBEDDING_MODEL_PATH")
        if not model_path:
            raise ValueError(
                "EMBEDDING_MODEL_PATH 未设置，无法使用本地 embedding 模式。\n"
                "请通过 .env 文件或系统环境变量设置 EMBEDDING_MODEL_PATH=/path/to/model"
            )
        model_name = os.getenv("EMBEDDING_MODEL") or None
        return cls(model_path=model_path, model_name=model_name)

    # ------------------------------------------------------------------
    # 同步方法
    # ------------------------------------------------------------------

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embeddings([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embeddings([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        import numpy as np
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs).tolist()

    # ------------------------------------------------------------------
    # 异步方法
    # ------------------------------------------------------------------

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return (await self._aget_text_embeddings([query]))[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return (await self._aget_text_embeddings([text]))[0]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        # sentence_transformers.encode() 是 CPU 密集型阻塞调用。
        # 通过 asyncio.to_thread() 将其推入线程池执行，避免阻塞事件循环。
        return await asyncio.to_thread(self._get_text_embeddings, texts)
