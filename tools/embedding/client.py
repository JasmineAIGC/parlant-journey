"""
tools.embedding.client
~~~~~~~~~~~~~~~~~~~~~~

通用 OpenAI 兼容 Embedding 客户端，以及可选的 LlamaIndex 适配层。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional

from tools._env import first_env, load_dotenv_candidates

try:
    from pydantic import PrivateAttr
    from llama_index.core.embeddings import BaseEmbedding as _BaseEmbedding
    _LLAMA_AVAILABLE = True
except ImportError:
    _LLAMA_AVAILABLE = False
    _BaseEmbedding = object  # type: ignore[assignment,misc]
    PrivateAttr = lambda **kw: None  # type: ignore[assignment]  # noqa: E731


def _make_openai_clients(
    base_url: str,
    api_key: Optional[str],
    access_key: Optional[str] = None,
):
    """返回 ``(sync_client, async_client)``，均已配置连接参数。"""
    try:
        from openai import OpenAI, AsyncOpenAI  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "Embedding 客户端需要安装 openai SDK：pip install openai>=1.0.0"
        ) from exc
    headers = {"access-key": access_key} if access_key else {}
    kw: dict[str, Any] = dict(
        api_key=api_key or "EMPTY",
        base_url=base_url,
        default_headers=headers or None,
    )
    return OpenAI(**kw), AsyncOpenAI(**kw)


@dataclass
class OpenAIEmbeddingConfig:
    """通用 Embedding 客户端配置。"""

    base_url: str
    model_name: str
    api_key: Optional[str] = None
    access_key: Optional[str] = None
    normalize_embeddings: bool = True
    embed_batch_size: int = 5

    @classmethod
    def from_env(cls) -> "OpenAIEmbeddingConfig":
        """从环境变量或 .env 创建配置。"""
        load_dotenv_candidates(start_path=__file__)
        return cls(
            base_url=first_env(
                ["EMBEDDING_BASE_URL", "QWEN3_BASE_URL", "LLM_BASE_URL"],
                default="http://127.0.0.1:8018/v1",
            ) or "http://127.0.0.1:8018/v1",
            model_name=first_env(
                ["EMBEDDING_MODEL", "QWEN3_MODEL", "LLM_MODEL"],
                default="bge-large-zh-v1.5",
            ) or "bge-large-zh-v1.5",
            api_key=first_env(["EMBEDDING_API_KEY", "QWEN3_API_KEY", "LLM_API_KEY"]),
            access_key=first_env(
                ["EMBEDDING_ACCESS_KEY", "QWEN3_ACCESS_KEY", "LLM_ACCESS_KEY", "ACCESS_KEY"]
            ),
        )


class OpenAIEmbeddingClient:
    """更通用的 OpenAI 兼容 Embedding 客户端。"""

    def __init__(self, config: OpenAIEmbeddingConfig) -> None:
        self.config = config
        self._client, self._aclient = _make_openai_clients(
            config.base_url,
            config.api_key,
            config.access_key,
        )

    @classmethod
    def from_env(cls) -> "OpenAIEmbeddingClient":
        return cls(OpenAIEmbeddingConfig.from_env())

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        response = self._client.embeddings.create(
            model=self.config.model_name,
            input=texts,
        )
        embeddings = [item.embedding for item in response.data]
        return self._maybe_normalize(embeddings)

    async def aembed_query(self, query: str) -> List[float]:
        return (await self.aembed_texts([query]))[0]

    async def aembed_text(self, text: str) -> List[float]:
        return (await self.aembed_texts([text]))[0]

    async def aembed_texts(self, texts: List[str]) -> List[List[float]]:
        response = await self._aclient.embeddings.create(
            model=self.config.model_name,
            input=texts,
        )
        embeddings = [item.embedding for item in response.data]
        return self._maybe_normalize(embeddings)

    def _maybe_normalize(self, embeddings: List[List[float]]) -> List[List[float]]:
        """若 normalize_embeddings=True，对每个向量做 L2 归一化。"""
        if not self.config.normalize_embeddings:
            return embeddings
        normalized: List[List[float]] = []
        for vector in embeddings:
            norm = math.sqrt(sum(float(value) * float(value) for value in vector)) or 1.0
            normalized.append([float(value) / norm for value in vector])
        return normalized


class OpenAILikeEmbedding(_BaseEmbedding):
    """
    LlamaIndex 适配层：复用 ``OpenAIEmbeddingClient``，保持旧 API 兼容。
    """

    base_url: str
    model_name: str
    api_key: Optional[str] = None
    access_key: Optional[str] = None
    normalize_embeddings: bool = True
    embed_batch_size: int = 5

    _embedding_client: Any = PrivateAttr()

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: Optional[str] = None,
        access_key: Optional[str] = None,
        normalize_embeddings: bool = True,
        embed_batch_size: int = 5,
        **kwargs: Any,
    ) -> None:
        if not _LLAMA_AVAILABLE:
            raise ImportError(
                "OpenAILikeEmbedding 需要安装 llama-index：\n"
                "pip install llama-index"
            )
        super().__init__(
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            access_key=access_key,
            normalize_embeddings=normalize_embeddings,
            embed_batch_size=embed_batch_size,
            **kwargs,
        )
        self._embedding_client = OpenAIEmbeddingClient(
            OpenAIEmbeddingConfig(
                base_url=base_url,
                model_name=model_name,
                api_key=api_key,
                access_key=access_key,
                normalize_embeddings=normalize_embeddings,
                embed_batch_size=embed_batch_size,
            )
        )

    @classmethod
    def from_env(cls) -> "OpenAILikeEmbedding":
        """从环境变量（或 ``.env`` 文件）创建实例。"""
        config = OpenAIEmbeddingConfig.from_env()
        return cls(
            base_url=config.base_url,
            model_name=config.model_name,
            api_key=config.api_key,
            access_key=config.access_key,
            normalize_embeddings=config.normalize_embeddings,
            embed_batch_size=config.embed_batch_size,
        )

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embedding_client.embed_query(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embedding_client.embed_text(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embedding_client.embed_texts(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await self._embedding_client.aembed_query(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await self._embedding_client.aembed_text(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return await self._embedding_client.aembed_texts(texts)
