"""
tools.embedding.client
~~~~~~~~~~~~~~~~~~~~~~

OpenAI 兼容 Embedding 客户端 — 通过 API 调用远程 Embedding 服务。

使用 ``openai.AsyncOpenAI`` SDK 调用 vLLM / OpenAI 兼容的 ``/v1/embeddings`` 端点，
同时支持同步和异步两种调用方式，以及私有部署网关的 ``access-key`` 请求头。

继承自 LlamaIndex ``BaseEmbedding``，与知识库构建阶段使用完全相同的 Embedding 接口，
保证向量维度一致。

依赖::

    pip install llama-index openai
"""
from __future__ import annotations

import os
from typing import Any, List, Optional

try:
    from pydantic import PrivateAttr
    from llama_index.core.embeddings import BaseEmbedding as _BaseEmbedding
    _LLAMA_AVAILABLE = True
except ImportError:
    _LLAMA_AVAILABLE = False
    _BaseEmbedding = object  # type: ignore[assignment,misc]
    PrivateAttr = lambda **kw: None  # type: ignore[assignment]  # noqa: E731


def _first_env(keys: list, default: Optional[str] = None) -> Optional[str]:
    """按优先级读取环境变量，返回首个非空值。"""
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return default


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
            "OpenAILikeEmbedding 需要安装 openai SDK：pip install openai>=1.0.0"
        ) from exc
    headers = {"access-key": access_key} if access_key else {}
    kw: dict = dict(
        api_key=api_key or "EMPTY",
        base_url=base_url,
        default_headers=headers or None,
    )
    return OpenAI(**kw), AsyncOpenAI(**kw)


class OpenAILikeEmbedding(_BaseEmbedding):
    """
    最小化 OpenAI 兼容 Embedding 客户端，继承自 LlamaIndex ``BaseEmbedding``。

    使用 ``openai.OpenAI`` / ``openai.AsyncOpenAI`` SDK 调用 vLLM / OpenAI 兼容的
    ``/v1/embeddings`` 端点，支持同步和异步调用，以及私有部署网关的 ``access-key`` 请求头。

    Args:
        base_url:             Embedding 服务的 base URL（如 ``http://127.0.0.1:8018/v1``）。
        model_name:           Embedding 模型名称（如 ``bge-large-zh-v1.5``）。
        api_key:              API Key；本地部署时可设为任意非空字符串（默认 ``"EMPTY"``）。
        access_key:           可选的 ``access-key`` 请求头（部分私有部署网关需要）。
        normalize_embeddings: 是否对返回向量做 L2 归一化（默认 ``True``）。
                              归一化后向量为单位向量，与 ``IndexFlatIP`` 配合时
                              内积等价于余弦相似度，排名不受向量模长干扰。
                              **注意**：构建知识库与检索查询必须使用相同的设置；
                              修改此参数后需重建知识库索引。
    """
    base_url: str
    model_name: str
    api_key: Optional[str] = None
    access_key: Optional[str] = None
    normalize_embeddings: bool = True
    embed_batch_size: int = 5

    _client: Any = PrivateAttr()
    _aclient: Any = PrivateAttr()

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
        self._client, self._aclient = _make_openai_clients(base_url, api_key, access_key)

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "OpenAILikeEmbedding":
        """
        从环境变量（或 ``.env`` 文件）创建实例：

        ================  ==========================================
        参数               读取的环境变量
        ================  ==========================================
        base_url          EMBEDDING_BASE_URL / QWEN3_BASE_URL / LLM_BASE_URL
        api_key           EMBEDDING_API_KEY / QWEN3_API_KEY / LLM_API_KEY
        access_key        EMBEDDING_ACCESS_KEY / QWEN3_ACCESS_KEY /
                          LLM_ACCESS_KEY / ACCESS_KEY
        model_name        EMBEDDING_MODEL / QWEN3_MODEL / LLM_MODEL
        ================  ==========================================
        """
        try:
            from dotenv import load_dotenv  # type: ignore[import]
            load_dotenv()
        except ImportError:
            pass

        base_url = _first_env(
            ["EMBEDDING_BASE_URL", "QWEN3_BASE_URL", "LLM_BASE_URL"],
            default="http://127.0.0.1:8018/v1",
        )
        api_key = _first_env(["EMBEDDING_API_KEY", "QWEN3_API_KEY", "LLM_API_KEY"])
        access_key = _first_env(
            ["EMBEDDING_ACCESS_KEY", "QWEN3_ACCESS_KEY", "LLM_ACCESS_KEY", "ACCESS_KEY"]
        )
        model = _first_env(
            ["EMBEDDING_MODEL", "QWEN3_MODEL", "LLM_MODEL"],
            default="bge-large-zh-v1.5",
        )
        return cls(base_url=base_url, model_name=model, api_key=api_key, access_key=access_key)

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _maybe_normalize(self, embeddings: List[List[float]]) -> List[List[float]]:
        """若 normalize_embeddings=True，对每个向量做 L2 归一化。"""
        if not self.normalize_embeddings:
            return embeddings
        import numpy as np
        vecs = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (vecs / norms).tolist()

    # ------------------------------------------------------------------
    # 同步方法
    # ------------------------------------------------------------------

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embeddings([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embeddings([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        response = self._client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        embeddings = [item.embedding for item in response.data]
        return self._maybe_normalize(embeddings)

    # ------------------------------------------------------------------
    # 异步方法
    # ------------------------------------------------------------------

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return (await self._aget_text_embeddings([query]))[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return (await self._aget_text_embeddings([text]))[0]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        response = await self._aclient.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        embeddings = [item.embedding for item in response.data]
        return self._maybe_normalize(embeddings)
