"""
model_client 内部共享工具函数。

提供：
  - :func:`first_env` — 按优先级读取环境变量
  - :func:`make_openai_clients` — 创建 OpenAI 同步/异步客户端对

``api.llm`` 和 ``api.embedding`` 均通过此模块获得 OpenAI 兼容 API 连接能力。
"""
from __future__ import annotations

import os
from typing import Any, Optional, Sequence, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI as _OpenAI, AsyncOpenAI as _AsyncOpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    _OpenAI = None      # type: ignore[assignment]
    _AsyncOpenAI = None # type: ignore[assignment]


_LITELLM_PROVIDER_PREFIXES = (
    "openai/", "anthropic/", "azure/", "gemini/", "google/",
    "mistral/", "cohere/", "huggingface/", "deepseek/", "ollama/",
    "bedrock/", "vertex_ai/", "groq/", "together_ai/", "replicate/",
    "xinference/", "fireworks_ai/",
)


def first_env(keys: Sequence[str], default: Optional[str] = None) -> Optional[str]:
    """按优先级读取环境变量，返回首个非空值。"""
    for key in keys:
        if value := os.getenv(key):
            return value
    return default


def _normalize_litellm_model_name(model: str) -> str:
    """若模型名未显式携带 provider 前缀，则补齐 ``openai/`` 前缀。"""
    if any(model.startswith(prefix) for prefix in _LITELLM_PROVIDER_PREFIXES):
        return model
    return f"openai/{model}"


def bridge_parlant_litellm_env() -> dict[str, str]:
    """
    用现有 `.env` 中的 LLM 配置补齐 Parlant LiteLLM 所需环境变量。

    优先保留用户显式设置的 ``LITELLM_*``，仅在缺失时回退到当前仓库
    已使用的 ``QWEN3_*`` / ``LLM_*`` / ``OPENAI_*`` 等命名。

    Returns:
        已解析出的 ``LITELLM_*`` 键值对。
    """
    resolved: dict[str, str] = {}

    if base_url := os.getenv("LITELLM_PROVIDER_BASE_URL") or first_env([
        "QWEN3_BASE_URL", "LLM_BASE_URL", "CONTEXTMIND_BASE_URL",
        "OPENAI_BASE_URL", "DEEPSEEK_BASE_URL", "QWEN_BASE_URL",
    ]):
        os.environ.setdefault("LITELLM_PROVIDER_BASE_URL", base_url)
        resolved["LITELLM_PROVIDER_BASE_URL"] = os.environ["LITELLM_PROVIDER_BASE_URL"]

    if api_key := os.getenv("LITELLM_PROVIDER_API_KEY") or first_env([
        "QWEN3_API_KEY", "LLM_API_KEY", "CONTEXTMIND_API_KEY",
        "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "QWEN_API_KEY",
    ]):
        os.environ.setdefault("LITELLM_PROVIDER_API_KEY", api_key)
        resolved["LITELLM_PROVIDER_API_KEY"] = os.environ["LITELLM_PROVIDER_API_KEY"]

    if model := os.getenv("LITELLM_PROVIDER_MODEL_NAME") or first_env([
        "QWEN3_MODEL", "LLM_MODEL", "CONTEXTMIND_MODEL",
        "OPENAI_MODEL", "DEEPSEEK_MODEL", "QWEN_MODEL",
    ]):
        normalized_model = _normalize_litellm_model_name(model)
        os.environ.setdefault("LITELLM_PROVIDER_MODEL_NAME", normalized_model)
        resolved["LITELLM_PROVIDER_MODEL_NAME"] = os.environ["LITELLM_PROVIDER_MODEL_NAME"]

    if embedding_model := os.getenv("LITELLM_EMBEDDING_MODEL_NAME") or first_env([
        "EMBEDDING_MODEL", "QWEN3_EMBEDDING_MODEL", "LLM_EMBEDDING_MODEL",
    ]):
        normalized_embedding_model = _normalize_litellm_model_name(embedding_model)
        os.environ.setdefault("LITELLM_EMBEDDING_MODEL_NAME", normalized_embedding_model)
        resolved["LITELLM_EMBEDDING_MODEL_NAME"] = os.environ["LITELLM_EMBEDDING_MODEL_NAME"]

    return resolved


def make_openai_clients(
    base_url: str,
    api_key: Optional[str],
    access_key: Optional[str] = None,
) -> Tuple[Any, Any]:
    """返回 ``(sync_client, async_client)``，均已配置连接参数。

    Args:
        base_url:   OpenAI 兼容 API 的 base URL。
        api_key:    API Key；留空时使用 ``"EMPTY"``（本地部署惯例）。
        access_key: 可选的 ``access-key`` 请求头。

    Returns:
        ``(openai.OpenAI, openai.AsyncOpenAI)`` 元组。
    """
    if not _OPENAI_AVAILABLE:
        raise ImportError("make_openai_clients 需要安装 openai SDK：pip install openai>=1.0.0")
    headers = {"access-key": access_key} if access_key else {}
    kw: dict = dict(
        api_key=api_key or "EMPTY",
        base_url=base_url,
        default_headers=headers or None,
    )
    return _OpenAI(**kw), _AsyncOpenAI(**kw)
