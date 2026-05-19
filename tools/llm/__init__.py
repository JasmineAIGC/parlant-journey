"""
tools.llm
~~~~~~~~~

基于 LiteLLM 的 LLM API 客户端。

快速导入::

    from tools.llm import LLMClient, LLMConfig

    client = LLMClient()
    reply = await client.chat_async([{"role": "user", "content": "你好"}])
"""
from .client import (
    LLMClient,
    LLMConfig,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
)

__all__ = [
    "LLMClient",
    "LLMConfig",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MAX_TOKENS",
]
