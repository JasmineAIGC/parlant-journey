"""
model_client.api — 通过 OpenAI 兼容 API 调用远程模型。

提供：
  - :class:`LLMConfig`           — LLM 服务连接配置（从 .env 自动加载）
  - :class:`LLMClient`           — 异步 LLM 客户端（基于 LiteLLM）
  - :class:`OpenAILikeEmbedding` — OpenAI 兼容 Embedding 客户端（基于 openai SDK）

适用场景：LLM 和 Embedding 服务均以 OpenAI 兼容接口对外暴露（vLLM / OpenAI / DeepSeek 等）。
"""
from model_client.api.llm import LLMConfig, LLMClient
from model_client.api.embedding import OpenAILikeEmbedding

__all__ = [
    "LLMConfig",
    "LLMClient",
    "OpenAILikeEmbedding",
]
