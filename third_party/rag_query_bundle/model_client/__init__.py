"""
model_client — 统一模型调用模块。

将模型调用按运行方式分为两个独立子包：

  ``model_client.api``   — 通过 OpenAI 兼容 API 调用远程模型
  ``model_client.local`` — 在本地直接运行模型（无需 API 服务）

快速导入::

    # API 模式（LLM + Embedding）
    from model_client.api import LLMClient, LLMConfig, OpenAILikeEmbedding

    # 本地模式（Embedding）
    from model_client.local import SentenceTransformerEmbedding

    # 或直接从顶层导入
    from model_client import LLMClient, LLMConfig, OpenAILikeEmbedding, SentenceTransformerEmbedding
"""
from model_client.api.llm import LLMConfig, LLMClient
from model_client.api.embedding import OpenAILikeEmbedding
from model_client.local.embedding import SentenceTransformerEmbedding

__all__ = [
    # API 模式
    "LLMConfig",
    "LLMClient",
    "OpenAILikeEmbedding",
    # 本地模式
    "SentenceTransformerEmbedding",
]
