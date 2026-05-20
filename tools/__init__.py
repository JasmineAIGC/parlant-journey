"""
tools
~~~~~

parlant-journey 工具库 — 按功能分子包组织的可复用工具代码。

子包一览
--------
``tools.llm``            — 基于 LiteLLM 的通用 LLM API 客户端与环境适配工具
``tools.embedding``      — OpenAI 兼容 Embedding 通用客户端与 LlamaIndex 适配层
``tools.parlant_server`` — Parlant Server 的程序化上下文管理器

快速导入::

    from tools.llm import LLMClient, LLMConfig
    from tools.embedding import OpenAIEmbeddingClient, OpenAILikeEmbedding
    from tools.parlant_server import ParlantServerConfig, ParlantServerContext
"""
