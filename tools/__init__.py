"""
tools
~~~~~

parlant-journey 工具库 — 按功能分子包组织的可复用工具代码。

子包一览
--------
``tools.llm``            — 基于 LiteLLM 的 LLM API 客户端（LLMClient / LLMConfig）
``tools.embedding``      — OpenAI 兼容 Embedding 客户端（OpenAILikeEmbedding）
``tools.parlant_server`` — 无 UI 的 Parlant Server 上下文管理器（ParlantServerContext）

快速导入::

    from tools.llm import LLMClient, LLMConfig
    from tools.embedding import OpenAILikeEmbedding
    from tools.parlant_server import ParlantServerContext
"""
