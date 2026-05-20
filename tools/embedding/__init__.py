"""
tools.embedding
~~~~~~~~~~~~~~~

OpenAI 兼容 Embedding 客户端及适配层。

快速导入::

    from tools.embedding import OpenAIEmbeddingClient, OpenAILikeEmbedding

    embed = OpenAIEmbeddingClient.from_env()
    vectors = embed.embed_texts(["文本一", "文本二"])
"""
from .client import OpenAIEmbeddingClient, OpenAIEmbeddingConfig, OpenAILikeEmbedding

__all__ = ["OpenAIEmbeddingClient", "OpenAIEmbeddingConfig", "OpenAILikeEmbedding"]
