"""
tools.embedding
~~~~~~~~~~~~~~~

OpenAI 兼容 Embedding 客户端。

快速导入::

    from tools.embedding import OpenAILikeEmbedding

    embed = OpenAILikeEmbedding.from_env()
    vectors = embed._get_text_embeddings(["文本一", "文本二"])
"""
from .client import OpenAILikeEmbedding

__all__ = ["OpenAILikeEmbedding"]
