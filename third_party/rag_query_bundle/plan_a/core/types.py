"""
types.py — Plan A 模块共享数据类型。

此文件不依赖任何外部模块，只使用标准库。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


_SCORE_PRECISION = 4
"""score 字段执行 round() 时使用的小数位数。"""


class RAGMode(str, Enum):
    """
    RAG 实现方式选择。

    STANDALONE : 不依赖 Parlant，直接使用 RAGPipeline 检索（适合脚本、测试等轻量场景）。
    PARLANT    : 依赖 Parlant 框架，通过 SR[Scoped Retriever]、GL[Guidelines]、
                 Tool[RAG Tool] 三件套接入 Agent（完整 Agent 对话场景）。
    """

    STANDALONE = "standalone"
    PARLANT = "parlant"


@dataclass
class DocChunk:
    """
    检索到的文档片段。

    由 VectorStore 生成，经 RAGPipeline 返回，
    最终传入 Scoped Retriever / RAG Tool 序列化给 Parlant。
    """

    content: str
    """文档正文内容。"""

    source: str
    """文档来源标识（文件名、URL 等）。"""

    score: float
    """相似度分数（越高越相关）。"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """可选的附加元数据（来自文档库）。"""

    def to_dict(self, include_metadata: bool = False) -> Dict[str, Any]:
        """序列化为字典，兼容 Parlant Tool/Retriever 返回格式。"""
        data: Dict[str, Any] = {
            "content": self.content,
            "source": self.source,
            "score": round(self.score, _SCORE_PRECISION),
        }
        if include_metadata:
            data["metadata"] = self.metadata
        return data

    def __repr__(self) -> str:
        return f"DocChunk(source={self.source!r}, score={self.score:.3f})"


@dataclass
class RetrievalRequest:
    """标准化检索输入。"""

    query: str
    top_k: Optional[int] = None
    include_metadata: bool = False

    def normalized_query(self) -> str:
        return self.query.strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.normalized_query(),
            "top_k": self.top_k,
            "include_metadata": self.include_metadata,
        }


@dataclass
class RetrievalResponse:
    """标准化检索输出。"""

    query: str
    chunks: List[DocChunk]
    top_k: int
    took_ms: int

    def to_dict(self, include_metadata: bool = False) -> Dict[str, Any]:
        return {
            "query": self.query,
            "top_k": self.top_k,
            "chunk_count": len(self.chunks),
            "took_ms": self.took_ms,
            "chunks": [c.to_dict(include_metadata=include_metadata) for c in self.chunks],
        }
