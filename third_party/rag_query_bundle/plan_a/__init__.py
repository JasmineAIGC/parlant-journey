"""
plan_a — 基于 Parlant 框架的极简版 RAG 方案。

目录结构按 docs/planA.md 的模块关系组织：

    User ←→ Agent[Parlant Agent]   ← 外部入口（examples/05_plan_a_agent.py）
                     │
                GL[Guidelines]      ← plan_a/guidelines/
               ╱               ╲
    SR[Scoped Retriever]   Tool[RAG Tool]   ← plan_a/retrievers/ / plan_a/tools/
               ╲               ╱
              RAG[RAG 服务]        ← plan_a/rag/
                     │
              VDB[(向量数据库)]    ← plan_a/vdb/

其中 core/ 提供跨模块复用的配置与共享类型。

RAG 方式（RAGMode）
------------------
plan_a 支持两种 RAG 实现方式，可通过 :class:`~plan_a.core.types.RAGMode` 枚举选择：

* ``RAGMode.STANDALONE``：不依赖 Parlant，直接使用 :class:`RAGPipeline` 检索，
  适合脚本测试或轻量集成。

* ``RAGMode.PARLANT``：依赖 Parlant 框架，通过 SR[Scoped Retriever]、
  GL[Guidelines]、Tool[RAG Tool] 三件套接入 Agent 对话场景。
"""
from plan_a.core import DocChunk, PlanAConfig, RAGMode, RetrievalRequest, RetrievalResponse
from plan_a.rag import Embedder, RAGPipeline
from plan_a.vdb import VectorStore

__all__ = [
    # 配置 & 类型
    "PlanAConfig",
    "DocChunk",
    "RAGMode",
    "RetrievalRequest",
    "RetrievalResponse",
    # 基础组件
    "Embedder",
    "VectorStore",
    # RAG 管道（核心）
    "RAGPipeline",
    # Parlant 集成（懒加载，需安装 parlant）
    "make_scoped_retriever",
    "make_rag_tool",
    "setup_guidelines",
    # Runner（两种模式的 query-answer 入口）
    "load_test_queries",
    "run_standalone",
    "run_parlant",
    "run_parlant_server",
    "ParlantServerContext",
]


def make_scoped_retriever(*args, **kwargs):
    from plan_a.retrievers import make_scoped_retriever as _impl

    return _impl(*args, **kwargs)


def make_rag_tool(*args, **kwargs):
    from plan_a.tools import make_rag_tool as _impl

    return _impl(*args, **kwargs)


def setup_guidelines(*args, **kwargs):
    from plan_a.guidelines import setup_guidelines as _impl

    return _impl(*args, **kwargs)


def load_test_queries(*args, **kwargs):
    from plan_a.runner import load_test_queries as _impl

    return _impl(*args, **kwargs)


async def run_standalone(*args, **kwargs):
    from plan_a.runner import run_standalone as _impl

    return await _impl(*args, **kwargs)


async def run_parlant(*args, **kwargs):
    from plan_a.runner import run_parlant as _impl

    return await _impl(*args, **kwargs)


async def run_parlant_server(*args, **kwargs):
    from plan_a.runner import run_parlant_server as _impl

    return await _impl(*args, **kwargs)


class ParlantServerContext:
    """懒加载代理：将构造延迟到 plan_a.runner.ParlantServerContext 实现。"""

    def __init__(self, *args, **kwargs):
        from plan_a.runner import ParlantServerContext as _impl
        self._impl = _impl(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._impl, name)

    async def __aenter__(self):
        return await self._impl.__aenter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self._impl.__aexit__(exc_type, exc_val, exc_tb)
