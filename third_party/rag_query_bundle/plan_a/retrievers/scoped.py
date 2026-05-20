"""
scoped.py — Scoped Retriever（自动检索）组件。

对应架构图中的 ``SR[Scoped Retriever]`` 模块。

工作流程::

    Guideline 匹配
        │
        ▼ Parlant 自动调用 RetrieverFunction
    _scoped_retriever(context)
        │
        ▼ RAGPipeline.retrieve(last_customer_message)
    RetrieverResult（注入 Agent 推理上下文）

这是推荐的**主力**检索方式：无需 Agent 显式调用工具，
Guideline 匹配时 Parlant 引擎自动执行检索并将结果注入上下文，
对话体验更流畅、延迟更低。

使用方式
--------
::

    from plan_a.rag import RAGPipeline
    from plan_a.retrievers import make_scoped_retriever

    pipeline = RAGPipeline()
    await pipeline.initialize()

    retriever_fn = make_scoped_retriever(pipeline)

    guideline = await agent.create_guideline(
        condition="用户询问知识库相关问题",
        action="检索知识库后回答",
    )
    await guideline.attach_retriever(retriever_fn)
"""
from __future__ import annotations

from typing import Awaitable, Callable

import parlant.sdk as p

from plan_a.core import RetrievalRequest
from plan_a.rag import RAGPipeline
from utils.logging import get_logger


logger = get_logger("plan_a.retrievers.scoped")


def make_scoped_retriever(
    pipeline: RAGPipeline,
    top_k: int = None,
    include_metadata: bool = False,
) -> Callable[[p.RetrieverContext], Awaitable[p.RetrieverResult | None]]:
    """
    工厂函数：创建绑定到 ``pipeline`` 的 Parlant Scoped Retriever 函数。

    返回值可直接传入 ``guideline.attach_retriever()``。
    Guideline 匹配时，Parlant 引擎自动调用该函数，
    将检索结果注入 Agent 的推理上下文。

    Args:
        pipeline:         :class:`~plan_a.rag.pipeline.RAGPipeline` 实例（懒初始化安全）。
        top_k:            检索返回的最大片段数；``None`` 表示使用 pipeline 默认值。
        include_metadata: 是否在元数据中包含文档元数据（默认 False）。

    Returns:
        符合 ``RetrieverFunction`` 签名的异步函数。
    """

    async def _scoped_retriever(
        context: p.RetrieverContext,
    ) -> p.RetrieverResult | None:
        """Scoped Retriever：Guideline 匹配时自动检索知识库并注入上下文。"""
        last_msg = context.interaction.last_customer_message
        if not last_msg:
            logger.debug("[ScopedRetriever] 无用户消息，跳过检索")
            return None

        request = RetrievalRequest(query=last_msg.content, top_k=top_k, include_metadata=include_metadata)
        response = await pipeline.retrieve_standard(request)
        chunks = response.chunks

        if not chunks:
            logger.info("[ScopedRetriever] 未命中相关内容")
            return None

        # 将文档片段格式化为可读上下文文本，注入 Agent 推理上下文
        context_text = "\n\n".join(
            f"[来源: {c.source}]\n{c.content}" for c in chunks
        )

        logger.info(f"[ScopedRetriever] 返回 {len(chunks)} 个片段")
        return p.RetrieverResult(
            data=context_text,
            metadata={
                "request": request.to_dict(),
                "response": response.to_dict(include_metadata=False),
            },
        )

    return _scoped_retriever
