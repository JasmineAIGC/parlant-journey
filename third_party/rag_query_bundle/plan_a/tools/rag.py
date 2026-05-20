"""
rag.py — RAG Tool（手动检索工具）组件。

对应架构图中的 ``Tool[RAG Tool]`` 模块。

工作流程::

    Guideline 触发
        │
        ▼ Agent 判断需要检索，主动调用
    search_knowledge(context, query)
        │
        ▼ RAGPipeline.retrieve(query)
    ToolResult（作为工具结果注入 Agent 上下文）

这是**辅助**检索方式：Agent 主动调用，控制更精准，
适合用户明确要求检索或多步推理场景。

使用方式
--------
::

    from plan_a.rag import RAGPipeline
    from plan_a.tools import make_rag_tool

    pipeline = RAGPipeline()
    search_knowledge = make_rag_tool(pipeline)

    await agent.create_guideline(
        condition="用户明确要求查询知识库",
        action="调用知识库检索工具后回答",
        tools=[search_knowledge],
    )
"""
import parlant.sdk as p
from parlant.sdk import ToolContext, ToolResult

from plan_a.core import RetrievalRequest
from plan_a.rag import RAGPipeline
from utils.logging import get_logger


logger = get_logger("plan_a.tools.rag")


def make_rag_tool(
    pipeline: RAGPipeline,
    top_k: int = None,
    include_metadata: bool = False,
):
    """
    工厂函数：创建绑定到 ``pipeline`` 的 Parlant RAG Tool。

    返回值是 ``@p.tool`` 装饰的 ``ToolEntry``，
    可直接传入 ``agent.create_guideline(tools=[...])``。

    Args:
        pipeline:         :class:`~plan_a.rag.pipeline.RAGPipeline` 实例（懒初始化安全）。
        top_k:            检索返回的最大片段数；``None`` 表示使用 pipeline 默认值。
        include_metadata: 是否在返回结果中包含文档元数据（默认 False）。

    Returns:
        ``parlant.sdk.ToolEntry`` — 已注册的 Parlant Tool 对象。
    """

    @p.tool
    async def search_knowledge(context: ToolContext, query: str) -> ToolResult:
        """
        在知识库中检索与 query 相关的文档片段。

        Args:
            query: 要检索的问题或关键词。

        Returns:
            检索到的知识库内容，按相关性排序。
        """
        logger.info(f"[RagTool] search_knowledge 调用  query={query!r}  top_k={top_k}")

        request = RetrievalRequest(query=query, top_k=top_k, include_metadata=include_metadata)
        response = await pipeline.retrieve_standard(request)
        chunks = response.chunks

        if not chunks:
            logger.info("[RagTool] 未命中相关内容")
            return p.ToolResult(
                data={
                    "request": request.to_dict(),
                    "response": {
                        **response.to_dict(include_metadata=False),
                        "found": False,
                        "message": "知识库中未找到与该问题相关的内容。",
                    },
                }
            )

        formatted = "\n\n".join(
            f"[{i + 1}] 来源: {c.source}\n相关度: {c.score:.3f}\n{c.content}"
            for i, c in enumerate(chunks)
        )

        logger.info(f"[RagTool] 返回 {len(chunks)} 个片段")
        return p.ToolResult(
            data={
                "request": request.to_dict(),
                "response": {
                    **response.to_dict(include_metadata=request.include_metadata),
                    "found": True,
                    "formatted": formatted,
                },
            }
        )

    return search_knowledge
