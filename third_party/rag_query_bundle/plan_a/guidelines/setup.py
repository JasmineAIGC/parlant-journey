"""
setup.py — Guidelines 配置层（GL）。

对应架构图中的 ``GL[Guidelines]`` 模块。

职责：
  为已有的 Parlant Agent 创建并配置两条 RAG Guidelines：

  - **GL 1 — Scoped Retriever（主力）**：
    Guideline 匹配时 Parlant 引擎自动触发知识库检索，
    将结果注入 Agent 上下文，无需 Agent 显式调用工具。

  - **GL 2 — RAG Tool（辅助）**：
    Agent 在判断需要时主动调用 ``search_knowledge`` 工具，
    适合用户明确要求检索或多步推理场景。

设计原则：
  此模块只负责 Guidelines 配置，**不负责创建 Agent**。
  Agent 由外部入口脚本（如 ``examples/05_plan_a_agent.py``）创建后传入。
  这保持了 RAG 功能与 Agent 生命周期管理的清晰分离。

使用方式
--------
::

    # 外部入口脚本（Agent 是外部入口，不在 RAG 模块内）
    import parlant.sdk as p
    from plan_a import PlanAConfig, RAGPipeline
    from plan_a.guidelines import setup_guidelines
    from plan_a.runner import run_parlant_server

    config = PlanAConfig.from_env()
    pipeline = RAGPipeline(config)
    await pipeline.initialize()

    # 轻量用法（无 Server，后台执行）
    from plan_a.runner import run_parlant
    results = await run_parlant(pipeline, ["用户问题"], llm=llm)

    # 完整 Server 用法
    agent_info, results = await run_parlant_server(pipeline, ["用户问题"], llm=llm)
"""
from __future__ import annotations

from typing import Optional, Tuple

import parlant.sdk as p

from plan_a.rag import RAGPipeline
from plan_a.retrievers import make_scoped_retriever
from plan_a.tools import make_rag_tool
from utils.logging import get_logger


logger = get_logger("plan_a.guidelines.setup")


# ---------------------------------------------------------------------------
# 默认 Guideline 配置
# ---------------------------------------------------------------------------

_SCOPED_RETRIEVER_CONDITION = (
    "用户提问涉及知识库内容，例如政策、产品信息、"
    "会员权益、退货规则、服务说明或其他业务知识"
)
_SCOPED_RETRIEVER_ACTION = (
    "自动检索知识库，将相关内容作为上下文，基于该内容给出准确回答；"
    "若检索内容不足，说明知识库暂无相关信息"
)

_RAG_TOOL_CONDITION = (
    "用户明确要求检索、搜索或查询知识库内容，"
    "例如：'帮我查…'、'搜索一下…'、'查找关于…的信息'"
)
_RAG_TOOL_ACTION = (
    "调用知识库检索工具精准搜索相关文档，"
    "并基于检索结果给出详细、有来源依据的回答"
)


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

async def setup_guidelines(
    agent: p.Agent,
    pipeline: RAGPipeline,
    scoped_retriever_condition: Optional[str] = None,
    scoped_retriever_action: Optional[str] = None,
    rag_tool_condition: Optional[str] = None,
    rag_tool_action: Optional[str] = None,
    top_k: Optional[int] = None,
    include_metadata: bool = False,
) -> Tuple[p.Guideline, p.Guideline]:
    """
    为 Agent 创建并配置 RAG 相关的 Guidelines（GL 层）。

    Args:
        agent:
            已创建的 Parlant Agent 实例（由外部入口脚本创建后传入）。
        pipeline:
            已初始化的 :class:`~plan_a.rag.pipeline.RAGPipeline` 实例。
        scoped_retriever_condition:
            Scoped Retriever Guideline 的触发条件（``None`` 使用内置默认值）。
        scoped_retriever_action:
            Scoped Retriever Guideline 的执行动作（``None`` 使用内置默认值）。
        rag_tool_condition:
            RAG Tool Guideline 的触发条件（``None`` 使用内置默认值）。
        rag_tool_action:
            RAG Tool Guideline 的执行动作（``None`` 使用内置默认值）。
        top_k:
            两种检索方式的最大返回片段数；``None`` 使用各自组件的默认值。
        include_metadata:
            是否在检索结果中包含文档元数据（默认 False）。

    Returns:
        ``(scoped_guideline, rag_tool_guideline)`` — 创建好的两条 Guidelines。
    """
    # --- GL 1: Scoped Retriever（主力，自动检索）---
    sr_condition = scoped_retriever_condition or _SCOPED_RETRIEVER_CONDITION
    sr_action = scoped_retriever_action or _SCOPED_RETRIEVER_ACTION

    logger.info("[Guidelines] 创建 Scoped Retriever Guideline")
    scoped_guideline = await agent.create_guideline(
        condition=sr_condition,
        action=sr_action,
    )
    retriever_fn = make_scoped_retriever(pipeline, top_k=top_k, include_metadata=include_metadata)
    await scoped_guideline.attach_retriever(retriever_fn)
    logger.info(f"[Guidelines] Scoped Retriever 就绪  id={scoped_guideline.id}")

    # --- GL 2: RAG Tool（辅助，手动检索）---
    tool_condition = rag_tool_condition or _RAG_TOOL_CONDITION
    tool_action = rag_tool_action or _RAG_TOOL_ACTION

    logger.info("[Guidelines] 创建 RAG Tool Guideline")
    search_knowledge = make_rag_tool(pipeline, top_k=top_k, include_metadata=include_metadata)
    rag_tool_guideline = await agent.create_guideline(
        condition=tool_condition,
        action=tool_action,
        tools=[search_knowledge],
    )
    logger.info(f"[Guidelines] RAG Tool 就绪  id={rag_tool_guideline.id}")

    return scoped_guideline, rag_tool_guideline
