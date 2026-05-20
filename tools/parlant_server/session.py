"""
tools.parlant_server.session
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parlant 多轮对话辅助函数。
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, List, Tuple


async def run_conversation(
    ctx: Any,
    agent: Any,
    messages: List[str],
    timeout: float = 300.0,
    title: str = "对话",
) -> AsyncGenerator[Tuple[str, str], None]:
    """
    驱动用户与 Agent 的多轮对话，逐轮 yield ``(user_msg, agent_reply)``。

    Args:
        ctx:      ParlantServerContext 实例（已进入上下文）。
        agent:    Parlant Agent 对象（由 ``ctx.create_agent()`` 创建）。
        messages: 用户消息列表（每条消息触发一轮 Agent 回复）。
        timeout:  等待 Agent 回复的超时秒数（默认 300s）。
        title:    会话标题。

    Yields:
        ``(user_msg, agent_reply)`` 元组；若超时或无回复则 ``agent_reply`` 为空字符串。
    """
    from parlant.core.app_modules.sessions import SessionModule, Moderation
    from parlant.core.sessions import EventKind, EventSource, Timeout
    from parlant.core.customers import CustomerStore

    server = ctx.server
    session_module = server._container[SessionModule]
    session = await session_module.create(
        customer_id=CustomerStore.GUEST_ID,
        agent_id=agent.id,
        title=title,
    )

    for user_msg in messages:
        evt = await session_module.create_customer_message(
            session_id=session.id,
            moderation=Moderation.NONE,
            message=user_msg,
            source=EventSource.CUSTOMER,
            trigger_processing=True,
            metadata=None,
        )
        next_offset = evt.offset + 1

        try:
            await asyncio.wait_for(
                session_module.wait_for_more_events(
                    session_id=session.id,
                    min_offset=next_offset,
                    kinds=[EventKind.MESSAGE],
                    source=EventSource.AI_AGENT,
                    timeout=Timeout(timeout),
                ),
                timeout=timeout,
            )
        except (asyncio.TimeoutError, TimeoutError):
            yield user_msg, ""
            continue

        agent_events = await session_module.find_events(
            session_id=session.id,
            min_offset=next_offset,
            source=EventSource.AI_AGENT,
            kinds=[EventKind.MESSAGE],
            trace_id=None,
        )

        reply = ""
        for ev in agent_events:
            if isinstance(ev.data, dict) and "message" in ev.data:
                reply = ev.data["message"]
                break

        yield user_msg, reply
