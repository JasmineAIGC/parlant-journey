"""
tools.parlant_server
~~~~~~~~~~~~~~~~~~~~

无 UI 的 Parlant Server 上下文管理器及多轮对话辅助函数。

使用 Parlant 框架跑全流程时，默认会启动 HTTP Server 并打印 UI 链接。
此子包通过哨兵异常技巧抑制 UI 弹出，实现后台程序化运行。

快速导入::

    from tools.parlant_server import ParlantServerConfig, ParlantServerContext, run_conversation
    from journey import create_journey

    async with ParlantServerContext() as ctx:
        agent = await ctx.create_agent("my-agent", "示例 Agent")
        await create_journey(agent, config)
        async for user_msg, reply in run_conversation(ctx, agent, messages):
            print(f"用户: {user_msg}")
            print(f"Agent: {reply}")
"""
from .server import ParlantServerConfig, ParlantServerContext
from .session import run_conversation

__all__ = ["ParlantServerConfig", "ParlantServerContext", "run_conversation"]
