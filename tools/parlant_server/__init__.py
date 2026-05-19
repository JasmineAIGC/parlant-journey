"""
tools.parlant_server
~~~~~~~~~~~~~~~~~~~~

无 UI 的 Parlant Server 上下文管理器。

使用 Parlant 框架跑全流程时，默认会启动 HTTP Server 并打印 UI 链接。
此子包通过哨兵异常技巧抑制 UI 弹出，实现后台程序化运行。

快速导入::

    from tools.parlant_server import ParlantServerContext
    from journey import create_journey, JourneyConfig

    async with ParlantServerContext() as ctx:
        agent = await ctx.create_agent("my-agent", "示例 Agent")
        await create_journey(agent, config)
"""
from .server import ParlantServerContext

__all__ = ["ParlantServerContext"]
