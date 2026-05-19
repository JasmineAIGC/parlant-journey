"""
tools.parlant_server.server
~~~~~~~~~~~~~~~~~~~~~~~~~~~

无 UI 的 Parlant Server 上下文管理器。

使用 Parlant 框架跑全流程时，默认会启动 HTTP Server 并打印 UI 链接。
本模块通过"哨兵异常"技巧，在初始化完成后立即退出 Server 启动序列，
实现**后台程序化运行，不弹出 UI 界面**。

用法::

    import asyncio
    import parlant.sdk as p
    from tools.parlant_server import ParlantServerContext
    from journey import create_journey, JourneyConfig

    async def main():
        config = JourneyConfig(title="示例流程", ...)
        async with ParlantServerContext() as ctx:
            agent = await ctx.create_agent("my-agent", "示例 Agent")
            await create_journey(agent, config)
            # 在此处理业务逻辑……

    asyncio.run(main())
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 内部哨兵：跳过 Parlant HTTP Server 启动
# ---------------------------------------------------------------------------

class _ParlantSkipServerStart(Exception):
    """
    哨兵异常：以程序化方式退出 p.Server 上下文时传入，
    令 p.Server.__aexit__ 走"有异常"分支（仅清理资源），
    跳过实体评估（~20s）、uvicorn HTTP Server 启动和 UI 链接打印。
    """


# ---------------------------------------------------------------------------
# ParlantServerContext
# ---------------------------------------------------------------------------

class ParlantServerContext:
    """
    无 UI 的 Parlant Server 上下文管理器。

    **启动时初始化一次**，多轮业务逻辑复用。与直接使用 ``p.Server`` 相比，
    此类会在 ``__aexit__`` 时通过哨兵异常绕过 uvicorn HTTP Server 启动，
    程序不会打开浏览器、弹出 UI 界面，也不会打印 Sandbox UI 链接。

    典型用法::

        async with ParlantServerContext() as ctx:
            agent = await ctx.create_agent("fraud-agent", "欺诈处理 Agent")
            await register_journeys(agent, [fraud_config, loan_config])
            # 在此处理对话或测试逻辑

    若需要直接访问底层 ``p.Server`` 对象（例如调用 SDK 高级 API），
    可通过 ``ctx.server`` 属性获取。

    Args:
        nlp_service:   NLP 服务类型（默认 ``p.NLPServices.litellm``）。
                       传入 ``None`` 时使用 Parlant 默认值。
        bridge_env:    是否自动桥接 LiteLLM 所需的环境变量（默认 ``True``）。
                       通过 ``model_client._utils.bridge_parlant_litellm_env`` 实现；
                       若未安装 model_client，此选项被静默忽略。

    Raises:
        RuntimeError: 若未安装 ``parlant``（pip install parlant）。
        RuntimeError: 若未安装 ``torch``（Parlant LiteLLM 适配器依赖）。
    """

    def __init__(
        self,
        nlp_service: Any = None,
        bridge_env: bool = True,
    ) -> None:
        self._nlp_service = nlp_service
        self._bridge_env = bridge_env
        self._server_ctx: Any = None
        self._server: Any = None

    async def __aenter__(self) -> "ParlantServerContext":
        try:
            import parlant.sdk as p  # type: ignore[import]
        except ModuleNotFoundError as exc:
            if getattr(exc, "name", None) == "torch" or "No module named 'torch'" in str(exc):
                raise RuntimeError(
                    "Parlant 依赖 torch，请先安装：pip install torch"
                ) from exc
            raise RuntimeError(
                "未安装 parlant，请先安装：pip install parlant"
            ) from exc

        if self._bridge_env:
            try:
                from model_client._utils import bridge_parlant_litellm_env  # type: ignore[import]
                bridge_parlant_litellm_env()
            except ImportError:
                pass  # model_client 未安装，静默忽略

        nlp_service = self._nlp_service
        if nlp_service is None:
            nlp_service = p.NLPServices.litellm

        logger.info("[ParlantServerContext] 启动 Parlant Server（无 UI）  nlp=%s", nlp_service)
        self._server_ctx = p.Server(nlp_service=nlp_service)
        self._server = await self._server_ctx.__aenter__()
        logger.info("[ParlantServerContext] Server 就绪")
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if self._server_ctx is not None:
            server_ctx = self._server_ctx
            self._server_ctx = None
            self._server = None
            # 传入哨兵异常：令 p.Server.__aexit__ 走"有异常"分支（只做资源清理），
            # 跳过 _process_evaluations()、uvicorn 启动和 UI 链接打印。
            _sentinel = _ParlantSkipServerStart()
            await server_ctx.__aexit__(type(_sentinel), _sentinel, None)
            logger.info("[ParlantServerContext] Server 已关闭")
        return False

    @property
    def server(self) -> Any:
        """底层 Parlant Server 对象（``p.Server.__aenter__`` 返回的实例）。"""
        if self._server is None:
            raise RuntimeError(
                "ParlantServerContext 尚未进入上下文，请在 async with 块内使用。"
            )
        return self._server

    async def create_agent(self, name: str, description: str = "") -> Any:
        """在当前 Server 上创建一个 Parlant Agent。

        Args:
            name:        Agent 唯一名称。
            description: Agent 描述（可选）。

        Returns:
            创建完成的 Parlant ``Agent`` 对象。
        """
        agent = await self.server.create_agent(name=name, description=description)
        logger.info("[ParlantServerContext] Agent 已创建  name=%s  id=%s", name, agent.id)
        return agent
