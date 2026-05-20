"""
tools.parlant_server.server
~~~~~~~~~~~~~~~~~~~~~~~~~~~

无 UI 的 Parlant Server 上下文管理器。

用法::

    async with ParlantServerContext() as ctx:
        agent = await ctx.create_agent("my-agent", "描述")
        # 在此处理业务逻辑
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
    """哨兵异常：令 p.Server.__aexit__ 走清理分支，跳过 uvicorn 启动和 UI 链接打印。"""


# ---------------------------------------------------------------------------
# ParlantServerContext
# ---------------------------------------------------------------------------

class ParlantServerContext:
    """
    无 UI 的 Parlant Server 上下文管理器。

    通过哨兵异常绕过 uvicorn HTTP Server 启动，适用于程序化运行场景。

    用法::

        async with ParlantServerContext() as ctx:
            agent = await ctx.create_agent("fraud-agent", "欺诈处理 Agent")
            # 在此处理对话或测试逻辑
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
                from tools.llm.client import bridge_parlant_litellm_env
                bridge_parlant_litellm_env()
            except ImportError:
                pass

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
        """在当前 Server 上创建一个 Parlant Agent。"""
        agent = await self.server.create_agent(name=name, description=description)
        logger.info("[ParlantServerContext] Agent 已创建  name=%s  id=%s", name, agent.id)
        return agent
