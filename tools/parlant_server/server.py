"""
tools.parlant_server.server
~~~~~~~~~~~~~~~~~~~~~~~~~~~

无 UI 的 Parlant Server 上下文管理器。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class _ParlantSkipServerStart(Exception):
    """哨兵异常：令 p.Server.__aexit__ 走清理分支，跳过 uvicorn 启动和 UI 链接打印。"""


@dataclass
class ParlantServerConfig:
    """Parlant Server 程序化运行配置。"""

    nlp_service: Any = None
    port: Optional[int] = None
    prepare_environment: Optional[Callable[[], Any]] = None
    server_factory: Optional[Callable[..., Any]] = None
    server_kwargs: dict[str, Any] = field(default_factory=dict)


class ParlantServerContext:
    """
    无 UI 的 Parlant Server 上下文管理器。

    通过哨兵异常绕过 uvicorn HTTP Server 启动，适用于程序化运行场景。
    """

    def __init__(
        self,
        nlp_service: Any = None,
        bridge_env: bool = True,
        *,
        config: Optional[ParlantServerConfig] = None,
        port: Optional[int] = None,
        prepare_environment: Optional[Callable[[], Any]] = None,
        server_factory: Optional[Callable[..., Any]] = None,
        server_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        if config is None:
            if prepare_environment is None and bridge_env:
                prepare_environment = _default_prepare_environment
            config = ParlantServerConfig(
                nlp_service=nlp_service,
                port=port,
                prepare_environment=prepare_environment,
                server_factory=server_factory,
                server_kwargs=dict(server_kwargs or {}),
            )
        self._config = config
        self._server_ctx: Any = None
        self._server: Any = None

    async def __aenter__(self) -> "ParlantServerContext":
        p = _import_parlant_sdk()

        if self._config.prepare_environment is not None:
            self._config.prepare_environment()

        nlp_service = self._config.nlp_service
        if nlp_service is None:
            nlp_service = p.NLPServices.litellm

        server_factory = self._config.server_factory or p.Server
        server_kwargs = dict(self._config.server_kwargs)
        if self._config.port is not None:
            server_kwargs.setdefault("port", self._config.port)

        logger.info(
            "[ParlantServerContext] 启动 Parlant Server（无 UI）  nlp=%s  port=%s",
            nlp_service,
            server_kwargs.get("port", "default"),
        )
        self._server_ctx = server_factory(nlp_service=nlp_service, **server_kwargs)
        self._server = await self._server_ctx.__aenter__()
        logger.info("[ParlantServerContext] Server 就绪")
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if self._server_ctx is not None:
            server_ctx = self._server_ctx
            self._server_ctx = None
            self._server = None
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

    async def create_agent(
        self,
        name: Optional[str] = None,
        description: str = "",
        **kwargs: Any,
    ) -> Any:
        """在当前 Server 上创建一个 Parlant Agent。"""
        if name is not None:
            kwargs.setdefault("name", name)
        if description:
            kwargs.setdefault("description", description)
        if not kwargs.get("name"):
            raise ValueError("create_agent() 需要提供 name。")
        agent = await self.server.create_agent(**kwargs)
        logger.info("[ParlantServerContext] Agent 已创建  name=%s  id=%s", kwargs["name"], agent.id)
        return agent


def _default_prepare_environment() -> Any:
    from tools.llm import prepare_litellm_environment

    return prepare_litellm_environment()


def _import_parlant_sdk() -> Any:
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
    return p
