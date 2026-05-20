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

# Parlant 默认端口差值：内置 Tool Service 固定为 8818，主 Server 默认 8800，差值为 18
DEFAULT_PARLANT_TOOL_SERVICE_PORT_OFFSET = 18


class _ParlantSkipServerStart(Exception):
    """哨兵异常：令 p.Server.__aexit__ 走清理分支，跳过 uvicorn 启动和 UI 链接打印。"""


@dataclass
class ParlantServerConfig:
    """Parlant Server 程序化运行配置。"""

    nlp_service: Any = None
    port: Optional[int] = None
    tool_service_port: Optional[int] = None
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
        tool_service_port: Optional[int] = None,
        prepare_environment: Optional[Callable[[], Any]] = None,
        server_factory: Optional[Callable[..., Any]] = None,
        server_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """初始化一个无 UI 的 Parlant Server 上下文。

        Args:
            nlp_service: Parlant 使用的 NLP service；为空时默认使用 ``p.NLPServices.litellm``。
            bridge_env: 若为 True 且未显式提供 ``prepare_environment``，则自动桥接项目中的 LLM 环境变量。
            config: 可选的完整 ``ParlantServerConfig``；提供后优先使用。
            port: 显式指定 Parlant Server 监听端口；当同时提供 ``server_kwargs['port']`` 时，以此参数为准。
            tool_service_port: 显式指定内置 Tool Service 监听端口；未提供时，若设置了 ``port``，则自动按默认偏移量派生。
            prepare_environment: 进入上下文前执行的环境准备函数。
            server_factory: 自定义 Server 工厂，默认使用 ``parlant.sdk.Server``。
            server_kwargs: 传递给底层 Server 的额外关键字参数。
        """
        if config is None:
            if prepare_environment is None and bridge_env:
                prepare_environment = _default_prepare_environment
            config = ParlantServerConfig(
                nlp_service=nlp_service,
                port=port,
                tool_service_port=tool_service_port,
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
        port, tool_service_port = self._resolve_ports(server_kwargs)
        if port is not None:
            server_kwargs["port"] = port
        if tool_service_port is not None:
            server_kwargs["tool_service_port"] = tool_service_port

        logger.info(
            "[ParlantServerContext] 启动 Parlant Server（无 UI）  nlp=%s  port=%s  tool_service_port=%s",
            nlp_service,
            server_kwargs.get("port", "default"),
            server_kwargs.get("tool_service_port", "default"),
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

    def _resolve_ports(
        self,
        server_kwargs: dict[str, Any],
    ) -> tuple[Optional[int], Optional[int]]:
        port = self._config.port
        if port is None and "port" in server_kwargs:
            raw_port = server_kwargs["port"]
            port = raw_port if isinstance(raw_port, int) else None

        tool_service_port = self._config.tool_service_port
        if tool_service_port is None and "tool_service_port" in server_kwargs:
            raw_tool_service_port = server_kwargs["tool_service_port"]
            tool_service_port = (
                raw_tool_service_port if isinstance(raw_tool_service_port, int) else None
            )

        if tool_service_port is None and port is not None:
            tool_service_port = port + DEFAULT_PARLANT_TOOL_SERVICE_PORT_OFFSET

        return port, tool_service_port


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
