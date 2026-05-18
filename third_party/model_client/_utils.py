"""
model_client 内部共享工具函数。

提供：
  - :func:`first_env` — 按优先级读取环境变量
  - :func:`make_openai_clients` — 创建 OpenAI 同步/异步客户端对

``api.llm`` 和 ``api.embedding`` 均通过此模块获得 OpenAI 兼容 API 连接能力。
"""
from __future__ import annotations

import os
from typing import Any, Optional, Sequence, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI as _OpenAI, AsyncOpenAI as _AsyncOpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    _OpenAI = None      # type: ignore[assignment]
    _AsyncOpenAI = None # type: ignore[assignment]


def first_env(keys: Sequence[str], default: Optional[str] = None) -> Optional[str]:
    """按优先级读取环境变量，返回首个非空值。"""
    for key in keys:
        if value := os.getenv(key):
            return value
    return default


def make_openai_clients(
    base_url: str,
    api_key: Optional[str],
    access_key: Optional[str] = None,
) -> Tuple[Any, Any]:
    """返回 ``(sync_client, async_client)``，均已配置连接参数。

    Args:
        base_url:   OpenAI 兼容 API 的 base URL。
        api_key:    API Key；留空时使用 ``"EMPTY"``（本地部署惯例）。
        access_key: 可选的 ``access-key`` 请求头。

    Returns:
        ``(openai.OpenAI, openai.AsyncOpenAI)`` 元组。
    """
    if not _OPENAI_AVAILABLE:
        raise ImportError("make_openai_clients 需要安装 openai SDK：pip install openai>=1.0.0")
    headers = {"access-key": access_key} if access_key else {}
    kw: dict = dict(
        api_key=api_key or "EMPTY",
        base_url=base_url,
        default_headers=headers or None,
    )
    return _OpenAI(**kw), _AsyncOpenAI(**kw)
