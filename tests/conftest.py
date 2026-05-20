"""
tests/conftest.py
~~~~~~~~~~~~~~~~~

pytest 共享配置：

1. 在所有测试模块被收集之前注入 parlant.sdk 桩模块，
   使 journey.* 可在无真实 Parlant 包的环境中被导入。

2. 提供公共 pytest fixture，供 builder / factory 测试共享，
   避免每个测试文件重复定义相同的 mock 工厂代码。
"""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import AsyncMock, MagicMock

# ---------------------------------------------------------------------------
# 确保项目根目录在 sys.path 中，支持在任意工作目录下运行 pytest 或直接执行脚本
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest


# ---------------------------------------------------------------------------
# 注入 parlant.sdk 桩模块（在任何测试文件被导入前执行）
# ---------------------------------------------------------------------------

def _inject_parlant_stub() -> None:
    """在 sys.modules 中注册最小化的 parlant.sdk 桩，模拟真实 SDK 接口。"""
    if "parlant.sdk" in sys.modules:
        return  # 已被其他测试文件注入，跳过

    parlant = types.ModuleType("parlant")
    sdk = types.ModuleType("parlant.sdk")

    class CompositionMode:
        STRICT = "STRICT"
        FLUID = "FLUID"

    sdk.CompositionMode = CompositionMode
    sdk.Agent = object
    sdk.Journey = object
    sdk.Server = object

    parlant.sdk = sdk
    sys.modules.setdefault("parlant", parlant)
    sys.modules.setdefault("parlant.sdk", sdk)


_inject_parlant_stub()


# ---------------------------------------------------------------------------
# 公共 mock fixtures（供 builder 和 factory 测试使用）
# ---------------------------------------------------------------------------

def make_state_mock(name: str = "state") -> MagicMock:
    """返回一个模拟 Parlant 状态对象的 MagicMock。

    返回的对象支持：
      - state.transition_to(...)  → 异步调用，返回 transition mock
      - transition.target         → 下一个状态 mock
    """
    state = MagicMock(name=name)
    transition = MagicMock(name=f"{name}_transition")
    transition.target = MagicMock(name=f"{name}_target")
    transition.target.transition_to = AsyncMock(return_value=transition)
    state.transition_to = AsyncMock(return_value=transition)
    return state


@pytest.fixture
def mock_journey() -> MagicMock:
    """返回一个预配置的 mock Journey 对象。"""
    j = MagicMock(name="journey")
    j.create_guideline = AsyncMock()
    j.initial_state = make_state_mock("initial_state")
    return j


@pytest.fixture
def mock_canned() -> MagicMock:
    """返回一个模拟 CannedResponse 对象的 MagicMock。"""
    return MagicMock(name="canned_response")


@pytest.fixture
def mock_agent(mock_journey: MagicMock, mock_canned: MagicMock) -> MagicMock:
    """返回一个预配置的 mock Agent 对象。

    mock_agent.create_journey()        → 返回 mock_journey
    mock_agent.create_canned_response() → 返回 mock_canned
    """
    agent = MagicMock(name="agent")
    agent.create_journey = AsyncMock(return_value=mock_journey)
    agent.create_canned_response = AsyncMock(return_value=mock_canned)
    return agent

