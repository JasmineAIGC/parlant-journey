"""
02_test_builder.py — journey.builder.JourneyBuilder 单元测试

使用 conftest.py 中的公共 fixtures（mock_agent / mock_journey / mock_canned），
以及本文件的辅助函数 _trans()，简化 Parlant SDK 的 mock 搭建。

parlant.sdk 桩由 conftest.py 在收集阶段统一注入，无需在此重复定义。
"""

from __future__ import annotations

import os
import sys
import types

# ---------------------------------------------------------------------------
# 路径设置：支持 `python tests/02_test_builder.py` 在任意目录下直接执行
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# parlant.sdk 桩（pytest 通过 conftest.py 注入，直接执行时在此处注入）
if "parlant.sdk" not in sys.modules:
    _parlant = types.ModuleType("parlant")
    _sdk = types.ModuleType("parlant.sdk")
    _sdk.CompositionMode = type("CompositionMode", (), {"STRICT": "STRICT", "FLUID": "FLUID"})
    _sdk.Agent = object
    _sdk.Journey = object
    _sdk.Server = object
    _parlant.sdk = _sdk
    sys.modules.setdefault("parlant", _parlant)
    sys.modules.setdefault("parlant.sdk", _sdk)

from unittest.mock import AsyncMock, MagicMock

import pytest

from journey.builder import JourneyBuilder
from journey.models import (
    CannedResponseConfig,
    GuidelineConfig,
    JourneyConfig,
    StateConfig,
    TransitionConfig,
)


# ---------------------------------------------------------------------------
# 测试辅助
# ---------------------------------------------------------------------------

def _trans(chat: str = "默认指令") -> tuple:
    """创建一对 (transition_mock, state_mock)，模拟 transition_to() 的返回值链。

    Args:
        chat: 仅用于 mock 的名称标注，方便调试时识别。

    Returns:
        (transition_mock, state_mock)：
          - transition_mock.target == state_mock
          - state_mock.transition_to  是可被 await 的 AsyncMock
    """
    state = MagicMock(name=f"state[{chat}]")
    trans = MagicMock(name=f"trans[{chat}]")
    trans.target = state
    state.transition_to = AsyncMock()
    return trans, state


def _简单配置(title: str = "测试流程") -> JourneyConfig:
    """返回最小可用的 JourneyConfig（单状态，无转移）。"""
    return JourneyConfig(
        title=title,
        conditions=["go"],
        description="单元测试用配置",
        states=[StateConfig(name="A", chat_state="执行 A")],
        transitions=[],
    )


def _线性配置() -> JourneyConfig:
    """返回 A → B → C 的线性配置。"""
    return JourneyConfig(
        title="线性流程",
        conditions=["go"],
        description="线性测试",
        states=[
            StateConfig(name="A", chat_state="执行 A"),
            StateConfig(name="B", chat_state="执行 B"),
            StateConfig(name="C", chat_state="执行 C"),
        ],
        transitions=[
            TransitionConfig(source="A", target="B"),
            TransitionConfig(source="B", target="C"),
        ],
    )


# ---------------------------------------------------------------------------
# _create_journey
# ---------------------------------------------------------------------------

class Test创建Journey外壳:
    @pytest.mark.asyncio
    async def test_使用正确参数调用SDK(self, mock_agent, mock_journey):
        config = _线性配置()
        builder = JourneyBuilder(mock_agent, config)
        await builder._create_journey()

        mock_agent.create_journey.assert_awaited_once_with(
            title="线性流程",
            description="线性测试",
            conditions=["go"],
        )
        assert builder._journey is mock_journey


# ---------------------------------------------------------------------------
# _create_canned_responses
# ---------------------------------------------------------------------------

class Test预创建预设话术:
    @pytest.mark.asyncio
    async def test_有话术配置时创建CannedResponse(self, mock_agent, mock_journey):
        config = JourneyConfig(
            title="T", conditions=["t"], description="d",
            states=[
                StateConfig(name="verify",
                            canned_response=CannedResponseConfig(template="请验证身份。")),
                StateConfig(name="proceed", chat_state="继续。"),
            ],
            transitions=[TransitionConfig(source="verify", target="proceed")],
        )
        builder = JourneyBuilder(mock_agent, config)
        builder._journey = mock_journey
        await builder._create_canned_responses()

        mock_agent.create_canned_response.assert_awaited_once_with(template="请验证身份。")
        assert "verify" in builder._canrep_map
        assert "proceed" not in builder._canrep_map

    @pytest.mark.asyncio
    async def test_无话术配置时不调用SDK(self, mock_agent, mock_journey):
        builder = JourneyBuilder(mock_agent, _线性配置())
        builder._journey = mock_journey
        await builder._create_canned_responses()

        mock_agent.create_canned_response.assert_not_awaited()
        assert builder._canrep_map == {}


# ---------------------------------------------------------------------------
# _build_dag — 线性拓扑
# ---------------------------------------------------------------------------

class TestDAG线性拓扑:
    @pytest.mark.asyncio
    async def test_线性流程按顺序创建状态(self, mock_agent, mock_journey):
        # 搭建 initial → A → B → C 的 mock 链
        trans_A, state_A = _trans("A")
        trans_B, state_B = _trans("B")
        trans_C, state_C = _trans("C")

        mock_journey.initial_state.transition_to = AsyncMock(return_value=trans_A)
        state_A.transition_to = AsyncMock(return_value=trans_B)
        state_B.transition_to = AsyncMock(return_value=trans_C)

        builder = JourneyBuilder(mock_agent, _线性配置())
        builder._journey = mock_journey
        await builder._build_dag()

        # 根状态 A 从 initial_state 创建
        mock_journey.initial_state.transition_to.assert_awaited_once()
        assert mock_journey.initial_state.transition_to.call_args.kwargs["chat_state"] == "执行 A"

        # B 从 A 创建，C 从 B 创建
        assert state_A.transition_to.call_args.kwargs["chat_state"] == "执行 B"
        assert state_B.transition_to.call_args.kwargs["chat_state"] == "执行 C"

        # 状态映射正确
        assert builder._state_obj_map == {"A": state_A, "B": state_B, "C": state_C}

    @pytest.mark.asyncio
    async def test_有条件转移传递condition参数(self, mock_agent, mock_journey):
        trans_A, state_A = _trans("A")
        trans_B, _ = _trans("B")

        mock_journey.initial_state.transition_to = AsyncMock(return_value=trans_A)
        state_A.transition_to = AsyncMock(return_value=trans_B)

        config = JourneyConfig(
            title="T", conditions=["t"], description="d",
            states=[StateConfig(name="A", chat_state="a"), StateConfig(name="B", chat_state="b")],
            transitions=[TransitionConfig(source="A", target="B", condition="用户已确认")],
        )
        builder = JourneyBuilder(mock_agent, config)
        builder._journey = mock_journey
        await builder._build_dag()

        assert state_A.transition_to.call_args.kwargs["condition"] == "用户已确认"


# ---------------------------------------------------------------------------
# _build_dag — 分支拓扑
# ---------------------------------------------------------------------------

class TestDAG分支拓扑:
    @pytest.mark.asyncio
    async def test_分支创建两个独立目标状态(self, mock_agent, mock_journey):
        trans_A, state_A = _trans("A")
        trans_B, _ = _trans("B")
        trans_C, _ = _trans("C")

        mock_journey.initial_state.transition_to = AsyncMock(return_value=trans_A)
        state_A.transition_to = AsyncMock(side_effect=[trans_B, trans_C])

        config = JourneyConfig(
            title="T", conditions=["t"], description="d",
            states=[
                StateConfig(name="A", chat_state="开始"),
                StateConfig(name="B", chat_state="分支B"),
                StateConfig(name="C", chat_state="分支C"),
            ],
            transitions=[
                TransitionConfig(source="A", target="B", condition="条件B"),
                TransitionConfig(source="A", target="C", condition="条件C"),
            ],
        )
        builder = JourneyBuilder(mock_agent, config)
        builder._journey = mock_journey
        await builder._build_dag()

        assert state_A.transition_to.await_count == 2
        conditions = {c.kwargs["condition"] for c in state_A.transition_to.call_args_list}
        assert conditions == {"条件B", "条件C"}


# ---------------------------------------------------------------------------
# _build_dag — 汇聚拓扑
# ---------------------------------------------------------------------------

class TestDAG汇聚拓扑:
    @pytest.mark.asyncio
    async def test_汇聚时复用已有状态对象(self, mock_agent, mock_journey):
        # A → B, A → C, B → D, C → D（D 是汇聚点）
        trans_A, state_A = _trans("A")
        trans_B, state_B = _trans("B")
        trans_C, state_C = _trans("C")
        trans_D, state_D = _trans("D")

        mock_journey.initial_state.transition_to = AsyncMock(return_value=trans_A)
        state_A.transition_to = AsyncMock(side_effect=[trans_B, trans_C])
        state_B.transition_to = AsyncMock(return_value=trans_D)
        state_C.transition_to = AsyncMock()  # 汇聚：复用 state_D，不返回新 target

        config = JourneyConfig(
            title="T", conditions=["t"], description="d",
            states=[
                StateConfig(name="A", chat_state="开始"),
                StateConfig(name="B", chat_state="分支B"),
                StateConfig(name="C", chat_state="分支C"),
                StateConfig(name="D", chat_state="终点"),
            ],
            transitions=[
                TransitionConfig(source="A", target="B"),
                TransitionConfig(source="A", target="C"),
                TransitionConfig(source="B", target="D"),
                TransitionConfig(source="C", target="D"),  # 汇聚
            ],
        )
        builder = JourneyBuilder(mock_agent, config)
        builder._journey = mock_journey
        await builder._build_dag()

        # D 只被创建一次（从 B）
        assert builder._state_obj_map["D"] is state_D

        # C → D 的转移使用 state=state_D（复用模式）
        state_C.transition_to.assert_awaited_once()
        assert state_C.transition_to.call_args.kwargs["state"] is state_D


# ---------------------------------------------------------------------------
# _build_dag — 预设话术附加
# ---------------------------------------------------------------------------

class TestDAG话术附加:
    @pytest.mark.asyncio
    async def test_严格模式话术正确附加(self, mock_agent, mock_journey, mock_canned):
        trans_A, _ = _trans("A")
        mock_journey.initial_state.transition_to = AsyncMock(return_value=trans_A)

        config = JourneyConfig(
            title="T", conditions=["t"], description="d",
            states=[StateConfig(
                name="A", chat_state="验证",
                canned_response=CannedResponseConfig(template="请验证身份。", strict=True),
            )],
            transitions=[],
        )
        builder = JourneyBuilder(mock_agent, config)
        builder._journey = mock_journey
        builder._canrep_map["A"] = mock_canned  # 模拟步骤②已预创建
        await builder._build_dag()

        kwargs = mock_journey.initial_state.transition_to.call_args.kwargs
        assert kwargs["canned_responses"] == [mock_canned]
        assert kwargs["composition_mode"] == "STRICT"

    @pytest.mark.asyncio
    async def test_流式模式话术使用FLUID(self, mock_agent, mock_journey, mock_canned):
        trans_A, _ = _trans("A")
        mock_journey.initial_state.transition_to = AsyncMock(return_value=trans_A)

        config = JourneyConfig(
            title="T", conditions=["t"], description="d",
            states=[StateConfig(
                name="A", chat_state="问候",
                canned_response=CannedResponseConfig(template="你好。", strict=False),
            )],
            transitions=[],
        )
        builder = JourneyBuilder(mock_agent, config)
        builder._journey = mock_journey
        builder._canrep_map["A"] = mock_canned
        await builder._build_dag()

        kwargs = mock_journey.initial_state.transition_to.call_args.kwargs
        assert kwargs["composition_mode"] == "FLUID"


# ---------------------------------------------------------------------------
# _register_node_guidelines
# ---------------------------------------------------------------------------

class Test注册节点规则:
    @pytest.mark.asyncio
    async def test_注册所有节点规则(self, mock_agent, mock_journey):
        config = JourneyConfig(
            title="T", conditions=["t"], description="d",
            states=[
                StateConfig(name="A", guidelines=[
                    GuidelineConfig(condition="情况X", action="处理X", priority=80),
                    GuidelineConfig(condition="情况Y", action="处理Y", priority=60),
                ]),
                StateConfig(name="B"),
            ],
            transitions=[],
        )
        builder = JourneyBuilder(mock_agent, config)
        builder._journey = mock_journey
        await builder._register_node_guidelines()

        assert mock_journey.create_guideline.await_count == 2
        conditions = {c.kwargs["condition"]
                      for c in mock_journey.create_guideline.call_args_list}
        assert conditions == {"情况X", "情况Y"}

    @pytest.mark.asyncio
    async def test_无规则配置时不调用SDK(self, mock_agent, mock_journey):
        builder = JourneyBuilder(mock_agent, _线性配置())
        builder._journey = mock_journey
        await builder._register_node_guidelines()
        mock_journey.create_guideline.assert_not_awaited()


# ---------------------------------------------------------------------------
# _register_global_guidelines
# ---------------------------------------------------------------------------

class Test注册全局规则:
    @pytest.mark.asyncio
    async def test_注册全局规则(self, mock_agent, mock_journey):
        config = _线性配置()
        config.global_guidelines = [
            GuidelineConfig(condition="情绪激动", action="安抚", priority=90),
        ]
        builder = JourneyBuilder(mock_agent, config)
        builder._journey = mock_journey
        await builder._register_global_guidelines()

        mock_journey.create_guideline.assert_awaited_once_with(
            condition="情绪激动", action="安抚", priority=90
        )


# ---------------------------------------------------------------------------
# build() 整体集成
# ---------------------------------------------------------------------------

class TestBuild整体流程:
    @pytest.mark.asyncio
    async def test_所有步骤被调用并返回Journey(self, mock_agent, mock_journey):
        trans_A, _ = _trans("A")
        mock_journey.initial_state.transition_to = AsyncMock(return_value=trans_A)

        config = JourneyConfig(
            title="冒烟测试", conditions=["s"], description="d",
            states=[StateConfig(name="A", chat_state="你好")],
            transitions=[],
            global_guidelines=[GuidelineConfig(condition="c", action="a", priority=50)],
        )
        result = await JourneyBuilder(mock_agent, config).build()

        assert result is mock_journey
        mock_agent.create_journey.assert_awaited_once()
        mock_journey.initial_state.transition_to.assert_awaited_once()
        mock_journey.create_guideline.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_缺少StateConfig时抛出ValueError(self, mock_agent, mock_journey):
        # B 在 transitions 中引用，但未在 states 中定义
        config = JourneyConfig(
            title="T", conditions=["t"], description="d",
            states=[StateConfig(name="A", chat_state="a")],
            transitions=[TransitionConfig(source="A", target="B")],
        )
        builder = JourneyBuilder(mock_agent, config)
        builder._journey = mock_journey
        # A 已在 state_obj_map 中（模拟已被创建）
        builder._state_obj_map["A"] = MagicMock(
            transition_to=AsyncMock(return_value=MagicMock(target=MagicMock()))
        )
        with pytest.raises(ValueError, match="B"):
            await builder._build_dag()


# ---------------------------------------------------------------------------
# 直接运行支持：python tests/02_test_builder.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

