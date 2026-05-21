from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from journey.builder import JourneyBuilder
from journey.models import JourneyConfig, StateConfig, TransitionConfig


def _transition_pair(label: str) -> tuple[MagicMock, MagicMock]:
    state = MagicMock(name=f"state_{label}")
    transition = MagicMock(name=f"transition_{label}")
    transition.target = state
    state.transition_to = AsyncMock()
    return transition, state


@pytest.mark.asyncio
async def test_builder_handles_unsorted_transitions(mock_agent, mock_journey):
    trans_a, state_a = _transition_pair("a")
    trans_b, state_b = _transition_pair("b")
    trans_c, state_c = _transition_pair("c")

    mock_journey.initial_state.transition_to = AsyncMock(return_value=trans_a)
    state_a.transition_to = AsyncMock(return_value=trans_b)
    state_b.transition_to = AsyncMock(return_value=trans_c)

    config = JourneyConfig(
        title="generic-flow",
        conditions=["start"],
        description="supports arbitrary transition order",
        states=[
            StateConfig(name="a", chat_state="step a"),
            StateConfig(name="b", chat_state="step b"),
            StateConfig(name="c", chat_state="step c"),
        ],
        transitions=[
            TransitionConfig(source="b", target="c"),
            TransitionConfig(source="a", target="b"),
        ],
    )
    builder = JourneyBuilder(mock_agent, config)
    builder._journey = mock_journey

    await builder._build_dag()

    assert builder._state_obj_map == {"a": state_a, "b": state_b, "c": state_c}


def test_builder_rejects_unknown_transition_state(mock_agent):
    config = JourneyConfig(
        title="invalid-flow",
        conditions=["start"],
        description="unknown state in transitions",
        states=[StateConfig(name="a", chat_state="step a")],
        transitions=[TransitionConfig(source="a", target="missing")],
    )

    with pytest.raises(ValueError, match="未定义状态"):
        JourneyBuilder(mock_agent, config)


def test_builder_rejects_state_with_chat_and_tool(mock_agent):
    config = JourneyConfig(
        title="invalid-state",
        conditions=["start"],
        description="chat and tool both set",
        states=[StateConfig(name="a", chat_state="text", tool_state={"name": "tool"})],
        transitions=[],
    )

    with pytest.raises(ValueError, match="chat_state 与 tool_state"):
        JourneyBuilder(mock_agent, config)


@pytest.mark.asyncio
async def test_builder_raises_clear_error_for_unreachable_cycle(mock_agent, mock_journey):
    config = JourneyConfig(
        title="cycle-flow",
        conditions=["start"],
        description="cycle without root",
        states=[
            StateConfig(name="a", chat_state="step a"),
            StateConfig(name="b", chat_state="step b"),
        ],
        transitions=[
            TransitionConfig(source="a", target="b"),
            TransitionConfig(source="b", target="a"),
        ],
    )
    builder = JourneyBuilder(mock_agent, config)
    builder._journey = mock_journey

    with pytest.raises(ValueError, match="循环依赖"):
        await builder._build_dag()
