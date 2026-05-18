"""
Tests for parlant_journey.builder.JourneyBuilder.

All Parlant SDK objects are replaced with :class:`unittest.mock.AsyncMock` /
:class:`unittest.mock.MagicMock` so the tests run without a live Parlant server.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Stub out the parlant.sdk module so tests run without the real package
# ---------------------------------------------------------------------------

def _make_parlant_stub() -> types.ModuleType:
    parlant = types.ModuleType("parlant")
    sdk = types.ModuleType("parlant.sdk")

    class CompositionMode:
        STRICT = "STRICT"
        FLUID = "FLUID"

    sdk.CompositionMode = CompositionMode
    sdk.Agent = object
    sdk.Journey = object

    parlant.sdk = sdk
    sys.modules.setdefault("parlant", parlant)
    sys.modules.setdefault("parlant.sdk", sdk)
    return sdk


_sdk = _make_parlant_stub()

# Now we can import our own modules safely
from parlant_journey.builder import JourneyBuilder  # noqa: E402
from parlant_journey.models import (  # noqa: E402
    CannedResponseConfig,
    GuidelineConfig,
    JourneyConfig,
    StateConfig,
    TransitionConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state_mock(name: str = "state") -> MagicMock:
    """Return a mock that behaves like a Parlant state object."""
    state = MagicMock(name=name)
    transition = MagicMock(name=f"{name}_transition")
    transition.target = MagicMock(name=f"{name}_target")
    transition.target.transition_to = AsyncMock(return_value=transition)
    state.transition_to = AsyncMock(return_value=transition)
    return state


def _make_agent_and_journey() -> tuple:
    """Return (mock_agent, mock_journey, mock_initial_state)."""
    agent = MagicMock()
    journey = MagicMock()
    initial_state = _make_state_mock("initial")

    journey.initial_state = initial_state
    journey.create_guideline = AsyncMock()

    canned_response = MagicMock(name="canned_response")
    agent.create_journey = AsyncMock(return_value=journey)
    agent.create_canned_response = AsyncMock(return_value=canned_response)

    return agent, journey, initial_state, canned_response


def _linear_config() -> JourneyConfig:
    """A → B → C (linear, no conditions)."""
    return JourneyConfig(
        title="Linear",
        triggers=["go"],
        description="Linear test journey",
        states=[
            StateConfig(name="A", chat_state="Do A"),
            StateConfig(name="B", chat_state="Do B"),
            StateConfig(name="C", chat_state="Do C"),
        ],
        transitions=[
            TransitionConfig(source="A", target="B"),
            TransitionConfig(source="B", target="C"),
        ],
    )


# ---------------------------------------------------------------------------
# _create_journey
# ---------------------------------------------------------------------------

class TestCreateJourney:
    @pytest.mark.asyncio
    async def test_calls_agent_create_journey_with_correct_args(self):
        agent, journey, *_ = _make_agent_and_journey()
        config = _linear_config()
        builder = JourneyBuilder(agent, config)
        await builder._create_journey()

        agent.create_journey.assert_awaited_once_with(
            title="Linear",
            triggers=["go"],
            description="Linear test journey",
        )
        assert builder._journey is journey


# ---------------------------------------------------------------------------
# _create_canned_responses
# ---------------------------------------------------------------------------

class TestCreateCannedResponses:
    @pytest.mark.asyncio
    async def test_creates_canned_response_for_states_that_have_one(self):
        agent, journey, *_ = _make_agent_and_journey()
        config = JourneyConfig(
            title="T",
            triggers=["t"],
            description="d",
            states=[
                StateConfig(
                    name="verify",
                    canned_response=CannedResponseConfig(template="Please verify."),
                ),
                StateConfig(name="proceed", chat_state="Go ahead."),
            ],
            transitions=[TransitionConfig(source="verify", target="proceed")],
        )
        builder = JourneyBuilder(agent, config)
        builder._journey = journey
        await builder._create_canned_responses()

        agent.create_canned_response.assert_awaited_once_with(template="Please verify.")
        assert "verify" in builder._canrep_map
        assert "proceed" not in builder._canrep_map

    @pytest.mark.asyncio
    async def test_no_canned_responses_when_none_configured(self):
        agent, journey, *_ = _make_agent_and_journey()
        config = _linear_config()
        builder = JourneyBuilder(agent, config)
        builder._journey = journey
        await builder._create_canned_responses()

        agent.create_canned_response.assert_not_awaited()
        assert builder._canrep_map == {}


# ---------------------------------------------------------------------------
# _build_dag — linear topology
# ---------------------------------------------------------------------------

class TestBuildDagLinear:
    @pytest.mark.asyncio
    async def test_linear_creates_states_in_order(self):
        agent, journey, initial_state, _ = _make_agent_and_journey()

        # Set up chained states: initial → A → B → C
        trans_A = MagicMock()
        state_A = MagicMock()
        state_A.transition_to = AsyncMock()
        trans_A.target = state_A

        trans_B = MagicMock()
        state_B = MagicMock()
        state_B.transition_to = AsyncMock()
        trans_B.target = state_B

        trans_C = MagicMock()
        state_C = MagicMock()
        trans_C.target = state_C

        initial_state.transition_to = AsyncMock(return_value=trans_A)
        state_A.transition_to = AsyncMock(return_value=trans_B)
        state_B.transition_to = AsyncMock(return_value=trans_C)

        config = _linear_config()
        builder = JourneyBuilder(agent, config)
        builder._journey = journey
        await builder._build_dag()

        # Root A is created from initial_state
        initial_state.transition_to.assert_awaited_once()
        call_kwargs = initial_state.transition_to.call_args.kwargs
        assert call_kwargs.get("chat_state") == "Do A"

        # B is created from A
        state_A.transition_to.assert_awaited_once()
        b_kwargs = state_A.transition_to.call_args.kwargs
        assert b_kwargs.get("chat_state") == "Do B"

        # C is created from B
        state_B.transition_to.assert_awaited_once()
        c_kwargs = state_B.transition_to.call_args.kwargs
        assert c_kwargs.get("chat_state") == "Do C"

        # State map populated correctly
        assert builder._state_obj_map["A"] is state_A
        assert builder._state_obj_map["B"] is state_B
        assert builder._state_obj_map["C"] is state_C

    @pytest.mark.asyncio
    async def test_conditional_transition_passes_condition(self):
        agent, journey, initial_state, _ = _make_agent_and_journey()

        trans_A = MagicMock()
        state_A = MagicMock()
        trans_A.target = state_A
        trans_B = MagicMock()
        state_B = MagicMock()
        trans_B.target = state_B

        initial_state.transition_to = AsyncMock(return_value=trans_A)
        state_A.transition_to = AsyncMock(return_value=trans_B)

        config = JourneyConfig(
            title="T", triggers=["t"], description="d",
            states=[
                StateConfig(name="A", chat_state="Do A"),
                StateConfig(name="B", chat_state="Do B"),
            ],
            transitions=[TransitionConfig(source="A", target="B", condition="user confirmed")],
        )
        builder = JourneyBuilder(agent, config)
        builder._journey = journey
        await builder._build_dag()

        state_A.transition_to.assert_awaited_once()
        kwargs = state_A.transition_to.call_args.kwargs
        assert kwargs.get("condition") == "user confirmed"


# ---------------------------------------------------------------------------
# _build_dag — branching topology
# ---------------------------------------------------------------------------

class TestBuildDagBranching:
    @pytest.mark.asyncio
    async def test_branching_creates_separate_states(self):
        agent, journey, initial_state, _ = _make_agent_and_journey()

        trans_A = MagicMock()
        state_A = MagicMock()
        trans_A.target = state_A
        trans_B = MagicMock()
        state_B = MagicMock()
        trans_B.target = state_B
        trans_C = MagicMock()
        state_C = MagicMock()
        trans_C.target = state_C

        initial_state.transition_to = AsyncMock(return_value=trans_A)
        state_A.transition_to = AsyncMock(side_effect=[trans_B, trans_C])

        config = JourneyConfig(
            title="T", triggers=["t"], description="d",
            states=[
                StateConfig(name="A", chat_state="start"),
                StateConfig(name="B", chat_state="branch B"),
                StateConfig(name="C", chat_state="branch C"),
            ],
            transitions=[
                TransitionConfig(source="A", target="B", condition="cond B"),
                TransitionConfig(source="A", target="C", condition="cond C"),
            ],
        )
        builder = JourneyBuilder(agent, config)
        builder._journey = journey
        await builder._build_dag()

        assert state_A.transition_to.await_count == 2
        calls = state_A.transition_to.call_args_list
        conditions = {c.kwargs.get("condition") for c in calls}
        assert conditions == {"cond B", "cond C"}


# ---------------------------------------------------------------------------
# _build_dag — converging topology
# ---------------------------------------------------------------------------

class TestBuildDagConverging:
    @pytest.mark.asyncio
    async def test_converging_reuses_existing_state(self):
        agent, journey, initial_state, _ = _make_agent_and_journey()

        # A is root; B and C are branches; D is the convergence point
        trans_A = MagicMock()
        state_A = MagicMock()
        trans_A.target = state_A
        trans_B = MagicMock()
        state_B = MagicMock()
        trans_B.target = state_B
        trans_C = MagicMock()
        state_C = MagicMock()
        trans_C.target = state_C
        trans_D = MagicMock()
        state_D = MagicMock()
        trans_D.target = state_D

        initial_state.transition_to = AsyncMock(return_value=trans_A)
        state_A.transition_to = AsyncMock(side_effect=[trans_B, trans_C])
        state_B.transition_to = AsyncMock(return_value=trans_D)
        state_C.transition_to = AsyncMock()  # convergence: no new transition created

        config = JourneyConfig(
            title="T", triggers=["t"], description="d",
            states=[
                StateConfig(name="A", chat_state="start"),
                StateConfig(name="B", chat_state="branch B"),
                StateConfig(name="C", chat_state="branch C"),
                StateConfig(name="D", chat_state="end"),
            ],
            transitions=[
                TransitionConfig(source="A", target="B"),
                TransitionConfig(source="A", target="C"),
                TransitionConfig(source="B", target="D"),
                TransitionConfig(source="C", target="D"),  # convergence
            ],
        )
        builder = JourneyBuilder(agent, config)
        builder._journey = journey
        await builder._build_dag()

        # D was created once (from B)
        assert builder._state_obj_map["D"] is state_D

        # The convergence transition C→D called transition_to with state=state_D
        state_C.transition_to.assert_awaited_once()
        kwargs = state_C.transition_to.call_args.kwargs
        assert kwargs.get("state") is state_D


# ---------------------------------------------------------------------------
# _build_dag — canned responses applied
# ---------------------------------------------------------------------------

class TestBuildDagCannedResponses:
    @pytest.mark.asyncio
    async def test_canned_response_attached_to_state(self):
        agent, journey, initial_state, canned_mock = _make_agent_and_journey()

        trans_A = MagicMock()
        state_A = MagicMock()
        trans_A.target = state_A
        initial_state.transition_to = AsyncMock(return_value=trans_A)

        config = JourneyConfig(
            title="T", triggers=["t"], description="d",
            states=[
                StateConfig(
                    name="A",
                    chat_state="verify",
                    canned_response=CannedResponseConfig(
                        template="Please verify.", strict=True
                    ),
                ),
            ],
            transitions=[],
        )
        builder = JourneyBuilder(agent, config)
        builder._journey = journey
        # Pre-populate canrep_map as _create_canned_responses would
        builder._canrep_map["A"] = canned_mock
        await builder._build_dag()

        kwargs = initial_state.transition_to.call_args.kwargs
        assert kwargs.get("canned_responses") == [canned_mock]
        assert kwargs.get("composition_mode") == "STRICT"

    @pytest.mark.asyncio
    async def test_fluid_composition_mode_when_strict_false(self):
        agent, journey, initial_state, canned_mock = _make_agent_and_journey()

        trans_A = MagicMock()
        state_A = MagicMock()
        trans_A.target = state_A
        initial_state.transition_to = AsyncMock(return_value=trans_A)

        config = JourneyConfig(
            title="T", triggers=["t"], description="d",
            states=[
                StateConfig(
                    name="A",
                    chat_state="say hi",
                    canned_response=CannedResponseConfig(template="Hi.", strict=False),
                ),
            ],
            transitions=[],
        )
        builder = JourneyBuilder(agent, config)
        builder._journey = journey
        builder._canrep_map["A"] = canned_mock
        await builder._build_dag()

        kwargs = initial_state.transition_to.call_args.kwargs
        assert kwargs.get("composition_mode") == "FLUID"


# ---------------------------------------------------------------------------
# _register_node_guidelines
# ---------------------------------------------------------------------------

class TestRegisterNodeGuidelines:
    @pytest.mark.asyncio
    async def test_registers_each_node_guideline(self):
        agent, journey, *_ = _make_agent_and_journey()
        config = JourneyConfig(
            title="T", triggers=["t"], description="d",
            states=[
                StateConfig(
                    name="A",
                    guidelines=[
                        GuidelineConfig(condition="x", action="do x", priority=80),
                        GuidelineConfig(condition="y", action="do y", priority=60),
                    ],
                ),
                StateConfig(name="B"),
            ],
            transitions=[],
        )
        builder = JourneyBuilder(agent, config)
        builder._journey = journey
        await builder._register_node_guidelines()

        assert journey.create_guideline.await_count == 2
        calls = journey.create_guideline.call_args_list
        conditions = {c.kwargs["condition"] for c in calls}
        assert conditions == {"x", "y"}

    @pytest.mark.asyncio
    async def test_no_guidelines_when_none_configured(self):
        agent, journey, *_ = _make_agent_and_journey()
        builder = JourneyBuilder(agent, _linear_config())
        builder._journey = journey
        await builder._register_node_guidelines()
        journey.create_guideline.assert_not_awaited()


# ---------------------------------------------------------------------------
# _register_global_guidelines
# ---------------------------------------------------------------------------

class TestRegisterGlobalGuidelines:
    @pytest.mark.asyncio
    async def test_registers_global_guidelines(self):
        agent, journey, *_ = _make_agent_and_journey()
        config = _linear_config()
        config.global_guidelines = [
            GuidelineConfig(condition="angry", action="calm", priority=90),
        ]
        builder = JourneyBuilder(agent, config)
        builder._journey = journey
        await builder._register_global_guidelines()

        journey.create_guideline.assert_awaited_once_with(
            condition="angry", action="calm", priority=90
        )


# ---------------------------------------------------------------------------
# build() — full integration (all steps called)
# ---------------------------------------------------------------------------

class TestBuildIntegration:
    @pytest.mark.asyncio
    async def test_build_calls_all_steps_and_returns_journey(self):
        agent, journey, initial_state, _ = _make_agent_and_journey()

        trans_A = MagicMock()
        state_A = MagicMock()
        trans_A.target = state_A
        initial_state.transition_to = AsyncMock(return_value=trans_A)

        config = JourneyConfig(
            title="Smoke", triggers=["s"], description="d",
            states=[StateConfig(name="A", chat_state="Hello")],
            transitions=[],
            global_guidelines=[GuidelineConfig(condition="c", action="a", priority=50)],
        )
        result = await JourneyBuilder(agent, config).build()

        assert result is journey
        agent.create_journey.assert_awaited_once()
        initial_state.transition_to.assert_awaited_once()
        journey.create_guideline.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_missing_state_config_raises_value_error(self):
        agent, journey, initial_state, _ = _make_agent_and_journey()
        initial_state.transition_to = AsyncMock(
            return_value=MagicMock(target=MagicMock())
        )

        config = JourneyConfig(
            title="T", triggers=["t"], description="d",
            states=[StateConfig(name="A", chat_state="a")],
            # B is referenced in transitions but not in states
            transitions=[TransitionConfig(source="A", target="B")],
        )
        builder = JourneyBuilder(agent, config)
        builder._journey = journey
        builder._state_obj_map["A"] = MagicMock(
            transition_to=AsyncMock(return_value=MagicMock(target=MagicMock()))
        )
        with pytest.raises(ValueError, match="B"):
            await builder._build_dag()
