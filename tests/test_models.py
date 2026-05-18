"""Unit tests for parlant_journey.models — pure data layer, no mocking needed."""

import pytest
from dataclasses import fields

from parlant_journey.models import (
    CannedResponseConfig,
    GuidelineConfig,
    JourneyConfig,
    StateConfig,
    TransitionConfig,
)


# ---------------------------------------------------------------------------
# CannedResponseConfig
# ---------------------------------------------------------------------------

class TestCannedResponseConfig:
    def test_defaults(self):
        cfg = CannedResponseConfig(template="Hello, I am an agent.")
        assert cfg.template == "Hello, I am an agent."
        assert cfg.strict is True

    def test_non_strict(self):
        cfg = CannedResponseConfig(template="Hello.", strict=False)
        assert cfg.strict is False

    def test_is_dataclass(self):
        field_names = {f.name for f in fields(CannedResponseConfig)}
        assert field_names == {"template", "strict"}


# ---------------------------------------------------------------------------
# GuidelineConfig
# ---------------------------------------------------------------------------

class TestGuidelineConfig:
    def test_defaults(self):
        g = GuidelineConfig(condition="user is angry", action="de-escalate")
        assert g.condition == "user is angry"
        assert g.action == "de-escalate"
        assert g.priority == 50

    def test_custom_priority(self):
        g = GuidelineConfig(condition="x", action="y", priority=90)
        assert g.priority == 90


# ---------------------------------------------------------------------------
# StateConfig
# ---------------------------------------------------------------------------

class TestStateConfig:
    def test_minimal(self):
        s = StateConfig(name="verify")
        assert s.name == "verify"
        assert s.description == ""
        assert s.chat_state is None
        assert s.tool_state is None
        assert s.tool_instruction is None
        assert s.guidelines == []
        assert s.canned_response is None

    def test_with_chat_state(self):
        s = StateConfig(name="verify", chat_state="Ask user for ID.")
        assert s.chat_state == "Ask user for ID."

    def test_with_canned_response(self):
        cr = CannedResponseConfig(template="Fixed reply.", strict=True)
        s = StateConfig(name="verify", canned_response=cr)
        assert s.canned_response is cr

    def test_guidelines_are_independent_per_instance(self):
        """Each StateConfig must have its own list (not shared via mutable default)."""
        s1 = StateConfig(name="a")
        s2 = StateConfig(name="b")
        s1.guidelines.append(GuidelineConfig(condition="c", action="a"))
        assert s2.guidelines == []


# ---------------------------------------------------------------------------
# TransitionConfig
# ---------------------------------------------------------------------------

class TestTransitionConfig:
    def test_unconditional(self):
        t = TransitionConfig(source="A", target="B")
        assert t.source == "A"
        assert t.target == "B"
        assert t.condition is None

    def test_conditional(self):
        t = TransitionConfig(source="A", target="B", condition="user verified")
        assert t.condition == "user verified"


# ---------------------------------------------------------------------------
# JourneyConfig
# ---------------------------------------------------------------------------

class TestJourneyConfig:
    def _minimal_config(self) -> JourneyConfig:
        return JourneyConfig(
            title="Test Journey",
            triggers=["start"],
            description="A test.",
            states=[StateConfig(name="A", chat_state="Hello")],
            transitions=[],
        )

    def test_minimal(self):
        cfg = self._minimal_config()
        assert cfg.title == "Test Journey"
        assert cfg.triggers == ["start"]
        assert cfg.global_guidelines == []

    def test_with_global_guidelines(self):
        g = GuidelineConfig(condition="angry", action="calm down", priority=80)
        cfg = self._minimal_config()
        cfg.global_guidelines.append(g)
        assert len(cfg.global_guidelines) == 1

    def test_global_guidelines_independent_per_instance(self):
        c1 = self._minimal_config()
        c2 = self._minimal_config()
        c1.global_guidelines.append(GuidelineConfig(condition="x", action="y"))
        assert c2.global_guidelines == []

    def test_multiple_states(self):
        cfg = JourneyConfig(
            title="T",
            triggers=["t"],
            description="d",
            states=[
                StateConfig(name="A"),
                StateConfig(name="B"),
                StateConfig(name="C"),
            ],
            transitions=[
                TransitionConfig(source="A", target="B"),
                TransitionConfig(source="B", target="C"),
            ],
        )
        assert len(cfg.states) == 3
        assert len(cfg.transitions) == 2
