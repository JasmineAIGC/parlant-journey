from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from demos.decision_engine.demo_decision_engine import (
    DECISION_JOURNEY_CONFIGS,
    DECISION_JOURNEY_FILES,
    DEMO_PORT,
    DEMO_SCENARIOS,
    _load_demo_port,
)


def test_decision_engine_demo_loads_all_strategy_journeys() -> None:
    assert DECISION_JOURNEY_FILES == [
        '01_refusal.yaml',
        '02_user_terminate.yaml',
        '03_clarify.yaml',
        '04_direct_answer.yaml',
        '05_repeat_explain.yaml',
        '06_greeting_transition.yaml',
        '07_resolved.yaml',
    ]
    assert [cfg.title for cfg in DECISION_JOURNEY_CONFIGS] == [
        '策略1：拒绝回答',
        '策略2：用户主动终止咨询',
        '策略3：确认/追问',
        '策略4：直接作答',
        '策略5：重复说明',
        '策略6：问候/话题过渡',
        '策略7：问题已解决',
    ]


def test_decision_engine_demo_uses_multi_state_journeys() -> None:
    assert DEMO_PORT > 0
    for cfg in DECISION_JOURNEY_CONFIGS:
        assert len(cfg.states) >= 3
        assert len(cfg.transitions) >= 2
        assert cfg.states[0].canned_response is not None


def test_decision_engine_demo_scenarios_match_loaded_journeys() -> None:
    journey_titles = {cfg.title for cfg in DECISION_JOURNEY_CONFIGS}

    assert len(DEMO_SCENARIOS) == 7
    for scenario in DEMO_SCENARIOS:
        assert scenario['journey_title'] in journey_titles
        assert len(scenario['messages']) >= 2


def test_decision_engine_demo_invalid_port_env_raises(monkeypatch) -> None:
    monkeypatch.setenv("PARLANT_SERVER_PORT", "bad-port")

    import pytest

    with pytest.raises(ValueError, match="PARLANT_SERVER_PORT 必须为整数"):
        _load_demo_port()
