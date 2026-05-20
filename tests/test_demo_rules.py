from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from demos.rules.demo_rules import DEMO_SCENARIOS, RULE_CONFIGS, RULE_FILES


def test_rules_demo_loads_all_strategy_journeys() -> None:
    assert RULE_FILES == [
        "01_refusal.yaml",
        "02_user_terminate.yaml",
        "03_clarify.yaml",
        "04_direct_answer.yaml",
        "05_repeat_explain.yaml",
        "06_greeting_transition.yaml",
        "07_resolved.yaml",
    ]
    assert [cfg.title for cfg in RULE_CONFIGS] == [
        "策略1：拒绝回答",
        "策略2：用户主动终止咨询",
        "策略3：确认/追问",
        "策略4：直接作答",
        "策略5：重复说明",
        "策略6：问候/话题过渡",
        "策略7：问题已解决",
    ]


def test_rules_demo_scenarios_match_loaded_journeys() -> None:
    journey_titles = {cfg.title for cfg in RULE_CONFIGS}

    assert len(DEMO_SCENARIOS) == 7
    for scenario in DEMO_SCENARIOS:
        assert scenario["journey_title"] in journey_titles
        assert scenario["messages"]
