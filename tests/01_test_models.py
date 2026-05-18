"""
01_test_models.py — journey.models 数据层单元测试

纯数据测试，无需任何 mock 或外部依赖。
"""

import pytest
from dataclasses import fields

from journey.models import (
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
    def test_默认值(self):
        cfg = CannedResponseConfig(template="你好，我是客服。")
        assert cfg.template == "你好，我是客服。"
        assert cfg.strict is True  # 默认严格模式

    def test_非严格模式(self):
        cfg = CannedResponseConfig(template="你好。", strict=False)
        assert cfg.strict is False

    def test_字段集合(self):
        # 确保只有 template 和 strict 两个字段（防止意外字段混入）
        assert {f.name for f in fields(CannedResponseConfig)} == {"template", "strict"}


# ---------------------------------------------------------------------------
# GuidelineConfig
# ---------------------------------------------------------------------------

class TestGuidelineConfig:
    def test_默认优先级(self):
        g = GuidelineConfig(condition="用户情绪激动", action="先安抚情绪")
        assert g.condition == "用户情绪激动"
        assert g.action == "先安抚情绪"
        assert g.priority == 50

    def test_自定义优先级(self):
        g = GuidelineConfig(condition="x", action="y", priority=90)
        assert g.priority == 90


# ---------------------------------------------------------------------------
# StateConfig
# ---------------------------------------------------------------------------

class TestStateConfig:
    def test_最小构造(self):
        s = StateConfig(name="verify")
        assert s.name == "verify"
        assert s.description == ""
        assert s.chat_state is None
        assert s.tool_state is None
        assert s.tool_instruction is None
        assert s.guidelines == []
        assert s.canned_response is None

    def test_带对话指令(self):
        s = StateConfig(name="verify", chat_state="请验证您的身份")
        assert s.chat_state == "请验证您的身份"

    def test_带预设话术(self):
        cr = CannedResponseConfig(template="固定回复。", strict=True)
        s = StateConfig(name="verify", canned_response=cr)
        assert s.canned_response is cr

    def test_规则列表各实例独立(self):
        """验证不同实例的 guidelines 列表互不共享（dataclass 可变默认值陷阱）。"""
        s1 = StateConfig(name="a")
        s2 = StateConfig(name="b")
        s1.guidelines.append(GuidelineConfig(condition="c", action="a"))
        assert s2.guidelines == []  # s2 不受 s1 影响


# ---------------------------------------------------------------------------
# TransitionConfig
# ---------------------------------------------------------------------------

class TestTransitionConfig:
    def test_无条件转移(self):
        t = TransitionConfig(source="A", target="B")
        assert t.source == "A"
        assert t.target == "B"
        assert t.condition is None

    def test_有条件转移(self):
        t = TransitionConfig(source="A", target="B", condition="用户已验证")
        assert t.condition == "用户已验证"


# ---------------------------------------------------------------------------
# JourneyConfig
# ---------------------------------------------------------------------------

class TestJourneyConfig:
    def _最小配置(self) -> JourneyConfig:
        return JourneyConfig(
            title="测试流程",
            triggers=["开始"],
            description="测试用。",
            states=[StateConfig(name="A", chat_state="你好")],
            transitions=[],
        )

    def test_最小构造(self):
        cfg = self._最小配置()
        assert cfg.title == "测试流程"
        assert cfg.triggers == ["开始"]
        assert cfg.global_guidelines == []

    def test_添加全局规则(self):
        g = GuidelineConfig(condition="情绪激动", action="安抚", priority=80)
        cfg = self._最小配置()
        cfg.global_guidelines.append(g)
        assert len(cfg.global_guidelines) == 1

    def test_全局规则各实例独立(self):
        c1 = self._最小配置()
        c2 = self._最小配置()
        c1.global_guidelines.append(GuidelineConfig(condition="x", action="y"))
        assert c2.global_guidelines == []

    def test_多状态多转移(self):
        cfg = JourneyConfig(
            title="T",
            triggers=["t"],
            description="d",
            states=[StateConfig(name=n) for n in ("A", "B", "C")],
            transitions=[
                TransitionConfig(source="A", target="B"),
                TransitionConfig(source="B", target="C"),
            ],
        )
        assert len(cfg.states) == 3
        assert len(cfg.transitions) == 2

