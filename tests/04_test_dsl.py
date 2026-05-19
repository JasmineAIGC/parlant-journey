"""
tests/04_test_dsl.py — journey.dsl.JourneyDSL 和 journey.loader 单元测试

验证两种"快速场景迁移"工具：
  1. JourneyDSL — 流式链式 API，无需直接操作 dataclass
  2. load_journey / load_journey_dict — 从 YAML/JSON/字典加载配置
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径设置：支持 `python tests/04_test_dsl.py` 在任意目录下直接执行
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

import pytest

from journey.dsl import JourneyDSL
from journey.loader import load_journey, load_journey_dict
from journey.models import JourneyConfig, StateConfig, TransitionConfig

_FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# JourneyDSL — 基本属性设置
# ---------------------------------------------------------------------------

class TestDSL基本属性:
    def test_标题被正确设置(self):
        config = JourneyDSL("欺诈处理").state("A", chat="你好").build()
        assert config.title == "欺诈处理"

    def test_触发条件被追加(self):
        config = (
            JourneyDSL("T")
            .triggers("条件一", "条件二")
            .triggers("条件三")
            .state("A", chat="a")
            .build()
        )
        assert config.triggers == ["条件一", "条件二", "条件三"]

    def test_描述被设置(self):
        config = (
            JourneyDSL("T")
            .describe("这是一个测试流程")
            .state("A", chat="a")
            .build()
        )
        assert config.description == "这是一个测试流程"

    def test_无描述时默认空字符串(self):
        config = JourneyDSL("T").state("A", chat="a").build()
        assert config.description == ""


# ---------------------------------------------------------------------------
# JourneyDSL — 状态定义
# ---------------------------------------------------------------------------

class TestDSL状态定义:
    def test_基本对话状态(self):
        config = JourneyDSL("T").state("verify", chat="请验证身份").build()
        assert len(config.states) == 1
        s = config.states[0]
        assert s.name == "verify"
        assert s.chat_state == "请验证身份"
        assert s.canned_response is None

    def test_带描述的状态(self):
        config = JourneyDSL("T").state("A", description="身份验证节点", chat="验证").build()
        assert config.states[0].description == "身份验证节点"

    def test_带预设话术的状态(self):
        config = (
            JourneyDSL("T")
            .state("verify", chat="验证", canned="请提供您的身份信息。", strict=True)
            .build()
        )
        s = config.states[0]
        assert s.canned_response is not None
        assert s.canned_response.template == "请提供您的身份信息。"
        assert s.canned_response.strict is True

    def test_非严格话术模式(self):
        config = (
            JourneyDSL("T")
            .state("A", chat="a", canned="参考话术", strict=False)
            .build()
        )
        assert config.states[0].canned_response.strict is False

    def test_多个状态按顺序保存(self):
        config = (
            JourneyDSL("T")
            .state("A", chat="a")
            .state("B", chat="b")
            .state("C", chat="c")
            .build()
        )
        assert [s.name for s in config.states] == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# JourneyDSL — 规则定义
# ---------------------------------------------------------------------------

class TestDSL规则定义:
    def test_节点级规则被挂载到对应状态(self):
        config = (
            JourneyDSL("T")
            .state("A", chat="a")
            .rule("A", condition="情绪激动", action="先安抚", priority=90)
            .build()
        )
        assert len(config.states[0].guidelines) == 1
        g = config.states[0].guidelines[0]
        assert g.condition == "情绪激动"
        assert g.priority == 90

    def test_规则挂载到正确状态(self):
        config = (
            JourneyDSL("T")
            .state("A", chat="a")
            .state("B", chat="b")
            .rule("B", condition="条件X", action="动作X")
            .build()
        )
        assert config.states[0].guidelines == []  # A 无规则
        assert len(config.states[1].guidelines) == 1  # B 有规则

    def test_规则挂载到不存在状态时抛出异常(self):
        with pytest.raises(ValueError, match="Z"):
            JourneyDSL("T").state("A", chat="a").rule("Z", condition="x", action="y")

    def test_全局规则被正确保存(self):
        config = (
            JourneyDSL("T")
            .state("A", chat="a")
            .global_rule("情绪激动", "先安抚", priority=90)
            .global_rule("多次询问", "耐心解释", priority=60)
            .build()
        )
        assert len(config.global_guidelines) == 2
        assert config.global_guidelines[0].priority == 90


# ---------------------------------------------------------------------------
# JourneyDSL — 转移定义
# ---------------------------------------------------------------------------

class TestDSL转移定义:
    def test_无条件转移(self):
        config = (
            JourneyDSL("T")
            .state("A", chat="a").state("B", chat="b")
            .go("A", "B")
            .build()
        )
        assert len(config.transitions) == 1
        t = config.transitions[0]
        assert t.source == "A"
        assert t.target == "B"
        assert t.condition is None

    def test_有条件转移(self):
        config = (
            JourneyDSL("T")
            .state("A", chat="a").state("B", chat="b")
            .go("A", "B", when="用户已确认")
            .build()
        )
        assert config.transitions[0].condition == "用户已确认"

    def test_多条转移(self):
        config = (
            JourneyDSL("T")
            .state("A", chat="a").state("B", chat="b").state("C", chat="c")
            .go("A", "B", when="条件一")
            .go("A", "C", when="条件二")
            .build()
        )
        assert len(config.transitions) == 2


# ---------------------------------------------------------------------------
# JourneyDSL — 边界情况
# ---------------------------------------------------------------------------

class TestDSL边界情况:
    def test_空状态列表时抛出ValueError(self):
        with pytest.raises(ValueError, match="状态"):
            JourneyDSL("T").build()

    def test_空标题时抛出ValueError(self):
        with pytest.raises(ValueError, match="标题"):
            JourneyDSL("").state("A", chat="a").build()

    def test_build返回独立的JourneyConfig对象(self):
        """多次 build() 应返回独立对象，互不干扰。"""
        dsl = JourneyDSL("T").triggers("触发").state("A", chat="a")
        c1 = dsl.build()
        c1.triggers.append("额外触发")
        c2 = dsl.build()
        assert len(c2.triggers) == 1  # c2 不受 c1 的修改影响


# ---------------------------------------------------------------------------
# load_journey_dict — 字典加载
# ---------------------------------------------------------------------------

_最小字典 = {
    "title": "测试流程",
    "states": [{"name": "A", "chat_state": "你好"}],
}


class TestLoader字典加载:
    def test_最小字典加载成功(self):
        config = load_journey_dict(_最小字典)
        assert isinstance(config, JourneyConfig)
        assert config.title == "测试流程"
        assert len(config.states) == 1
        assert config.states[0].name == "A"

    def test_触发条件被正确解析(self):
        data = {**_最小字典, "triggers": ["触发一", "触发二"]}
        config = load_journey_dict(data)
        assert config.triggers == ["触发一", "触发二"]

    def test_转移被正确解析(self):
        data = {
            "title": "T",
            "states": [
                {"name": "A", "chat_state": "a"},
                {"name": "B", "chat_state": "b"},
            ],
            "transitions": [{"source": "A", "target": "B", "condition": "确认"}],
        }
        config = load_journey_dict(data)
        assert len(config.transitions) == 1
        assert config.transitions[0].condition == "确认"

    def test_预设话术被正确解析(self):
        data = {
            "title": "T",
            "states": [{
                "name": "A",
                "canned_response": {"template": "锁定回复", "strict": True},
            }],
        }
        config = load_journey_dict(data)
        cr = config.states[0].canned_response
        assert cr is not None
        assert cr.template == "锁定回复"
        assert cr.strict is True

    def test_全局规则被正确解析(self):
        data = {
            **_最小字典,
            "global_guidelines": [
                {"condition": "情绪激动", "action": "先安抚", "priority": 90}
            ],
        }
        config = load_journey_dict(data)
        assert len(config.global_guidelines) == 1
        assert config.global_guidelines[0].priority == 90

    def test_缺少title时抛出ValueError(self):
        with pytest.raises(ValueError, match="title"):
            load_journey_dict({"states": [{"name": "A"}]})

    def test_缺少states时抛出ValueError(self):
        with pytest.raises(ValueError, match="states"):
            load_journey_dict({"title": "T"})

    def test_状态缺少name时抛出ValueError(self):
        with pytest.raises(ValueError, match="name"):
            load_journey_dict({"title": "T", "states": [{"chat_state": "a"}]})

    def test_转移缺少source时抛出ValueError(self):
        with pytest.raises(ValueError, match="source"):
            load_journey_dict({
                "title": "T",
                "states": [{"name": "A"}],
                "transitions": [{"target": "B"}],
            })


# ---------------------------------------------------------------------------
# load_journey — JSON 文件加载
# ---------------------------------------------------------------------------

class TestLoaderJSON文件:
    def test_JSON文件加载成功(self, tmp_path):
        data = {
            "title": "JSON流程",
            "description": "从文件加载",
            "triggers": ["触发"],
            "states": [{"name": "A", "chat_state": "你好"}],
            "transitions": [],
        }
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        config = load_journey(str(path))
        assert config.title == "JSON流程"
        assert config.description == "从文件加载"

    def test_文件不存在时抛出FileNotFoundError(self):
        with pytest.raises(FileNotFoundError):
            load_journey("/tmp/不存在的文件_xyz123.json")


# ---------------------------------------------------------------------------
# load_journey — YAML 文件加载（需要 pyyaml）
# ---------------------------------------------------------------------------

class TestLoaderYAML文件:
    @pytest.fixture(autouse=True)
    def skip_without_yaml(self):
        """若未安装 pyyaml，跳过本类所有测试。"""
        pytest.importorskip("yaml", reason="需要安装 pyyaml：pip install pyyaml")

    def test_YAML文件加载成功(self, tmp_path):
        yaml_text = """\
title: "YAML流程"
description: "YAML 加载测试"
triggers:
  - "触发一"
states:
  - name: verify
    description: "验证"
    chat_state: "请验证身份"
    canned_response:
      template: "请提供身份信息。"
      strict: true
transitions: []
global_guidelines:
  - condition: "情绪激动"
    action: "先安抚"
    priority: 80
"""
        path = tmp_path / "test.yaml"
        path.write_text(yaml_text, encoding="utf-8")

        config = load_journey(str(path))
        assert config.title == "YAML流程"
        assert config.triggers == ["触发一"]
        assert config.states[0].canned_response.strict is True
        assert config.global_guidelines[0].priority == 80

    def test_与Python硬编码配置等价(self, tmp_path):
        """验证 YAML 加载结果与 Python 代码等价（用于回归保证）。"""
        yaml_text = """\
title: "等价测试"
triggers: ["go"]
states:
  - name: A
    chat_state: "执行A"
  - name: B
    chat_state: "执行B"
transitions:
  - source: A
    target: B
    condition: "用户确认"
"""
        path = tmp_path / "equiv.yaml"
        path.write_text(yaml_text, encoding="utf-8")
        config = load_journey(str(path))

        assert config.title == "等价测试"
        assert len(config.states) == 2
        assert config.transitions[0].condition == "用户确认"

    def test_fraud_yaml示例文件可被正确加载(self):
        """验证 tests/fixtures/fraud.yaml 示例文件可被成功解析。"""
        yaml_path = _FIXTURES / "fraud.yaml"
        if not yaml_path.exists():
            pytest.skip("tests/fixtures/fraud.yaml 不存在，跳过")

        config = load_journey(str(yaml_path))
        assert config.title == "处理潜在欺诈交易"
        assert len(config.states) == 7
        assert len(config.transitions) == 9
        assert len(config.global_guidelines) == 3


# ---------------------------------------------------------------------------
# load_journey — fraud.json 示例文件加载
# ---------------------------------------------------------------------------

class TestLoaderFraudJSON示例:
    def test_fraud_json与yaml内容等价(self):
        """验证 tests/fixtures/fraud.json 与 tests/fixtures/fraud.yaml 解析结果等价。"""
        json_path = _FIXTURES / "fraud.json"
        if not json_path.exists():
            pytest.skip("tests/fixtures/fraud.json 不存在，跳过")

        config = load_journey(str(json_path))
        assert config.title == "处理潜在欺诈交易"
        assert len(config.states) == 7
        assert len(config.transitions) == 9
        assert len(config.global_guidelines) == 3

    def test_fraud_json状态名正确(self):
        """验证 fraud.json 中的状态名与 fraud.yaml 一致。"""
        json_path = _FIXTURES / "fraud.json"
        if not json_path.exists():
            pytest.skip("tests/fixtures/fraud.json 不存在，跳过")

        config = load_journey(str(json_path))
        state_names = [s.name for s in config.states]
        assert state_names == [
            "identity_verify", "collect_info", "risk_assess",
            "high_risk", "medium_risk", "low_risk", "closing",
        ]

    def test_fraud_json高风险状态有节点规则(self):
        """验证 fraud.json 中 high_risk 状态含有 2 条节点级规则。"""
        json_path = _FIXTURES / "fraud.json"
        if not json_path.exists():
            pytest.skip("tests/fixtures/fraud.json 不存在，跳过")

        config = load_journey(str(json_path))
        high_risk = next(s for s in config.states if s.name == "high_risk")
        assert len(high_risk.guidelines) == 2
        assert high_risk.guidelines[0].priority == 95

    def test_fraud_json预设话术正确解析(self):
        """验证 identity_verify 状态的 canned_response 被正确解析。"""
        json_path = _FIXTURES / "fraud.json"
        if not json_path.exists():
            pytest.skip("tests/fixtures/fraud.json 不存在，跳过")

        config = load_journey(str(json_path))
        verify = next(s for s in config.states if s.name == "identity_verify")
        assert verify.canned_response is not None
        assert verify.canned_response.strict is True
        assert "身份" in verify.canned_response.template


# ---------------------------------------------------------------------------
# 直接运行支持：python tests/04_test_dsl.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

