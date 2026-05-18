"""
journey.models
~~~~~~~~~~~~~~

纯数据层 — 无任何副作用，无外部依赖。

三个正交控制维度：

  流程（路径）   states + transitions       → transition_to()
  行为（规则）   guidelines（节点级 + 全局） → journey.create_guideline()
  话术（合规）   canned_response（节点级）   → canned_responses= + CompositionMode
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class CannedResponseConfig:
    """关键合规节点的预设话术配置。

    Args:
        template: 强制输出的话术模板文本。
        strict:   True（默认）→ CompositionMode.STRICT，彻底锁定输出，零幻觉；
                  False        → CompositionMode.FLUID，允许 Agent 在话术基础上调整。
    """

    template: str
    strict: bool = True


@dataclass
class GuidelineConfig:
    """行为规则配置：在满足 condition 时执行 action。

    可挂载在单个状态（节点级）或整个 Journey（全局级）。

    Args:
        condition: 触发规则的条件描述。
        action:    满足条件时 Agent 应执行的动作描述。
        priority:  规则优先级（0–100），值越大越优先生效，默认 50。
    """

    condition: str
    action: str
    priority: int = 50


@dataclass
class StateConfig:
    """单个 Journey 状态节点的声明式配置。

    ``chat_state`` 与 ``tool_state`` 二选一：
      - chat_state:  对话状态，提供自然语言指令给 Agent。
      - tool_state:  工具状态，触发指定工具函数调用。

    若两者均为 None，则以 description 或 name 作为 chat_state 的兜底值。

    Args:
        name:             状态唯一标识符。
        description:      可读描述，用于日志和 system prompt 显示。
        chat_state:       对话状态的行为指令文本。
        tool_state:       工具状态对象（与 chat_state 互斥）。
        tool_instruction: 工具调用的补充说明（仅 tool_state 模式有效）。
        guidelines:       节点级行为规则列表。
        canned_response:  预设话术配置（可选）。
    """

    name: str
    description: str = ""
    chat_state: Optional[str] = None
    tool_state: Optional[Any] = None
    tool_instruction: Optional[str] = None
    guidelines: List[GuidelineConfig] = field(default_factory=list)
    canned_response: Optional[CannedResponseConfig] = None


@dataclass
class TransitionConfig:
    """Journey DAG 中的一条有向边（状态转移规则）。

    Args:
        source:    源状态名称。
        target:    目标状态名称。
        condition: 触发本次转移的条件描述；None 表示无条件自动转移。

    注意：
        ``JourneyConfig.transitions`` 列表须满足拓扑顺序
        （父节点的边先于子节点的边出现）。
    """

    source: str
    target: str
    condition: Optional[str] = None


@dataclass
class JourneyConfig:
    """完整 Journey 的顶层声明式配置。

    Args:
        title:             Journey 名称，用于 Parlant 识别。
        triggers:          触发本 Journey 的条件列表（例如用户意图、关键词）。
        description:       Journey 的整体描述。
        states:            所有状态的配置，须覆盖 transitions 中出现的每个状态名。
        transitions:       有向图边集合（须按拓扑顺序排列）。
        global_guidelines: Journey 级别的全局行为规则（作用于所有状态）。
    """

    title: str
    triggers: List[str]
    description: str
    states: List[StateConfig]
    transitions: List[TransitionConfig]
    global_guidelines: List[GuidelineConfig] = field(default_factory=list)

