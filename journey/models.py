"""
journey.models
~~~~~~~~~~~~~~

Pure data layer — zero side-effects, zero external dependencies.

Three orthogonal control dimensions:

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
    """

    condition: str
    action: str
    priority: int = 50


@dataclass
class StateConfig:
    """单个 Journey 状态的声明式配置。

    chat_state 与 tool_state 二选一：
      - chat_state:  对话状态，提供自然语言指令给 Agent。
      - tool_state:  工具状态，触发指定工具函数调用。

    若两者均为 None，则以 description 或 name 作为 chat_state 的 fallback。
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
    """Journey DAG 中的一条有向边。

    Args:
        source:    源状态名称。
        target:    目标状态名称。
        condition: 触发本次迁移的条件（None 表示无条件迁移）。

    Note:
        ``JourneyConfig.transitions`` 列表须满足拓扑顺序（父节点的边先于子节点的边）。
    """

    source: str
    target: str
    condition: Optional[str] = None


@dataclass
class JourneyConfig:
    """完整 Journey 的声明式配置（顶层入口）。

    Args:
        title:             Journey 名称，用于 Parlant 识别。
        triggers:          触发本 Journey 的条件列表（例如用户意图、关键词）。
        description:       Journey 的整体描述。
        states:            所有状态的配置，须覆盖 transitions 中出现的每个目标状态。
        transitions:       有向图的边集合（拓扑顺序）。
        global_guidelines: Journey 级别全局行为规则（优先级通常低于节点级规则）。
    """

    title: str
    triggers: List[str]
    description: str
    states: List[StateConfig]
    transitions: List[TransitionConfig]
    global_guidelines: List[GuidelineConfig] = field(default_factory=list)
