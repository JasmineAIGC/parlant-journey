"""
journey.dsl
~~~~~~~~~~~

流式 DSL 构建器，让调用方无需手动操作 dataclass 即可快速构建 JourneyConfig。

适用场景：
  - 代码内快速搭建新场景（比直接拼 dataclass 更简洁）
  - 原型验证、脚本化配置

用法示例::

    from journey.dsl import JourneyDSL

    config = (
        JourneyDSL("欺诈处理")
        .triggers("我卡被盗刷了", "可疑交易")
        .describe("处理欺诈交易的标准流程")
        .state("verify", chat="请验证您的身份",
               canned="为了账户安全，请提供您的姓名和身份证后四位。", strict=True)
        .state("closing", chat="感谢您的配合")
        .go("verify", "closing", when="验证成功")
        .global_rule("用户情绪激动", "先安抚情绪", priority=90)
        .build()
    )
"""

from __future__ import annotations

from typing import Any, List, Optional

from .models import (
    CannedResponseConfig,
    GuidelineConfig,
    JourneyConfig,
    StateConfig,
    TransitionConfig,
)


class JourneyDSL:
    """流式 Journey 配置构建器。

    通过链式调用逐步组装 :class:`~journey.models.JourneyConfig`，
    避免直接构造多层嵌套 dataclass。

    调用顺序建议::

        JourneyDSL(title)
          .triggers(...)      # 触发条件
          .describe(...)      # 整体描述
          .state(...)         # 添加状态（可多次调用）
          .rule(...)          # 为某状态添加节点级规则（可多次调用）
          .go(...)            # 添加状态转移（可多次调用）
          .global_rule(...)   # 添加全局规则（可多次调用）
          .build()            # 生成 JourneyConfig
    """

    def __init__(self, title: str) -> None:
        """
        Args:
            title: Journey 名称，用于 Parlant 识别。
        """
        self._title = title
        self._triggers: List[str] = []
        self._description: str = ""
        self._states: List[StateConfig] = []
        self._states_map: dict = {}         # 状态名 → StateConfig，用于 O(1) 查找
        self._transitions: List[TransitionConfig] = []
        self._global_guidelines: List[GuidelineConfig] = []

    # ------------------------------------------------------------------
    # 基础属性设置
    # ------------------------------------------------------------------

    def triggers(self, *items: str) -> "JourneyDSL":
        """追加一个或多个触发条件。

        Args:
            *items: 触发本 Journey 的条件字符串（可传多个）。
        """
        self._triggers.extend(items)
        return self

    def describe(self, description: str) -> "JourneyDSL":
        """设置 Journey 的整体描述文本。

        Args:
            description: 描述本 Journey 用途的自然语言文本。
        """
        self._description = description
        return self

    # ------------------------------------------------------------------
    # 状态定义
    # ------------------------------------------------------------------

    def state(
        self,
        name: str,
        *,
        description: str = "",
        chat: Optional[str] = None,
        tool: Optional[Any] = None,
        tool_instruction: Optional[str] = None,
        canned: Optional[str] = None,
        strict: bool = True,
    ) -> "JourneyDSL":
        """添加一个 Journey 状态节点。

        ``chat`` 与 ``tool`` 二选一：
          - ``chat``  — 对话状态，提供自然语言行为指令给 Agent。
          - ``tool``  — 工具状态，绑定工具函数。

        Args:
            name:             状态唯一标识符（在 :meth:`go` 的 source/target 中引用）。
            description:      可读描述，用于日志和 system prompt 提示。
            chat:             对话状态指令文本（chat_state）。
            tool:             工具状态对象（tool_state），与 chat 互斥。
            tool_instruction: 工具调用补充说明（仅 tool 模式有效）。
            canned:           预设话术模板文本；为 None 则不启用话术锁定。
            strict:           True（默认）→ STRICT 模式，完全锁定输出；
                              False → FLUID 模式，允许 Agent 在话术基础上调整。
        """
        canned_cfg = CannedResponseConfig(template=canned, strict=strict) if canned else None
        s = StateConfig(
            name=name,
            description=description,
            chat_state=chat,
            tool_state=tool,
            tool_instruction=tool_instruction,
            canned_response=canned_cfg,
        )
        self._states.append(s)
        self._states_map[name] = s
        return self

    # ------------------------------------------------------------------
    # 规则定义
    # ------------------------------------------------------------------

    def rule(
        self,
        state_name: str,
        condition: str,
        action: str,
        priority: int = 50,
    ) -> "JourneyDSL":
        """为指定状态追加一条节点级行为规则。

        须在对应的 :meth:`state` 调用之后调用。

        Args:
            state_name: 目标状态名称（须已通过 :meth:`state` 添加）。
            condition:  触发规则的条件描述。
            action:     满足条件时 Agent 应执行的动作描述。
            priority:   规则优先级（0–100，默认 50）。

        Raises:
            ValueError: 若 state_name 对应的状态尚未添加。
        """
        s = self._states_map.get(state_name)
        if s is not None:
            s.guidelines.append(GuidelineConfig(
                condition=condition, action=action, priority=priority
            ))
            return self
        raise ValueError(
            f"状态 '{state_name}' 不存在，请先通过 .state('{state_name}', ...) 添加该状态。"
        )

    def global_rule(
        self,
        condition: str,
        action: str,
        priority: int = 50,
    ) -> "JourneyDSL":
        """追加一条全局行为规则（对整个 Journey 生效）。

        Args:
            condition: 触发规则的条件描述。
            action:    满足条件时 Agent 应执行的动作描述。
            priority:  规则优先级（0–100，默认 50）。
        """
        self._global_guidelines.append(GuidelineConfig(
            condition=condition, action=action, priority=priority
        ))
        return self

    # ------------------------------------------------------------------
    # 转移定义
    # ------------------------------------------------------------------

    def go(
        self,
        source: str,
        target: str,
        *,
        when: Optional[str] = None,
    ) -> "JourneyDSL":
        """添加一条状态转移边。

        Args:
            source: 源状态名称。
            target: 目标状态名称。
            when:   触发条件；为 None 表示无条件自动转移。
        """
        self._transitions.append(TransitionConfig(
            source=source, target=target, condition=when
        ))
        return self

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------

    def build(self) -> JourneyConfig:
        """构建并返回最终的 :class:`~journey.models.JourneyConfig` 对象。

        每次调用都返回一个新的独立对象：列表字段使用 list() 浅拷贝，
        避免后续对 DSL 实例的修改影响已构建的 config（防御性编程）。

        Raises:
            ValueError: 若 title 或 states 为空。
        """
        if not self._title:
            raise ValueError("Journey 标题不能为空。")
        if not self._states:
            raise ValueError("Journey 至少需要定义一个状态，请先调用 .state(...)。")
        return JourneyConfig(
            title=self._title,
            triggers=list(self._triggers),          # 浅拷贝，隔离后续 DSL 修改
            description=self._description,
            states=list(self._states),              # 浅拷贝
            transitions=list(self._transitions),    # 浅拷贝
            global_guidelines=list(self._global_guidelines),  # 浅拷贝
        )
