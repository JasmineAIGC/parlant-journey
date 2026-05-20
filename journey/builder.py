"""
journey.builder
~~~~~~~~~~~~~~~

内部翻译层：将 JourneyConfig 转换为 Parlant Journey API 调用序列。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import parlant.sdk as p

from .models import JourneyConfig, StateConfig

logger = logging.getLogger(__name__)


class JourneyBuilder:
    """将 :class:`~journey.models.JourneyConfig` 翻译为 Parlant Journey 的内部构建器。"""

    def __init__(self, agent: p.Agent, config: JourneyConfig) -> None:
        self._agent = agent
        self._config = config
        self._journey: Optional[p.Journey] = None

        # 状态名 → Parlant CannedResponse 对象（步骤②预创建）
        self._canrep_map: Dict[str, Any] = {}

        # 状态名 → Parlant 状态对象（用于后续 transition_to() 调用）
        self._state_obj_map: Dict[str, Any] = {}

        # 快速查找：状态名 → StateConfig
        self._state_cfg_map: Dict[str, StateConfig] = {
            s.name: s for s in config.states
        }

    # ------------------------------------------------------------------
    # 公开入口
    # ------------------------------------------------------------------

    async def build(self) -> p.Journey:
        """按顺序执行所有构建步骤，返回构建完成的 Journey 对象。"""
        logger.debug("[JourneyBuilder] 开始构建 Journey  title=%r", self._config.title)
        await self._create_journey()
        await self._create_canned_responses()
        await self._build_dag()
        await self._register_node_guidelines()
        await self._register_global_guidelines()
        assert self._journey is not None
        logger.debug("[JourneyBuilder] Journey 构建完成  title=%r  id=%s", self._config.title, self._journey.id)
        return self._journey

    # ------------------------------------------------------------------
    # 步骤①：创建 Journey 外壳
    # ------------------------------------------------------------------

    async def _create_journey(self) -> None:
        logger.debug("[JourneyBuilder] 步骤①：创建 Journey 外壳  title=%r", self._config.title)
        self._journey = await self._agent.create_journey(
            title=self._config.title,
            description=self._config.description,
            conditions=self._config.conditions,
        )
        logger.debug("[JourneyBuilder] Journey 外壳已创建  id=%s", self._journey.id)

    # ------------------------------------------------------------------
    # 步骤②：预创建所有预设话术
    # ------------------------------------------------------------------

    async def _create_canned_responses(self) -> None:
        """为配置了 canned_response 的状态预创建 CannedResponse 对象，存入 _canrep_map。"""
        canned_count = sum(1 for s in self._config.states if s.canned_response is not None)
        logger.debug("[JourneyBuilder] 步骤②：预创建预设话术  count=%d", canned_count)
        for state in self._config.states:
            if state.canned_response is not None:
                canned = await self._agent.create_canned_response(
                    template=state.canned_response.template,
                )
                self._canrep_map[state.name] = canned
                logger.debug("[JourneyBuilder] CannedResponse 已创建  state=%r", state.name)

    # ------------------------------------------------------------------
    # 步骤③：构建状态 DAG
    # ------------------------------------------------------------------

    async def _build_dag(self) -> None:
        """按调用方提供的拓扑顺序遍历 transitions，逐步创建 Parlant 状态节点。

        自动处理三种拓扑结构：

          线性    A → B → C
          分支    A → B（条件 X）  /  A → C（条件 Y）
          汇聚    B → D  /  C → D  （D 复用同一对象，不重复创建）

        根节点（无入边的状态）直接挂在 journey.initial_state 下。
        """
        assert self._journey is not None
        logger.debug(
            "[JourneyBuilder] 步骤③：构建状态 DAG  states=%d  transitions=%d",
            len(self._config.states),
            len(self._config.transitions),
        )

        # 计算所有作为 target 的状态名
        all_targets = {t.target for t in self._config.transitions}
        all_state_names = {s.name for s in self._config.states}

        # 根节点：从未出现在 target 中的状态（即无入边的状态）
        roots = all_state_names - all_targets
        logger.debug("[JourneyBuilder] 根节点  roots=%s", sorted(roots))

        # ── 从 journey.initial_state 创建所有根状态 ──────────────────
        for root_name in roots:
            root_cfg = self._state_cfg_map.get(root_name)
            if root_cfg is None:
                raise ValueError(
                    f"根状态 '{root_name}' 没有对应的 StateConfig。"
                    "请确保 JourneyConfig.states 中包含所有在 transitions 中出现的状态。"
                )
            kwargs = self._build_state_kwargs(root_cfg)
            root_trans = await self._journey.initial_state.transition_to(**kwargs)
            self._state_obj_map[root_name] = root_trans.target
            logger.debug("[JourneyBuilder] 根状态已创建  state=%r", root_name)

        # ── 按拓扑顺序处理所有转移边 ─────────────────────────────────
        for trans in self._config.transitions:
            source_obj = self._state_obj_map.get(trans.source)
            if source_obj is None:
                raise ValueError(
                    f"转移的源状态 '{trans.source}' 尚未创建。"
                    "请确保 JourneyConfig.transitions 按拓扑顺序排列。"
                )

            # 有条件转移时携带 condition 参数
            condition_kwargs: Dict[str, Any] = (
                {"condition": trans.condition} if trans.condition is not None else {}
            )

            if trans.target in self._state_obj_map:
                # 汇聚场景：目标状态已存在，直接复用（不重复创建）
                logger.debug(
                    "[JourneyBuilder] 复用已有状态（汇聚）  %r → %r  condition=%r",
                    trans.source, trans.target, trans.condition,
                )
                await source_obj.transition_to(
                    state=self._state_obj_map[trans.target],
                    **condition_kwargs,
                )
            else:
                # 普通场景：目标状态尚不存在，通过 transition_to 创建
                target_cfg = self._state_cfg_map.get(trans.target)
                if target_cfg is None:
                    raise ValueError(
                        f"目标状态 '{trans.target}' 没有对应的 StateConfig。"
                        "请确保 JourneyConfig.states 中包含所有在 transitions 中出现的状态。"
                    )
                kwargs = self._build_state_kwargs(target_cfg, **condition_kwargs)
                new_trans = await source_obj.transition_to(**kwargs)
                self._state_obj_map[trans.target] = new_trans.target
                logger.debug(
                    "[JourneyBuilder] 状态已创建  %r → %r  condition=%r",
                    trans.source, trans.target, trans.condition,
                )

    # ------------------------------------------------------------------
    # 步骤④：注册节点级行为规则
    # ------------------------------------------------------------------

    async def _register_node_guidelines(self) -> None:
        """遍历所有状态，逐条注册节点级 guideline。"""
        assert self._journey is not None
        total = sum(len(s.guidelines) for s in self._config.states)
        logger.debug("[JourneyBuilder] 步骤④：注册节点级指南  total=%d", total)
        for state in self._config.states:
            await self._register_guidelines(state.guidelines)

    # ------------------------------------------------------------------
    # 步骤⑤：注册全局行为规则
    # ------------------------------------------------------------------

    async def _register_global_guidelines(self) -> None:
        """注册 JourneyConfig.global_guidelines 中的所有全局规则。"""
        logger.debug(
            "[JourneyBuilder] 步骤⑤：注册全局指南  count=%d",
            len(self._config.global_guidelines),
        )
        await self._register_guidelines(self._config.global_guidelines)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    async def _register_guidelines(self, guidelines: list) -> None:
        """将 guideline 列表逐条注册到当前 Journey。"""
        assert self._journey is not None
        for guideline in guidelines:
            await self._journey.create_guideline(
                condition=guideline.condition,
                action=guideline.action,
                priority=guideline.priority,
            )

    def _build_state_kwargs(
        self,
        state_cfg: StateConfig,
        **extra: Any,
    ) -> Dict[str, Any]:
        """将 StateConfig 转换为 transition_to() 所需的关键字参数字典。"""
        kwargs: Dict[str, Any] = dict(extra)

        # 携带可读描述
        if state_cfg.description:
            kwargs["description"] = state_cfg.description

        # 挂载预设话术（步骤②已预创建）
        if state_cfg.name in self._canrep_map:
            kwargs["canned_responses"] = [self._canrep_map[state_cfg.name]]
            composition_mode = (
                p.CompositionMode.STRICT
                if state_cfg.canned_response and state_cfg.canned_response.strict
                else p.CompositionMode.FLUID
            )
            kwargs["composition_mode"] = composition_mode

        # 工具状态 vs 对话状态
        if state_cfg.tool_state is not None:
            kwargs["tool_state"] = state_cfg.tool_state
            if state_cfg.tool_instruction is not None:
                kwargs["tool_instruction"] = state_cfg.tool_instruction
        else:
            # chat_state 兜底：优先用显式值，其次 description，再次 name
            kwargs["chat_state"] = (
                state_cfg.chat_state
                if state_cfg.chat_state is not None
                else state_cfg.description or state_cfg.name
            )

        return kwargs
