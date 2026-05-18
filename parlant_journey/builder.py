"""
parlant_journey.builder
~~~~~~~~~~~~~~~~~~~~~~~

Translation layer: JourneyConfig → Parlant Journey API calls.

Internal implementation detail — not exposed in the public API.

Data flow (ARCHITECTURE_Version2.md §2.5):

    factory.create_journey(agent, config)
      └─ JourneyBuilder(agent, config).build()
           ├─ ① _create_journey()
           ├─ ② _create_canned_responses()
           ├─ ③ _build_dag()
           ├─ ④ _register_node_guidelines()
           └─ ⑤ _register_global_guidelines()
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import parlant.sdk as p

from .models import JourneyConfig, StateConfig


class JourneyBuilder:
    """Translates a :class:`~parlant_journey.models.JourneyConfig` into a live
    Parlant :class:`~parlant.sdk.Journey` by issuing the appropriate SDK calls
    in the correct order.

    This class is an **internal implementation detail** and must not be imported
    directly by callers — use :func:`~parlant_journey.factory.create_journey`
    instead.
    """

    def __init__(self, agent: p.Agent, config: JourneyConfig) -> None:
        self._agent = agent
        self._config = config
        self._journey: Optional[p.Journey] = None

        # state_name → p.CannedResponse
        self._canrep_map: Dict[str, Any] = {}

        # state_name → Parlant state object (what you call transition_to() on)
        self._state_obj_map: Dict[str, Any] = {}

        # fast lookup: state_name → StateConfig
        self._state_cfg_map: Dict[str, StateConfig] = {
            s.name: s for s in config.states
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def build(self) -> p.Journey:
        """Execute all build steps and return the constructed Journey."""
        await self._create_journey()
        await self._create_canned_responses()
        await self._build_dag()
        await self._register_node_guidelines()
        await self._register_global_guidelines()
        assert self._journey is not None
        return self._journey

    # ------------------------------------------------------------------
    # Step ①: create the Journey shell
    # ------------------------------------------------------------------

    async def _create_journey(self) -> None:
        self._journey = await self._agent.create_journey(
            title=self._config.title,
            triggers=self._config.triggers,
            description=self._config.description,
        )

    # ------------------------------------------------------------------
    # Step ②: pre-create all canned responses
    # ------------------------------------------------------------------

    async def _create_canned_responses(self) -> None:
        """Pre-create every CannedResponse and store in ``_canrep_map``."""
        for state in self._config.states:
            if state.canned_response is not None:
                canned = await self._agent.create_canned_response(
                    template=state.canned_response.template,
                )
                self._canrep_map[state.name] = canned

    # ------------------------------------------------------------------
    # Step ③: build the state DAG
    # ------------------------------------------------------------------

    async def _build_dag(self) -> None:
        """Walk ``config.transitions`` (in caller-provided topological order)
        and create Parlant states / transitions accordingly.

        Three topologies are handled automatically:

          线性    A → B → C
          分支    A → B (cond X)  /  A → C (cond Y)
          汇聚    B → D  /  C → D  (D reused, not duplicated)

        Root states (sources with no incoming edges) are created as direct
        children of ``journey.initial_state``.
        """
        assert self._journey is not None

        all_targets = {t.target for t in self._config.transitions}
        all_state_names = {s.name for s in self._config.states}
        # Root states: any state that is never a *target* of a transition.
        # This covers the common case (single root in a linear/branching DAG) as
        # well as isolated states (configs with no transitions at all).
        roots = all_state_names - all_targets

        # ---- create root states from journey.initial_state ----------------
        for root_name in roots:
            root_cfg = self._state_cfg_map.get(root_name)
            if root_cfg is None:
                raise ValueError(
                    f"No StateConfig found for root state '{root_name}'. "
                    "Every state referenced in transitions must be listed in "
                    "JourneyConfig.states."
                )
            kwargs = self._build_state_kwargs(root_cfg)
            root_trans = await self._journey.initial_state.transition_to(**kwargs)
            self._state_obj_map[root_name] = root_trans.target

        # ---- process all transitions in the declared topological order ----
        for trans in self._config.transitions:
            source_obj = self._state_obj_map.get(trans.source)
            if source_obj is None:
                raise ValueError(
                    f"Transition source '{trans.source}' has not been created yet. "
                    "Ensure JourneyConfig.transitions is listed in topological order."
                )

            condition_kwargs: Dict[str, Any] = (
                {"condition": trans.condition} if trans.condition is not None else {}
            )

            if trans.target in self._state_obj_map:
                # Convergence: reuse the already-created state object
                await source_obj.transition_to(
                    state=self._state_obj_map[trans.target],
                    **condition_kwargs,
                )
            else:
                # New state: create via transition
                target_cfg = self._state_cfg_map.get(trans.target)
                if target_cfg is None:
                    raise ValueError(
                        f"No StateConfig found for state '{trans.target}'. "
                        "Every state referenced in transitions must be listed in "
                        "JourneyConfig.states."
                    )
                kwargs = self._build_state_kwargs(target_cfg, **condition_kwargs)
                new_trans = await source_obj.transition_to(**kwargs)
                self._state_obj_map[trans.target] = new_trans.target

    # ------------------------------------------------------------------
    # Step ④: register per-state guidelines
    # ------------------------------------------------------------------

    async def _register_node_guidelines(self) -> None:
        assert self._journey is not None
        for state in self._config.states:
            for guideline in state.guidelines:
                await self._journey.create_guideline(
                    condition=guideline.condition,
                    action=guideline.action,
                    priority=guideline.priority,
                )

    # ------------------------------------------------------------------
    # Step ⑤: register global guidelines
    # ------------------------------------------------------------------

    async def _register_global_guidelines(self) -> None:
        assert self._journey is not None
        for guideline in self._config.global_guidelines:
            await self._journey.create_guideline(
                condition=guideline.condition,
                action=guideline.action,
                priority=guideline.priority,
            )

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _build_state_kwargs(
        self,
        state_cfg: StateConfig,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Assemble the keyword arguments for a ``transition_to()`` call that
        creates *state_cfg* as a new Parlant state."""
        kwargs: Dict[str, Any] = dict(extra)

        if state_cfg.description:
            kwargs["description"] = state_cfg.description

        # Attach canned response when pre-created
        if state_cfg.name in self._canrep_map:
            kwargs["canned_responses"] = [self._canrep_map[state_cfg.name]]
            composition_mode = (
                p.CompositionMode.STRICT
                if state_cfg.canned_response and state_cfg.canned_response.strict
                else p.CompositionMode.FLUID
            )
            kwargs["composition_mode"] = composition_mode

        # Tool state or chat state
        if state_cfg.tool_state is not None:
            kwargs["tool_state"] = state_cfg.tool_state
            if state_cfg.tool_instruction is not None:
                kwargs["tool_instruction"] = state_cfg.tool_instruction
        else:
            kwargs["chat_state"] = (
                state_cfg.chat_state
                if state_cfg.chat_state is not None
                else state_cfg.description or state_cfg.name
            )

        return kwargs
