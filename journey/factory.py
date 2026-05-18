"""
journey.factory
~~~~~~~~~~~~~~~

Public entry-point layer.

Thin wrappers that instantiate :class:`~journey.builder.JourneyBuilder`
and drive it to completion, hiding the builder from callers entirely.
"""

from __future__ import annotations

from typing import List

import parlant.sdk as p

from .builder import JourneyBuilder
from .models import JourneyConfig


async def create_journey(agent: p.Agent, config: JourneyConfig) -> p.Journey:
    """Create a single Parlant Journey from a declarative :class:`~journey.models.JourneyConfig`.

    Args:
        agent:  An already-created Parlant :class:`~parlant.sdk.Agent`.
        config: Fully populated :class:`~journey.models.JourneyConfig`.

    Returns:
        The constructed Parlant :class:`~parlant.sdk.Journey` instance.

    Example::

        journey = await create_journey(agent, fraud_config)
    """
    return await JourneyBuilder(agent, config).build()


async def register_journeys(
    agent: p.Agent,
    configs: List[JourneyConfig],
) -> List[p.Journey]:
    """Create multiple Journeys in sequence and return them as a list.

    Args:
        agent:   An already-created Parlant :class:`~parlant.sdk.Agent`.
        configs: Ordered list of :class:`~journey.models.JourneyConfig`.

    Returns:
        List of constructed :class:`~parlant.sdk.Journey` instances in the same
        order as *configs*.

    Example::

        journeys = await register_journeys(agent, [fraud_config, loan_config])
    """
    return [await create_journey(agent, cfg) for cfg in configs]
