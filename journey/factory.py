"""
journey.factory
~~~~~~~~~~~~~~~

公开入口层：对外提供简洁的 async 函数，内部实例化 JourneyBuilder 并驱动其完成构建。
调用方无需关心 JourneyBuilder 的存在。
"""

from __future__ import annotations

import logging
from typing import List

import parlant.sdk as p

from .builder import JourneyBuilder
from .models import JourneyConfig

logger = logging.getLogger(__name__)


async def create_journey(agent: p.Agent, config: JourneyConfig) -> p.Journey:
    """从声明式配置创建单个 Parlant Journey。

    Args:
        agent:  已创建的 Parlant :class:`~parlant.sdk.Agent` 对象。
        config: 填充完毕的 :class:`~journey.models.JourneyConfig` 对象。

    Returns:
        构建完成的 Parlant :class:`~parlant.sdk.Journey` 实例。

    示例::

        journey = await create_journey(agent, fraud_config)
    """
    logger.debug("[create_journey] 开始  title=%r", config.title)
    try:
        journey = await JourneyBuilder(agent, config).build()
    except Exception:
        logger.error("[create_journey] 构建失败  title=%r", config.title, exc_info=True)
        raise
    logger.debug("[create_journey] 完成  title=%r  id=%s", config.title, journey.id)
    return journey


async def register_journeys(
    agent: p.Agent,
    configs: List[JourneyConfig],
) -> List[p.Journey]:
    """批量创建多个 Journey，按顺序返回结果列表。

    Args:
        agent:   已创建的 Parlant :class:`~parlant.sdk.Agent` 对象。
        configs: :class:`~journey.models.JourneyConfig` 列表（按需排序）。

    Returns:
        与 configs 顺序一致的 Parlant :class:`~parlant.sdk.Journey` 列表。

    示例::

        journeys = await register_journeys(agent, [fraud_config, loan_config])
    """
    logger.debug("[register_journeys] 开始批量注册  count=%d", len(configs))
    journeys = [await create_journey(agent, cfg) for cfg in configs]
    logger.debug("[register_journeys] 批量注册完成  count=%d", len(journeys))
    return journeys

