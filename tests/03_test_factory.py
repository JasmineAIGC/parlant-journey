"""
03_test_factory.py — journey.factory 单元测试

parlant.sdk 桩由 conftest.py 统一注入，无需在此重复定义。
通过 patch JourneyBuilder 验证 factory 函数的委托行为。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from journey.factory import create_journey, register_journeys
from journey.models import JourneyConfig, StateConfig


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _简单配置(title: str = "测试") -> JourneyConfig:
    """返回最小可用的测试配置。"""
    return JourneyConfig(
        title=title,
        triggers=["go"],
        description="工厂测试用。",
        states=[StateConfig(name="A", chat_state="执行 A")],
        transitions=[],
    )


def _mock构建器(journey_mock: MagicMock):
    """返回替换 JourneyBuilder 的 mock 类及其实例。"""
    builder_instance = MagicMock()
    builder_instance.build = AsyncMock(return_value=journey_mock)
    builder_class = MagicMock(return_value=builder_instance)
    return builder_class, builder_instance


# ---------------------------------------------------------------------------
# create_journey
# ---------------------------------------------------------------------------

class TestCreateJourney:
    @pytest.mark.asyncio
    async def test_用正确参数实例化构建器(self, mock_agent):
        journey_mock = MagicMock(name="journey")
        config = _简单配置()
        builder_class, builder_instance = _mock构建器(journey_mock)

        with patch("journey.factory.JourneyBuilder", builder_class):
            result = await create_journey(mock_agent, config)

        # 验证构建器以 (agent, config) 实例化
        builder_class.assert_called_once_with(mock_agent, config)
        # 验证调用了 build()
        builder_instance.build.assert_awaited_once()
        # 返回值应为构建器产出的 journey
        assert result is journey_mock

    @pytest.mark.asyncio
    async def test_返回构建器产出的Journey(self, mock_agent):
        journey_mock = MagicMock(name="journey")
        builder_class, _ = _mock构建器(journey_mock)

        with patch("journey.factory.JourneyBuilder", builder_class):
            result = await create_journey(mock_agent, _简单配置())

        assert result is journey_mock


# ---------------------------------------------------------------------------
# register_journeys
# ---------------------------------------------------------------------------

class TestRegisterJourneys:
    @pytest.mark.asyncio
    async def test_为每个配置创建Journey(self, mock_agent):
        j1, j2, j3 = MagicMock(), MagicMock(), MagicMock()
        journeys = [j1, j2, j3]
        configs = [_简单配置(f"流程{i}") for i in range(3)]

        call_idx = 0

        async def fake_create(agent, config):
            nonlocal call_idx
            j = journeys[call_idx]
            call_idx += 1
            return j

        with patch("journey.factory.create_journey", side_effect=fake_create):
            result = await register_journeys(mock_agent, configs)

        assert result == [j1, j2, j3]
        assert call_idx == 3

    @pytest.mark.asyncio
    async def test_空列表返回空结果(self, mock_agent):
        with patch("journey.factory.create_journey") as mock_create:
            result = await register_journeys(mock_agent, [])

        mock_create.assert_not_awaited()
        assert result == []

    @pytest.mark.asyncio
    async def test_保持输入顺序(self, mock_agent):
        """验证返回的 Journey 列表顺序与配置列表一致。"""
        titles_seen: list[str] = []
        configs = [_简单配置(f"J{i}") for i in range(5)]

        async def fake_create(agent, config):
            titles_seen.append(config.title)
            return MagicMock()

        with patch("journey.factory.create_journey", side_effect=fake_create):
            await register_journeys(mock_agent, configs)

        assert titles_seen == [f"J{i}" for i in range(5)]

