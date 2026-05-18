"""Tests for parlant_journey.factory."""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Ensure the parlant stub exists before importing our modules
def _ensure_parlant_stub() -> None:
    if "parlant.sdk" not in sys.modules:
        parlant = types.ModuleType("parlant")
        sdk = types.ModuleType("parlant.sdk")

        class CompositionMode:
            STRICT = "STRICT"
            FLUID = "FLUID"

        sdk.CompositionMode = CompositionMode
        sdk.Agent = object
        sdk.Journey = object
        parlant.sdk = sdk
        sys.modules.setdefault("parlant", parlant)
        sys.modules.setdefault("parlant.sdk", sdk)


_ensure_parlant_stub()

from parlant_journey.factory import create_journey, register_journeys  # noqa: E402
from parlant_journey.models import JourneyConfig, StateConfig  # noqa: E402


def _simple_config(title: str = "Test") -> JourneyConfig:
    return JourneyConfig(
        title=title,
        triggers=["go"],
        description="A simple test journey.",
        states=[StateConfig(name="A", chat_state="Do A")],
        transitions=[],
    )


def _mock_builder(journey_mock: MagicMock):
    """Return a mock class that replaces JourneyBuilder."""
    builder_instance = MagicMock()
    builder_instance.build = AsyncMock(return_value=journey_mock)
    builder_class = MagicMock(return_value=builder_instance)
    return builder_class, builder_instance


class TestCreateJourney:
    @pytest.mark.asyncio
    async def test_instantiates_builder_with_agent_and_config(self):
        journey_mock = MagicMock(name="journey")
        agent_mock = MagicMock(name="agent")
        config = _simple_config()

        builder_class, builder_instance = _mock_builder(journey_mock)

        with patch("parlant_journey.factory.JourneyBuilder", builder_class):
            result = await create_journey(agent_mock, config)

        builder_class.assert_called_once_with(agent_mock, config)
        builder_instance.build.assert_awaited_once()
        assert result is journey_mock

    @pytest.mark.asyncio
    async def test_returns_journey_from_builder(self):
        journey_mock = MagicMock(name="journey")
        agent_mock = MagicMock()
        config = _simple_config()

        builder_class, _ = _mock_builder(journey_mock)

        with patch("parlant_journey.factory.JourneyBuilder", builder_class):
            result = await create_journey(agent_mock, config)

        assert result is journey_mock


class TestRegisterJourneys:
    @pytest.mark.asyncio
    async def test_creates_journey_for_each_config(self):
        agent_mock = MagicMock(name="agent")
        j1, j2, j3 = MagicMock(), MagicMock(), MagicMock()
        configs = [_simple_config("C1"), _simple_config("C2"), _simple_config("C3")]

        call_count = 0
        journeys = [j1, j2, j3]

        async def fake_create(agent, config):
            nonlocal call_count
            j = journeys[call_count]
            call_count += 1
            return j

        with patch("parlant_journey.factory.create_journey", side_effect=fake_create):
            result = await register_journeys(agent_mock, configs)

        assert result == [j1, j2, j3]
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_empty_configs_returns_empty_list(self):
        agent_mock = MagicMock()
        with patch("parlant_journey.factory.create_journey") as mock_create:
            result = await register_journeys(agent_mock, [])

        mock_create.assert_not_awaited()
        assert result == []

    @pytest.mark.asyncio
    async def test_preserves_order(self):
        agent_mock = MagicMock()
        titles_seen: list[str] = []
        configs = [_simple_config(f"J{i}") for i in range(5)]

        async def fake_create(agent, config):
            titles_seen.append(config.title)
            return MagicMock()

        with patch("parlant_journey.factory.create_journey", side_effect=fake_create):
            await register_journeys(agent_mock, configs)

        assert titles_seen == ["J0", "J1", "J2", "J3", "J4"]
