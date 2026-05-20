from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.parlant_server.server import (
    DEFAULT_PARLANT_PORT,
    DEFAULT_PARLANT_TOOL_SERVICE_PORT,
    ParlantServerConfig,
    ParlantServerContext,
)


class _FakeServerContextManager:
    def __init__(self, captured_kwargs):
        self.captured_kwargs = captured_kwargs
        self.exit_args = None

    async def __aenter__(self):
        return SimpleNamespace(id='server-1')

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.exit_args = (exc_type, exc_val, exc_tb)
        return False


@pytest.mark.asyncio
async def test_context_passes_explicit_port_to_server_factory(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_server_factory(**kwargs):
        captured.update(kwargs)
        return _FakeServerContextManager(captured)

    monkeypatch.setattr(
        'tools.parlant_server.server._import_parlant_sdk',
        lambda: SimpleNamespace(NLPServices=SimpleNamespace(litellm='litellm')),
    )

    async with ParlantServerContext(
        port=9321,
        bridge_env=False,
        server_factory=fake_server_factory,
    ):
        pass

    assert captured['nlp_service'] == 'litellm'
    assert captured['port'] == 9321
    assert 'tool_service_port' not in captured


@pytest.mark.asyncio
async def test_context_explicit_port_overrides_server_kwargs_port(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_server_factory(**kwargs):
        captured.update(kwargs)
        return _FakeServerContextManager(captured)

    monkeypatch.setattr(
        'tools.parlant_server.server._import_parlant_sdk',
        lambda: SimpleNamespace(NLPServices=SimpleNamespace(litellm='litellm')),
    )

    async with ParlantServerContext(
        port=9321,
        bridge_env=False,
        server_factory=fake_server_factory,
        server_kwargs={'port': 9456},
    ):
        pass

    assert captured['port'] == 9321
    assert 'tool_service_port' not in captured


@pytest.mark.asyncio
async def test_context_explicit_tool_service_port_overrides_derived_value(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = {}

    def fake_server_factory(**kwargs):
        captured.update(kwargs)
        return _FakeServerContextManager(captured)

    monkeypatch.setattr(
        'tools.parlant_server.server._import_parlant_sdk',
        lambda: SimpleNamespace(NLPServices=SimpleNamespace(litellm='litellm')),
    )

    async with ParlantServerContext(
        port=9321,
        tool_service_port=9777,
        bridge_env=False,
        server_factory=fake_server_factory,
    ):
        pass

    assert captured['port'] == 9321
    assert captured['tool_service_port'] == 9777


@pytest.mark.asyncio
async def test_context_supports_port_in_config(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_server_factory(**kwargs):
        captured.update(kwargs)
        return _FakeServerContextManager(captured)

    monkeypatch.setattr(
        'tools.parlant_server.server._import_parlant_sdk',
        lambda: SimpleNamespace(NLPServices=SimpleNamespace(litellm='litellm')),
    )

    config = ParlantServerConfig(
        nlp_service='custom-nlp',
        port=9567,
        tool_service_port=9599,
        server_factory=fake_server_factory,
    )

    async with ParlantServerContext(config=config):
        pass

    assert captured['nlp_service'] == 'custom-nlp'
    assert captured['port'] == 9567
    assert captured['tool_service_port'] == 9599


@pytest.mark.asyncio
async def test_context_no_tool_service_port_when_only_port_set(monkeypatch: pytest.MonkeyPatch):
    """port 设置时不自动派生 tool_service_port；Parlant Server 自行使用 DEFAULT_PARLANT_TOOL_SERVICE_PORT (8818)。"""
    captured = {}

    def fake_server_factory(**kwargs):
        captured.update(kwargs)
        return _FakeServerContextManager(captured)

    monkeypatch.setattr(
        'tools.parlant_server.server._import_parlant_sdk',
        lambda: SimpleNamespace(NLPServices=SimpleNamespace(litellm='litellm')),
    )

    async with ParlantServerContext(
        port=9000,
        bridge_env=False,
        server_factory=fake_server_factory,
    ):
        pass

    assert captured['port'] == 9000
    # tool_service_port 未被传入；Parlant Server 自行使用其默认值 DEFAULT_PARLANT_TOOL_SERVICE_PORT
    assert 'tool_service_port' not in captured

