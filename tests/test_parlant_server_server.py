from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.parlant_server.server import ParlantServerConfig, ParlantServerContext


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


@pytest.mark.asyncio
async def test_context_preserves_server_kwargs_port_over_explicit_port(monkeypatch: pytest.MonkeyPatch):
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

    assert captured['port'] == 9456


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
        server_factory=fake_server_factory,
    )

    async with ParlantServerContext(config=config):
        pass

    assert captured['nlp_service'] == 'custom-nlp'
    assert captured['port'] == 9567
