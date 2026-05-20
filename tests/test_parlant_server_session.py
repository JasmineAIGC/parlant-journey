from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from tools.parlant_server.session import run_conversation


class _FakeSessionModuleInstance:
    def __init__(self, events):
        self._events = events
        self.wait_calls = []

    async def create(self, **kwargs):
        return SimpleNamespace(id="s1")

    async def create_customer_message(self, **kwargs):
        return SimpleNamespace(offset=1)

    async def wait_for_more_events(self, **kwargs):
        self.wait_calls.append(kwargs)
        return None

    async def find_events(self, **kwargs):
        return self._events


def _install_parlant_core_stubs(monkeypatch: pytest.MonkeyPatch, events):
    class SessionModule:
        pass

    class Moderation:
        NONE = "none"

    class EventKind:
        MESSAGE = "message"

    class EventSource:
        CUSTOMER = "customer"
        AI_AGENT = "ai_agent"

    class Timeout(float):
        pass

    class CustomerStore:
        GUEST_ID = "guest"

    fake_mod_sessions_app = types.ModuleType("parlant.core.app_modules.sessions")
    fake_mod_sessions_app.SessionModule = SessionModule
    fake_mod_sessions_app.Moderation = Moderation

    fake_mod_sessions = types.ModuleType("parlant.core.sessions")
    fake_mod_sessions.EventKind = EventKind
    fake_mod_sessions.EventSource = EventSource
    fake_mod_sessions.Timeout = Timeout

    fake_mod_customers = types.ModuleType("parlant.core.customers")
    fake_mod_customers.CustomerStore = CustomerStore

    monkeypatch.setitem(sys.modules, "parlant", types.ModuleType("parlant"))
    monkeypatch.setitem(sys.modules, "parlant.core", types.ModuleType("parlant.core"))
    monkeypatch.setitem(sys.modules, "parlant.core.app_modules", types.ModuleType("parlant.core.app_modules"))
    monkeypatch.setitem(sys.modules, "parlant.core.app_modules.sessions", fake_mod_sessions_app)
    monkeypatch.setitem(sys.modules, "parlant.core.sessions", fake_mod_sessions)
    monkeypatch.setitem(sys.modules, "parlant.core.customers", fake_mod_customers)

    fake_instance = _FakeSessionModuleInstance(events=events)
    fake_ctx = SimpleNamespace(
        server=SimpleNamespace(_container={SessionModule: fake_instance})
    )
    fake_agent = SimpleNamespace(id="agent-1")
    return fake_ctx, fake_agent, fake_instance


@pytest.mark.asyncio
async def test_run_conversation_reads_agent_message(monkeypatch: pytest.MonkeyPatch):
    events = [SimpleNamespace(data={"message": "您好，我来帮您处理"})]
    ctx, agent, fake_session_module = _install_parlant_core_stubs(monkeypatch, events)

    replies = []
    async for _user, reply in run_conversation(ctx, agent, ["你好"], timeout=3.0):
        replies.append(reply)

    assert replies == ["您好，我来帮您处理"]
    assert fake_session_module.wait_calls
    wait_kwargs = fake_session_module.wait_calls[0]
    assert wait_kwargs["session_id"] == "s1"
    assert wait_kwargs["min_offset"] == 2
    assert wait_kwargs["kinds"] is None
    assert wait_kwargs["source"] is None
    assert float(wait_kwargs["timeout"]) == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_run_conversation_returns_empty_reply_when_no_agent_message(
    monkeypatch: pytest.MonkeyPatch,
):
    ctx, agent, fake_session_module = _install_parlant_core_stubs(monkeypatch, events=[])

    replies = []
    async for _user, reply in run_conversation(ctx, agent, ["你好"], timeout=3.0):
        replies.append(reply)

    assert replies == [""]
    assert fake_session_module.wait_calls
    wait_kwargs = fake_session_module.wait_calls[0]
    assert wait_kwargs["session_id"] == "s1"
    assert wait_kwargs["min_offset"] == 2
    assert wait_kwargs["kinds"] is None
    assert wait_kwargs["source"] is None
    assert float(wait_kwargs["timeout"]) == pytest.approx(3.0)
