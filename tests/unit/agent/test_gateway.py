"""Unit tests for InterfaceGateway — central multi-channel coordination."""

from __future__ import annotations

import asyncio
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from definable.agent.interface.base import BaseInterface
from definable.agent.interface.config import InterfaceConfig
from definable.agent.interface.gateway import (
  InterfaceErrorEvent,
  InterfaceGateway,
  InterfaceRestartedEvent,
  InterfaceStartedEvent,
  InterfaceStatus,
  InterfaceStoppedEvent,
  _GatewayHookBridge,
)
from definable.agent.interface.message import InterfaceMessage, InterfaceResponse
from definable.agent.interface.session import InterfaceSession
from definable.agent.run.agent import RunEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubInterface(BaseInterface):
  """Minimal stub satisfying the BaseInterface contract for gateway tests."""

  def __init__(self, platform: str = "test") -> None:
    super().__init__(config=InterfaceConfig(platform=platform))
    self._serve_side_effect: Any = None
    self._serve_called = asyncio.Event()
    self.sent_responses: List[InterfaceResponse] = []

  def bind(self, agent: Any) -> "_StubInterface":  # type: ignore[override]
    self.agent = agent
    return self  # type: ignore[return-value]

  async def _start_receiver(self) -> None:
    """No-op for stub."""

  async def _stop_receiver(self) -> None:
    """No-op for stub."""

  async def _convert_inbound(self, raw_message: Any) -> Optional[InterfaceMessage]:
    """No-op for stub."""
    return None

  async def _send_response(self, original_msg: Any, response: Any, raw_message: Any) -> None:
    """Record sent responses for assertion."""
    self.sent_responses.append(response)

  async def handle_platform_message(self, raw_message: Any) -> None:
    """Stub — does nothing; exists so the bridge wrapper can install."""

  async def serve_forever(self) -> None:
    self._serve_called.set()
    if self._serve_side_effect is not None:
      effect = self._serve_side_effect
      self._serve_side_effect = None  # only crash once
      if isinstance(effect, BaseException):
        raise effect
      if callable(effect):
        await effect()
    else:
      # Simulate a clean stop after a short delay
      await asyncio.sleep(0.05)


class _SpyHook:
  """Records all hook calls for assertion."""

  def __init__(self, *, veto: bool = False) -> None:
    self.received: List[InterfaceMessage] = []
    self.before_respond_calls: List[InterfaceMessage] = []
    self.after_respond_calls: List[Any] = []
    self.errors: List[Exception] = []
    self._veto = veto

  async def on_message_received(self, message: InterfaceMessage) -> Optional[bool]:
    self.received.append(message)
    if self._veto:
      return False
    return None

  async def on_before_respond(self, message: InterfaceMessage, session: InterfaceSession) -> Optional[InterfaceMessage]:
    self.before_respond_calls.append(message)
    return None

  async def on_after_respond(
    self,
    message: InterfaceMessage,
    response: InterfaceResponse,
    session: InterfaceSession,
  ) -> Optional[InterfaceResponse]:
    self.after_respond_calls.append(response)
    return None

  async def on_error(self, error: Exception, message: Optional[InterfaceMessage]) -> None:
    self.errors.append(error)


def _make_agent() -> MagicMock:
  agent = MagicMock()
  agent.agent_name = "test-agent"
  agent._event_bus = MagicMock()
  agent._event_bus.emit = AsyncMock()
  return agent


def _make_message(platform: str = "test", user_id: str = "u1", text: str = "hello") -> InterfaceMessage:
  return InterfaceMessage(
    platform=platform,
    platform_user_id=user_id,
    platform_chat_id="c1",
    platform_message_id="m1",
    text=text,
    metadata={},
  )


# ===========================================================================
# 1. Construction
# ===========================================================================


class TestGatewayConstruction:
  def test_default_construction(self) -> None:
    agent = _make_agent()
    gw = InterfaceGateway(agent)
    assert gw.agent is agent
    assert gw.interfaces == []
    assert gw.is_healthy  # no interfaces = vacuously healthy
    assert gw._shared_session_manager is None

  def test_shared_sessions_creates_manager(self) -> None:
    gw = InterfaceGateway(_make_agent(), shared_sessions=True, session_ttl_seconds=600)
    assert gw._shared_session_manager is not None
    assert gw._shared_session_manager._ttl == 600

  def test_identity_resolver_stored(self) -> None:
    resolver = MagicMock()
    gw = InterfaceGateway(_make_agent(), identity_resolver=resolver)
    assert gw._identity_resolver is resolver

  def test_identity_linking_creates_hook(self) -> None:
    resolver = MagicMock()
    gw = InterfaceGateway(
      _make_agent(),
      identity_resolver=resolver,
      enable_identity_linking=True,
      link_command="/connect",
      link_code_ttl=120,
    )
    assert gw._link_hook is not None
    assert gw._link_hook._command == "/connect"
    assert gw._link_hook._code_ttl == 120
    # Link hook should be first in hooks list
    assert gw._hooks[0] is gw._link_hook  # type: ignore[comparison-overlap]


# ===========================================================================
# 2. Interface add / remove
# ===========================================================================


class TestInterfaceManagement:
  def test_add_binds_and_tracks(self) -> None:
    agent = _make_agent()
    gw = InterfaceGateway(agent)
    iface = _StubInterface("telegram")
    result = gw.add(iface)
    assert result is gw  # method chaining
    assert iface in gw.interfaces
    assert iface.agent is agent
    assert gw.status(iface) == InterfaceStatus.pending

  def test_add_injects_hook_bridge(self) -> None:
    gw = InterfaceGateway(_make_agent())
    iface = _StubInterface()
    gw.add(iface)
    assert len(iface._hooks) == 1
    assert isinstance(iface._hooks[0], _GatewayHookBridge)

  def test_add_shares_session_manager(self) -> None:
    gw = InterfaceGateway(_make_agent(), shared_sessions=True)
    iface = _StubInterface()
    gw.add(iface)
    assert iface.session_manager is gw._shared_session_manager

  def test_add_propagates_identity_resolver(self) -> None:
    resolver = MagicMock()
    gw = InterfaceGateway(_make_agent(), identity_resolver=resolver)
    iface = _StubInterface()
    gw.add(iface)
    assert iface._identity_resolver is resolver

  def test_add_does_not_overwrite_existing_resolver(self) -> None:
    gw_resolver = MagicMock()
    iface_resolver = MagicMock()
    gw = InterfaceGateway(_make_agent(), identity_resolver=gw_resolver)
    iface = _StubInterface()
    iface._identity_resolver = iface_resolver
    gw.add(iface)
    assert iface._identity_resolver is iface_resolver

  def test_remove_cleans_up(self) -> None:
    gw = InterfaceGateway(_make_agent())
    iface = _StubInterface()
    gw.add(iface)
    assert gw.remove(iface)
    assert iface not in gw.interfaces
    assert len(iface._hooks) == 0  # bridge removed
    assert not gw.remove(iface)  # second remove returns False


# ===========================================================================
# 3. Hook injection + behavior
# ===========================================================================


class TestHookInjection:
  @pytest.mark.asyncio
  async def test_gateway_hooks_run_via_bridge(self) -> None:
    gw = InterfaceGateway(_make_agent())
    spy = _SpyHook()
    gw.add_hook(spy)

    iface = _StubInterface()
    gw.add(iface)
    bridge = iface._hooks[0]

    msg = _make_message()
    await bridge.on_message_received(msg)
    assert len(spy.received) == 1

  @pytest.mark.asyncio
  async def test_gateway_hook_veto_stops_pipeline(self) -> None:
    gw = InterfaceGateway(_make_agent())
    veto_hook = _SpyHook(veto=True)
    gw.add_hook(veto_hook)

    iface = _StubInterface()
    gw.add(iface)
    bridge = iface._hooks[0]

    msg = _make_message()
    result = await bridge.on_message_received(msg)
    assert result is False

  @pytest.mark.asyncio
  async def test_add_hook_after_registration_still_works(self) -> None:
    """Gateway hooks added after interface registration work because the
    bridge holds a reference to the gateway, not a snapshot."""
    gw = InterfaceGateway(_make_agent())
    iface = _StubInterface()
    gw.add(iface)

    spy = _SpyHook()
    gw.add_hook(spy)

    bridge = iface._hooks[0]
    msg = _make_message()
    await bridge.on_message_received(msg)
    assert len(spy.received) == 1

  def test_remove_hook(self) -> None:
    gw = InterfaceGateway(_make_agent())
    spy = _SpyHook()
    gw.add_hook(spy)
    assert gw.remove_hook(spy)
    assert spy not in gw._hooks
    assert not gw.remove_hook(spy)  # already removed

  @pytest.mark.asyncio
  async def test_bridge_on_before_respond(self) -> None:
    gw = InterfaceGateway(_make_agent())
    spy = _SpyHook()
    gw.add_hook(spy)

    iface = _StubInterface()
    gw.add(iface)
    bridge = iface._hooks[0]

    msg = _make_message()
    session = InterfaceSession()
    result = await bridge.on_before_respond(msg, session)
    assert len(spy.before_respond_calls) == 1
    assert result is msg

  @pytest.mark.asyncio
  async def test_bridge_on_error(self) -> None:
    gw = InterfaceGateway(_make_agent())
    spy = _SpyHook()
    gw.add_hook(spy)

    iface = _StubInterface()
    gw.add(iface)
    bridge = iface._hooks[0]

    err = RuntimeError("boom")
    await bridge.on_error(err, None)
    assert len(spy.errors) == 1
    assert spy.errors[0] is err


# ===========================================================================
# 4. Status tracking
# ===========================================================================


class TestStatusTracking:
  def test_initial_status_is_pending(self) -> None:
    gw = InterfaceGateway(_make_agent())
    iface = _StubInterface("telegram")
    gw.add(iface)
    assert gw.status(iface) == InterfaceStatus.pending

  def test_statuses_property(self) -> None:
    gw = InterfaceGateway(_make_agent())
    gw.add(_StubInterface("telegram"))
    gw.add(_StubInterface("discord"))
    statuses = gw.statuses
    assert "telegram" in statuses
    assert "discord" in statuses
    assert all(s == InterfaceStatus.pending for s in statuses.values())

  def test_status_raises_for_unknown_interface(self) -> None:
    gw = InterfaceGateway(_make_agent())
    iface = _StubInterface()
    with pytest.raises(ValueError, match="not registered"):
      gw.status(iface)

  def test_is_healthy_with_no_interfaces(self) -> None:
    gw = InterfaceGateway(_make_agent())
    assert gw.is_healthy


# ===========================================================================
# 5. Shared sessions
# ===========================================================================


class TestSharedSessions:
  def test_shared_manager_propagated_to_all(self) -> None:
    gw = InterfaceGateway(_make_agent(), shared_sessions=True)
    iface1 = _StubInterface("tg")
    iface2 = _StubInterface("dc")
    gw.add(iface1)
    gw.add(iface2)
    assert iface1.session_manager is iface2.session_manager
    assert iface1.session_manager is gw._shared_session_manager

  def test_no_shared_sessions_by_default(self) -> None:
    gw = InterfaceGateway(_make_agent())
    iface1 = _StubInterface("tg")
    iface2 = _StubInterface("dc")
    gw.add(iface1)
    gw.add(iface2)
    assert iface1.session_manager is not iface2.session_manager

  def test_shared_session_ttl_respected(self) -> None:
    gw = InterfaceGateway(_make_agent(), shared_sessions=True, session_ttl_seconds=42)
    assert gw._shared_session_manager is not None
    assert gw._shared_session_manager._ttl == 42


# ===========================================================================
# 6. Gateway events
# ===========================================================================


class TestGatewayEvents:
  def test_event_enum_values_exist(self) -> None:
    assert RunEvent.interface_started.value == "InterfaceStarted"
    assert RunEvent.interface_stopped.value == "InterfaceStopped"
    assert RunEvent.interface_restarted.value == "InterfaceRestarted"
    assert RunEvent.interface_error.value == "InterfaceError"

  def test_started_event_fields(self) -> None:
    ev = InterfaceStartedEvent(platform="telegram", interface_class="TelegramInterface")
    assert ev.platform == "telegram"
    assert ev.interface_class == "TelegramInterface"
    assert ev.event == "InterfaceStarted"

  def test_error_event_fields(self) -> None:
    ev = InterfaceErrorEvent(
      platform="discord",
      interface_class="DiscordInterface",
      error_message="connection refused",
    )
    assert ev.error_message == "connection refused"

  def test_restarted_event_fields(self) -> None:
    ev = InterfaceRestartedEvent(
      platform="test",
      interface_class="TestInterface",
      restart_count=3,
      backoff_seconds=8.0,
    )
    assert ev.restart_count == 3
    assert ev.backoff_seconds == 8.0


# ===========================================================================
# 7. Supervision / aserve
# ===========================================================================


class TestSupervision:
  @pytest.mark.asyncio
  async def test_aserve_raises_with_no_interfaces(self) -> None:
    gw = InterfaceGateway(_make_agent())
    with pytest.raises(ValueError, match="no interfaces"):
      await gw.aserve()

  @pytest.mark.asyncio
  async def test_aserve_clean_stop(self) -> None:
    agent = _make_agent()
    gw = InterfaceGateway(agent)
    iface = _StubInterface("test")
    gw.add(iface)
    await gw.aserve(name="test-gw")
    assert gw.status(iface) == InterfaceStatus.stopped

  @pytest.mark.asyncio
  async def test_aserve_emits_started_and_stopped_events(self) -> None:
    agent = _make_agent()
    gw = InterfaceGateway(agent)
    iface = _StubInterface("test")
    gw.add(iface)
    await gw.aserve()

    emit_calls = agent._event_bus.emit.call_args_list
    event_types = [type(call.args[0]) for call in emit_calls]
    assert InterfaceStartedEvent in event_types
    assert InterfaceStoppedEvent in event_types

  @pytest.mark.asyncio
  async def test_aserve_restarts_on_crash(self) -> None:
    agent = _make_agent()
    gw = InterfaceGateway(agent)
    iface = _StubInterface("crasher")
    iface._serve_side_effect = RuntimeError("boom")
    gw.add(iface)

    await gw.aserve()

    # Should have emitted error + restarted events
    emit_calls = agent._event_bus.emit.call_args_list
    event_types = [type(call.args[0]) for call in emit_calls]
    assert InterfaceErrorEvent in event_types
    assert InterfaceRestartedEvent in event_types
    assert gw._restart_counts[id(iface)] == 1


# ===========================================================================
# 8. Agent integration
# ===========================================================================


class TestAgentIntegration:
  def test_gateway_property_default_none(self) -> None:
    from definable.agent.agent import Agent
    from definable.model.openai.chat import OpenAIChat

    with patch.object(OpenAIChat, "__init__", lambda self, **kw: None):
      model = OpenAIChat.__new__(OpenAIChat)
      model.id = "gpt-4o-mini"
      model.name = "gpt-4o-mini"
      model.provider = "OpenAI"
      model.metrics = {}  # type: ignore[attr-defined]
      model.provider_request_headers = None  # type: ignore[attr-defined]
      agent = Agent(model=model)

    assert agent.gateway is None


# ===========================================================================
# 9. Identity linking — /link flow
# ===========================================================================


class TestIdentityLinking:
  @pytest.mark.asyncio
  async def test_link_generates_code(self) -> None:
    resolver = AsyncMock()
    gw = InterfaceGateway(
      _make_agent(),
      identity_resolver=resolver,
      enable_identity_linking=True,
    )
    hook = gw._link_hook
    assert hook is not None

    msg = _make_message(platform="telegram", text="/link")
    result = await hook.on_message_received(msg)
    assert result is False  # vetoed — don't pass to agent
    assert "_link_reply" in msg.metadata
    assert "Enter this code" in msg.metadata["_link_reply"]
    # A code should be pending
    assert len(hook._pending) == 1

  @pytest.mark.asyncio
  async def test_link_validates_code(self) -> None:
    resolver = AsyncMock()
    resolver.resolve = AsyncMock(return_value=None)
    gw = InterfaceGateway(
      _make_agent(),
      identity_resolver=resolver,
      enable_identity_linking=True,
    )
    hook = gw._link_hook
    assert hook is not None

    # Step 1: generate code from telegram
    msg1 = _make_message(platform="telegram", user_id="tg_user", text="/link")
    await hook.on_message_received(msg1)
    code = list(hook._pending.keys())[0]

    # Step 2: validate code from discord
    msg2 = _make_message(platform="discord", user_id="dc_user", text=f"/link {code}")
    result = await hook.on_message_received(msg2)
    assert result is False
    assert "Linked" in msg2.metadata["_link_reply"]

    # resolver.link should have been called for both platforms
    assert resolver.link.call_count == 2

  @pytest.mark.asyncio
  async def test_expired_code_rejected(self) -> None:
    resolver = AsyncMock()
    gw = InterfaceGateway(
      _make_agent(),
      identity_resolver=resolver,
      enable_identity_linking=True,
      link_code_ttl=0,  # expire immediately
    )
    hook = gw._link_hook
    assert hook is not None

    msg1 = _make_message(platform="telegram", text="/link")
    await hook.on_message_received(msg1)
    code = list(hook._pending.keys())[0]

    # Wait a tiny bit so TTL expires
    await asyncio.sleep(0.01)

    msg2 = _make_message(platform="discord", text=f"/link {code}")
    await hook.on_message_received(msg2)
    assert "Invalid or expired" in msg2.metadata["_link_reply"]

  @pytest.mark.asyncio
  async def test_link_reuses_existing_canonical_id(self) -> None:
    resolver = AsyncMock()
    resolver.resolve = AsyncMock(side_effect=lambda p, u: "canonical_alice" if p == "telegram" else None)
    gw = InterfaceGateway(
      _make_agent(),
      identity_resolver=resolver,
      enable_identity_linking=True,
    )
    hook = gw._link_hook
    assert hook is not None

    msg1 = _make_message(platform="telegram", user_id="tg_alice", text="/link")
    await hook.on_message_received(msg1)
    code = list(hook._pending.keys())[0]

    msg2 = _make_message(platform="discord", user_id="dc_alice", text=f"/link {code}")
    await hook.on_message_received(msg2)

    # Should use "canonical_alice" from telegram, not generate a new UUID
    link_calls = resolver.link.call_args_list
    assert link_calls[0].args[2] == "canonical_alice"
    assert link_calls[1].args[2] == "canonical_alice"


# ===========================================================================
# 9b. Identity linking — reply delivery
# ===========================================================================


class TestIdentityLinkingReply:
  @pytest.mark.asyncio
  async def test_link_reply_sent_to_user(self) -> None:
    """When /link vetoes the message, the reply is sent back to the user."""
    resolver = AsyncMock()
    gw = InterfaceGateway(
      _make_agent(),
      identity_resolver=resolver,
      enable_identity_linking=True,
    )
    iface = _StubInterface("telegram")
    gw.add(iface)
    bridge = iface._hooks[0]

    msg = _make_message(platform="telegram", text="/link")
    result = await bridge.on_message_received(msg)
    assert result is False
    assert len(iface.sent_responses) == 1
    assert iface.sent_responses[0].content is not None
    assert "Enter this code" in iface.sent_responses[0].content

  @pytest.mark.asyncio
  async def test_link_validate_reply_sent(self) -> None:
    """When /link CODE validates, the success reply is sent."""
    resolver = AsyncMock()
    resolver.resolve = AsyncMock(return_value=None)
    gw = InterfaceGateway(
      _make_agent(),
      identity_resolver=resolver,
      enable_identity_linking=True,
    )
    iface = _StubInterface("telegram")
    gw.add(iface)
    bridge = iface._hooks[0]
    hook = gw._link_hook
    assert hook is not None

    # Generate code
    msg1 = _make_message(platform="telegram", user_id="tg1", text="/link")
    await bridge.on_message_received(msg1)
    code = list(hook._pending.keys())[0]

    # Validate code from another "platform" (reuse same interface for simplicity)
    msg2 = _make_message(platform="discord", user_id="dc1", text=f"/link {code}")
    await bridge.on_message_received(msg2)

    assert len(iface.sent_responses) == 2
    assert iface.sent_responses[1].content is not None
    assert "Linked" in iface.sent_responses[1].content

  @pytest.mark.asyncio
  async def test_non_link_message_not_intercepted(self) -> None:
    """Normal messages pass through without triggering the link hook."""
    resolver = AsyncMock()
    gw = InterfaceGateway(
      _make_agent(),
      identity_resolver=resolver,
      enable_identity_linking=True,
    )
    iface = _StubInterface("telegram")
    gw.add(iface)
    bridge = iface._hooks[0]

    msg = _make_message(platform="telegram", text="hello world")
    result = await bridge.on_message_received(msg)
    assert result is None  # pass through
    assert len(iface.sent_responses) == 0


# ===========================================================================
# 10. Identity resolver propagation
# ===========================================================================


class TestIdentityResolverPropagation:
  def test_resolver_set_on_all_interfaces(self) -> None:
    resolver = MagicMock()
    gw = InterfaceGateway(_make_agent(), identity_resolver=resolver)
    iface1 = _StubInterface("tg")
    iface2 = _StubInterface("dc")
    gw.add(iface1)
    gw.add(iface2)
    assert iface1._identity_resolver is resolver
    assert iface2._identity_resolver is resolver

  def test_resolver_not_set_when_not_configured(self) -> None:
    gw = InterfaceGateway(_make_agent())
    iface = _StubInterface()
    gw.add(iface)
    assert iface._identity_resolver is None


# ===========================================================================
# 11. Exports
# ===========================================================================


class TestExports:
  def test_lazy_imports_from_interface_package(self) -> None:
    from definable.agent.interface import (
      InterfaceErrorEvent,
      InterfaceGateway,
      InterfaceRestartedEvent,
      InterfaceStartedEvent,
      InterfaceStatus,
      InterfaceStoppedEvent,
    )

    assert InterfaceGateway is not None
    assert InterfaceStatus is not None
    assert InterfaceStartedEvent is not None
    assert InterfaceStoppedEvent is not None
    assert InterfaceRestartedEvent is not None
    assert InterfaceErrorEvent is not None

  def test_run_event_registry(self) -> None:
    """Interface events should be resolvable from the registry."""
    from definable.agent.run.agent import _ensure_interface_events_registered, RUN_EVENT_TYPE_REGISTRY

    _ensure_interface_events_registered()
    assert "InterfaceStarted" in RUN_EVENT_TYPE_REGISTRY
    assert "InterfaceStopped" in RUN_EVENT_TYPE_REGISTRY
    assert "InterfaceRestarted" in RUN_EVENT_TYPE_REGISTRY
    assert "InterfaceError" in RUN_EVENT_TYPE_REGISTRY
