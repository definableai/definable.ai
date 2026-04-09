"""Tests for Desktop Bridge event observability.

Validates:
- BridgeCallEvent and DesktopActionEvent dataclass structure
- BridgeClient emits events on _post, domain methods
- MacOS skill propagates on_event to BridgeClient
- DebugExporter handles desktop events
- Events are registered in RunEvent enum and events re-exports
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from definable.agent.interface.desktop.events import BridgeCallEvent, DesktopActionEvent


# ---------------------------------------------------------------------------
# Event dataclass basics
# ---------------------------------------------------------------------------


class TestBridgeCallEvent:
  def test_defaults(self):
    event = BridgeCallEvent()
    assert event.event == "BridgeCall"
    assert event.endpoint == ""
    assert event.method == "POST"
    assert event.status_code == 0
    assert event.duration_ms == 0.0
    assert event.error == ""
    assert event.timestamp > 0

  def test_custom_values(self):
    event = BridgeCallEvent(
      endpoint="/screen/capture",
      status_code=200,
      duration_ms=42.5,
    )
    assert event.endpoint == "/screen/capture"
    assert event.status_code == 200
    assert event.duration_ms == 42.5

  def test_to_dict(self):
    event = BridgeCallEvent(endpoint="/health", status_code=200, duration_ms=10.0)
    d = event.to_dict()
    assert d["event"] == "BridgeCall"
    assert d["endpoint"] == "/health"
    assert d["status_code"] == 200

  def test_error_field(self):
    event = BridgeCallEvent(endpoint="/bad", status_code=500, error="Internal error")
    assert event.error == "Internal error"


class TestDesktopActionEvent:
  def test_defaults(self):
    event = DesktopActionEvent()
    assert event.event == "DesktopAction"
    assert event.category == ""
    assert event.action == ""
    assert event.target == ""
    assert event.value == ""
    assert event.result == ""
    assert event.error == ""
    assert event.timestamp > 0

  def test_custom_values(self):
    event = DesktopActionEvent(
      category="input",
      action="click",
      target="(500,300)",
      value="left x1",
    )
    assert event.category == "input"
    assert event.action == "click"
    assert event.target == "(500,300)"
    assert event.value == "left x1"

  def test_to_dict(self):
    event = DesktopActionEvent(category="app", action="open_app", target="Safari")
    d = event.to_dict()
    assert d["event"] == "DesktopAction"
    assert d["category"] == "app"
    assert d["target"] == "Safari"

  def test_error_field(self):
    event = DesktopActionEvent(
      category="shell",
      action="run",
      error="command not found",
    )
    assert event.error == "command not found"


# ---------------------------------------------------------------------------
# BridgeClient event emission
# ---------------------------------------------------------------------------


class TestBridgeClientEvents:
  """Test that BridgeClient emits events through on_event callback."""

  @pytest.fixture
  def collected(self):
    """Collects emitted events."""
    return []

  @pytest.fixture
  def client(self, collected):
    from definable.agent.interface.desktop.bridge_client import BridgeClient

    def collector(event):
      collected.append(event)

    return BridgeClient(
      host="127.0.0.1",
      port=7777,
      token="test-token",
      on_event=collector,
    )

  @pytest.mark.asyncio
  async def test_post_emits_bridge_call_event(self, client, collected):
    """_post emits BridgeCallEvent on success."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"ok": True, "data": {"result": "ok"}}
    mock_response.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.post.return_value = mock_response
    client._client = mock_http

    await client._post("/health")

    bridge_events = [e for e in collected if isinstance(e, BridgeCallEvent)]
    assert len(bridge_events) == 1
    evt = bridge_events[0]
    assert evt.endpoint == "/health"
    assert evt.status_code == 200
    assert evt.error == ""
    assert evt.duration_ms >= 0

  @pytest.mark.asyncio
  async def test_post_emits_bridge_call_on_error(self, client, collected):
    """_post emits BridgeCallEvent with error on failure."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"ok": False, "error": "not found"}
    mock_response.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.post.return_value = mock_response
    client._client = mock_http

    with pytest.raises(RuntimeError, match="Bridge error"):
      await client._post("/bad/endpoint")

    bridge_events = [e for e in collected if isinstance(e, BridgeCallEvent)]
    assert len(bridge_events) == 1
    assert bridge_events[0].error == "not found"

  @pytest.mark.asyncio
  async def test_capture_screen_emits_action(self, client, collected):
    """capture_screen emits both BridgeCallEvent and DesktopActionEvent."""
    import base64

    fake_image = base64.b64encode(b"\xff\xd8\xff\xe0fake-jpeg").decode()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"ok": True, "data": {"image": fake_image}}
    mock_response.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.post.return_value = mock_response
    client._client = mock_http

    await client.capture_screen(display=0, max_width=512)

    bridge_events = [e for e in collected if isinstance(e, BridgeCallEvent)]
    action_events = [e for e in collected if isinstance(e, DesktopActionEvent)]

    assert len(bridge_events) == 1
    assert bridge_events[0].endpoint == "/screen/capture"

    assert len(action_events) == 1
    assert action_events[0].category == "screen"
    assert action_events[0].action == "screenshot"

  @pytest.mark.asyncio
  async def test_click_emits_action(self, client, collected):
    """click emits DesktopActionEvent with input category."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"ok": True, "data": {}}
    mock_response.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.post.return_value = mock_response
    client._client = mock_http

    await client.click(x=500, y=300)

    action_events = [e for e in collected if isinstance(e, DesktopActionEvent)]
    assert len(action_events) == 1
    assert action_events[0].category == "input"
    assert action_events[0].action == "click"
    assert "(500,300)" in action_events[0].target

  @pytest.mark.asyncio
  async def test_open_app_emits_action(self, client, collected):
    """open_app emits DesktopActionEvent with app category."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"ok": True, "data": {"pid": 1234}}
    mock_response.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.post.return_value = mock_response
    client._client = mock_http

    await client.open_app("Safari")

    action_events = [e for e in collected if isinstance(e, DesktopActionEvent)]
    assert len(action_events) == 1
    assert action_events[0].category == "app"
    assert action_events[0].action == "open_app"
    assert action_events[0].target == "Safari"
    assert "1234" in action_events[0].result

  @pytest.mark.asyncio
  async def test_run_shell_emits_action(self, client, collected):
    """run_shell emits DesktopActionEvent with shell category."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"ok": True, "data": {"stdout": "ok", "stderr": "", "exit_code": 0, "success": True}}
    mock_response.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.post.return_value = mock_response
    client._client = mock_http

    await client.run_shell(command=["ls", "-la"])

    action_events = [e for e in collected if isinstance(e, DesktopActionEvent)]
    assert len(action_events) == 1
    assert action_events[0].category == "shell"
    assert action_events[0].action == "run"
    assert "ls -la" in action_events[0].value

  @pytest.mark.asyncio
  async def test_no_events_without_callback(self):
    """BridgeClient without on_event does not emit."""
    from definable.agent.interface.desktop.bridge_client import BridgeClient

    client = BridgeClient(token="test")
    # _emit_action should be a no-op
    client._emit_action("test", "test")  # Should not raise
    client._emit_bridge_call("/test", 200, 10.0, "")  # Should not raise


# ---------------------------------------------------------------------------
# MacOS skill wiring
# ---------------------------------------------------------------------------


class TestMacOSSkillEventWiring:
  def test_on_event_propagated_to_client(self):
    """MacOS skill passes on_event to BridgeClient."""
    from definable.skill.builtin.macos import MacOS

    callback = MagicMock()
    skill = MacOS(on_event=callback)
    client = skill._get_client()
    assert client._on_event is callback

  def test_on_event_default_none(self):
    """MacOS skill defaults to no event callback."""
    from definable.skill.builtin.macos import MacOS

    skill = MacOS()
    client = skill._get_client()
    assert client._on_event is None


# ---------------------------------------------------------------------------
# Event registration
# ---------------------------------------------------------------------------


class TestEventRegistration:
  def test_run_event_enum_has_desktop_entries(self):
    """RunEvent enum includes bridge_call and desktop_action."""
    from definable.agent.run.agent import RunEvent

    assert hasattr(RunEvent, "bridge_call")
    assert hasattr(RunEvent, "desktop_action")
    assert RunEvent.bridge_call.value == "BridgeCall"
    assert RunEvent.desktop_action.value == "DesktopAction"

  def test_events_re_exported(self):
    """BridgeCallEvent and DesktopActionEvent are accessible from agent.events."""
    from definable.agent.events import BridgeCallEvent as BCE
    from definable.agent.events import DesktopActionEvent as DAE

    assert BCE is BridgeCallEvent
    assert DAE is DesktopActionEvent

  def test_events_in_all(self):
    """Events are in __all__ of agent.events."""
    from definable.agent import events

    assert "BridgeCallEvent" in events.__all__
    assert "DesktopActionEvent" in events.__all__

  def test_events_in_desktop_init(self):
    """Events are re-exported from desktop __init__."""
    from definable.agent.interface.desktop import BridgeCallEvent as BCE
    from definable.agent.interface.desktop import DesktopActionEvent as DAE

    assert BCE is BridgeCallEvent
    assert DAE is DesktopActionEvent


# ---------------------------------------------------------------------------
# DebugExporter integration
# ---------------------------------------------------------------------------


class TestDebugExporterDesktopEvents:
  def test_handles_desktop_action_event(self):
    """DebugExporter.export() handles DesktopActionEvent without error."""
    from definable.agent.tracing.debug import DebugExporter

    exporter = DebugExporter()
    event = DesktopActionEvent(
      category="input",
      action="click",
      target="(500,300)",
      value="left x1",
      result="ok",
    )
    # Should not raise — output goes to stderr via rich
    exporter.export(event)

  def test_handles_bridge_call_event(self):
    """DebugExporter.export() handles BridgeCallEvent without error."""
    from definable.agent.tracing.debug import DebugExporter

    exporter = DebugExporter()
    event = BridgeCallEvent(
      endpoint="/screen/capture",
      status_code=200,
      duration_ms=42.5,
    )
    exporter.export(event)

  def test_handles_desktop_action_with_error(self):
    """DebugExporter shows error styling for failed actions."""
    from definable.agent.tracing.debug import DebugExporter

    exporter = DebugExporter()
    event = DesktopActionEvent(
      category="shell",
      action="run",
      value="rm -rf /",
      error="permission denied",
    )
    exporter.export(event)

  def test_handles_bridge_call_with_error(self):
    """DebugExporter shows error styling for failed bridge calls."""
    from definable.agent.tracing.debug import DebugExporter

    exporter = DebugExporter()
    event = BridgeCallEvent(
      endpoint="/bad",
      status_code=500,
      duration_ms=100.0,
      error="Internal Server Error",
    )
    exporter.export(event)


# ---------------------------------------------------------------------------
# Async callback handling
# ---------------------------------------------------------------------------


class TestAsyncCallbackHandling:
  @pytest.mark.asyncio
  async def test_sync_callback_works(self):
    """Sync on_event callback is invoked correctly."""
    from definable.agent.interface.desktop.bridge_client import BridgeClient

    collected = []
    client = BridgeClient(token="test", on_event=lambda e: collected.append(e))
    client._emit_action("test", "test_action")
    assert len(collected) == 1
    assert isinstance(collected[0], DesktopActionEvent)

  @pytest.mark.asyncio
  async def test_async_callback_works(self):
    """Async on_event callback is invoked correctly."""
    from definable.agent.interface.desktop.bridge_client import BridgeClient

    collected = []

    async def async_collector(event):
      collected.append(event)

    client = BridgeClient(token="test", on_event=async_collector)
    client._emit_action("test", "test_action")
    # Give the ensure_future a chance to run
    await asyncio.sleep(0.01)
    assert len(collected) == 1

  @pytest.mark.asyncio
  async def test_callback_error_does_not_propagate(self):
    """Errors in on_event callback are swallowed."""
    from definable.agent.interface.desktop.bridge_client import BridgeClient

    def bad_callback(event):
      raise ValueError("boom")

    client = BridgeClient(token="test", on_event=bad_callback)
    # Should not raise
    client._emit_action("test", "test_action")
    client._emit_bridge_call("/test", 200, 10.0, "")
