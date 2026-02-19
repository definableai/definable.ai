"""
Behavioral tests: MacOS skill tool dispatch with MockModel.

Migrated from: tests_e2e/behavioral/test_macos_agent.py

No bridge running, no API calls, no real macOS interaction.

A MagicMock BridgeClient is injected via ``skill._client`` so tools call
mock methods instead of HTTP. Uses MockModel side_effect to drive two-turn
conversations: turn 1 returns a tool call, turn 2 returns final content.

Covers:
  - screenshot tool -> client.capture_screen() called
  - read_screen tool -> client.ocr_screen() called
  - click tool -> client.click() called with correct coordinates
  - open_app tool -> client.open_app() called with correct app name
  - allowed_apps whitelist -> tool returns error string without calling client
  - blocked_apps -> tool returns error string without calling client
  - enable_input=False -> click tool absent, not dispatched
  - enable_applescript=False -> run_applescript tool absent
  - enable_file_write=False -> write_file tool absent
  - Bridge ConnectError -> tool returns friendly "Bridge is not running" message
  - tool execution returns string (never raises)
  - MockModel sees all skill tools on first call
  - send_notification tool dispatched correctly
  - read_file tool always present and dispatched correctly
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from definable.agent.testing import MockModel
from definable.model.metrics import Metrics
from definable.skill.builtin.macos import MacOS


# ---------------------------------------------------------------------------
# Shared: mock bridge client factory
# ---------------------------------------------------------------------------


def _make_mock_client() -> MagicMock:
  """Return a MagicMock BridgeClient where every method is an AsyncMock."""
  client = MagicMock()
  # Screen
  client.capture_screen = AsyncMock(return_value=b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
  client.ocr_screen = AsyncMock(return_value={"text": "Hello, world!", "elements": []})
  client.find_text_on_screen = AsyncMock(return_value=None)
  # Input
  client.click = AsyncMock(return_value=None)
  client.type_text = AsyncMock(return_value=None)
  client.press_key = AsyncMock(return_value=None)
  client.mouse_move = AsyncMock(return_value=None)
  client.scroll = AsyncMock(return_value=None)
  client.drag = AsyncMock(return_value=None)
  # Apps
  client.list_apps = AsyncMock(return_value=[])
  client.open_app = AsyncMock(return_value=12345)
  client.quit_app = AsyncMock(return_value=None)
  client.activate_app = AsyncMock(return_value=None)
  client.open_url = AsyncMock(return_value=None)
  client.open_file = AsyncMock(return_value=None)
  # Windows
  client.list_windows = AsyncMock(return_value=[])
  client.focus_window = AsyncMock(return_value=None)
  client.resize_window = AsyncMock(return_value=None)
  client.close_window = AsyncMock(return_value=None)
  # Accessibility
  client.get_focused_element = AsyncMock(return_value=None)
  client.get_ui_tree = AsyncMock(return_value={})
  client.find_ui_element = AsyncMock(return_value=None)
  client.click_ui_element = AsyncMock(return_value=None)
  client.set_ui_value = AsyncMock(return_value=None)
  # AppleScript
  client.run_applescript = AsyncMock(return_value={"output": "ok", "error": None})
  # Files
  client.list_files = AsyncMock(return_value=[])
  client.read_file = AsyncMock(return_value="file contents here")
  client.write_file = AsyncMock(return_value=None)
  client.move_file = AsyncMock(return_value=None)
  client.delete_file = AsyncMock(return_value=None)
  client.file_info = AsyncMock(return_value={})
  # Clipboard
  client.get_clipboard = AsyncMock(return_value="clipboard text")
  client.set_clipboard = AsyncMock(return_value=None)
  # System
  client.system_info = AsyncMock(return_value={"hostname": "testmac"})
  client.get_volume = AsyncMock(return_value=50)
  client.set_volume = AsyncMock(return_value=None)
  client.get_battery = AsyncMock(return_value={"level": 80, "charging": False})
  client.send_notification = AsyncMock(return_value=None)
  client.set_dark_mode = AsyncMock(return_value=None)
  client.lock_screen = AsyncMock(return_value=None)
  client.close = AsyncMock(return_value=None)
  client.health = AsyncMock(return_value={"status": "ok"})
  return client


def _make_response(content: str = "Done.", tool_calls: list | None = None) -> MagicMock:
  """Return a MagicMock model response."""
  r = MagicMock()
  r.response_usage = Metrics()
  r.reasoning_content = None
  r.citations = None
  r.images = None
  r.videos = None
  r.audios = None
  r.tool_executions = []
  r.content = content
  r.tool_calls = tool_calls or []
  return r


def _tool_call(name: str, arguments: str, call_id: str = "call_001") -> dict:
  return {
    "type": "function",
    "id": call_id,
    "function": {"name": name, "arguments": arguments},
  }


# ---------------------------------------------------------------------------
# Shared agent setup helper
# ---------------------------------------------------------------------------


def _make_agent(skill: MacOS, mock_model: MockModel) -> Any:
  """Create an Agent with the given skill and model (no tracing)."""
  from definable.agent import Agent
  from definable.agent.config import AgentConfig
  from definable.agent.tracing import Tracing

  return Agent(
    model=mock_model,  # type: ignore[arg-type]
    skills=[skill],
    config=AgentConfig(tracing=Tracing(enabled=False)),
  )


# ---------------------------------------------------------------------------
# Screenshot tool dispatch
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
@pytest.mark.integration
@pytest.mark.desktop
class TestScreenshotToolDispatch:
  async def test_screenshot_calls_capture_screen(self):
    """Agent dispatches screenshot -> client.capture_screen() is called."""
    mock_client = _make_mock_client()
    call_count = 0

    def side_effect(messages: Any, tools: Any, **kwargs: Any) -> MagicMock:
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        return _make_response(content=None, tool_calls=[_tool_call("screenshot", "{}")])  # type: ignore[arg-type]
      return _make_response("I can see the desktop with the screenshot.")

    skill = MacOS()
    skill._client = mock_client

    agent = _make_agent(skill, MockModel(side_effect=side_effect))
    output = await agent.arun("Take a screenshot")

    mock_client.capture_screen.assert_awaited_once()
    assert output.content is not None
    assert len(output.content) > 0

  async def test_screenshot_returns_data_uri(self):
    """screenshot tool result is a data URI starting with 'data:image/png;base64,'."""
    mock_client = _make_mock_client()
    result_seen: list[str] = []
    call_count = 0

    def side_effect(messages: Any, tools: Any, **kwargs: Any) -> MagicMock:
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        return _make_response(content=None, tool_calls=[_tool_call("screenshot", "{}")])  # type: ignore[arg-type]
      # On turn 2, capture the tool result from messages
      for msg in messages:
        if hasattr(msg, "role") and msg.role == "tool":
          result_seen.append(getattr(msg, "content", ""))
      return _make_response("Screenshot taken.")

    skill = MacOS()
    skill._client = mock_client
    agent = _make_agent(skill, MockModel(side_effect=side_effect))
    await agent.arun("Take a screenshot")

    assert len(result_seen) >= 1
    assert result_seen[0].startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# Read screen tool dispatch
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
@pytest.mark.integration
@pytest.mark.desktop
class TestReadScreenToolDispatch:
  async def test_read_screen_calls_ocr_screen(self):
    """Agent dispatches read_screen -> client.ocr_screen() is called."""
    mock_client = _make_mock_client()
    call_count = 0

    def side_effect(messages: Any, tools: Any, **kwargs: Any) -> MagicMock:
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        return _make_response(content=None, tool_calls=[_tool_call("read_screen", "{}")])  # type: ignore[arg-type]
      return _make_response("The screen shows: Hello, world!")

    skill = MacOS()
    skill._client = mock_client
    agent = _make_agent(skill, MockModel(side_effect=side_effect))
    await agent.arun("What text is on screen?")

    mock_client.ocr_screen.assert_awaited_once()


# ---------------------------------------------------------------------------
# Click tool dispatch
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
@pytest.mark.integration
@pytest.mark.desktop
class TestClickToolDispatch:
  async def test_click_calls_client_click_with_correct_coords(self):
    """Agent dispatches click(x, y) -> client.click(x=100, y=200) called."""
    mock_client = _make_mock_client()
    call_count = 0

    def side_effect(messages: Any, tools: Any, **kwargs: Any) -> MagicMock:
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        return _make_response(
          content=None,  # type: ignore[arg-type]
          tool_calls=[_tool_call("click", '{"x": 100, "y": 200}')],
        )
      return _make_response("Clicked at (100, 200).")

    skill = MacOS()
    skill._client = mock_client
    agent = _make_agent(skill, MockModel(side_effect=side_effect))
    await agent.arun("Click at 100, 200")

    mock_client.click.assert_awaited_once()
    call_kwargs = mock_client.click.call_args
    assert call_kwargs.kwargs.get("x") == 100
    assert call_kwargs.kwargs.get("y") == 200

  async def test_click_absent_when_input_disabled(self):
    """When enable_input=False, click is not in the tools list."""
    skill = MacOS(enable_input=False)
    tool_names = {t.name for t in skill.tools}
    assert "click" not in tool_names

  async def test_click_not_dispatched_when_input_disabled(self):
    """Model never sees click tool when enable_input=False."""
    mock_client = _make_mock_client()
    received_tool_names: set[str] = set()

    def side_effect(messages: Any, tools: Any, **kwargs: Any) -> MagicMock:
      for t in tools or []:
        received_tool_names.add(t.get("function", {}).get("name", ""))
      return _make_response("I can only read — no clicking.")

    skill = MacOS(enable_input=False)
    skill._client = mock_client
    agent = _make_agent(skill, MockModel(side_effect=side_effect))
    await agent.arun("Click on the button")

    assert "click" not in received_tool_names
    mock_client.click.assert_not_awaited()


# ---------------------------------------------------------------------------
# Open app tool dispatch
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
@pytest.mark.integration
@pytest.mark.desktop
class TestOpenAppToolDispatch:
  async def test_open_app_calls_client_open_app(self):
    """Agent dispatches open_app -> client.open_app() called with correct name."""
    mock_client = _make_mock_client()
    call_count = 0

    def side_effect(messages: Any, tools: Any, **kwargs: Any) -> MagicMock:
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        return _make_response(
          content=None,  # type: ignore[arg-type]
          tool_calls=[_tool_call("open_app", '{"name": "Safari"}')],
        )
      return _make_response("Safari is now open.")

    skill = MacOS()
    skill._client = mock_client
    agent = _make_agent(skill, MockModel(side_effect=side_effect))
    await agent.arun("Open Safari")

    mock_client.open_app.assert_awaited_once()
    call_args = mock_client.open_app.call_args
    # open_app is called positionally: client.open_app(name)
    assert "Safari" in (call_args.args + tuple(call_args.kwargs.values()))

  async def test_allowed_apps_whitelist_blocks_unlisted_app(self):
    """When allowed_apps={"TextEdit"}, opening Safari returns error without calling client."""
    mock_client = _make_mock_client()
    tool_result_content: list[str] = []
    call_count = 0

    def side_effect(messages: Any, tools: Any, **kwargs: Any) -> MagicMock:
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        return _make_response(
          content=None,  # type: ignore[arg-type]
          tool_calls=[_tool_call("open_app", '{"name": "Safari"}')],
        )
      # Capture tool result message
      for msg in messages:
        if hasattr(msg, "role") and msg.role == "tool":
          tool_result_content.append(getattr(msg, "content", ""))
      return _make_response("I cannot open Safari — it is not allowed.")

    skill = MacOS(allowed_apps={"TextEdit"})
    skill._client = mock_client
    agent = _make_agent(skill, MockModel(side_effect=side_effect))
    await agent.arun("Open Safari")

    # Client must NOT be called — error returned before bridge call
    mock_client.open_app.assert_not_awaited()
    # Tool result must contain an error message
    assert len(tool_result_content) >= 1
    assert "allowed list" in tool_result_content[0].lower() or "not in" in tool_result_content[0].lower()

  async def test_blocked_apps_rejects_blocked_app(self):
    """When blocked_apps={"Terminal"}, opening Terminal returns error without bridge call."""
    mock_client = _make_mock_client()
    tool_result_content: list[str] = []
    call_count = 0

    def side_effect(messages: Any, tools: Any, **kwargs: Any) -> MagicMock:
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        return _make_response(
          content=None,  # type: ignore[arg-type]
          tool_calls=[_tool_call("open_app", '{"name": "Terminal"}')],
        )
      for msg in messages:
        if hasattr(msg, "role") and msg.role == "tool":
          tool_result_content.append(getattr(msg, "content", ""))
      return _make_response("Terminal is blocked.")

    skill = MacOS(blocked_apps={"Terminal"})
    skill._client = mock_client
    agent = _make_agent(skill, MockModel(side_effect=side_effect))
    await agent.arun("Open Terminal")

    mock_client.open_app.assert_not_awaited()
    assert len(tool_result_content) >= 1
    assert "blocked" in tool_result_content[0].lower()


# ---------------------------------------------------------------------------
# AppleScript gating
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
@pytest.mark.integration
@pytest.mark.desktop
class TestAppleScriptGating:
  async def test_run_applescript_dispatched_when_enabled(self):
    """run_applescript dispatches to client.run_applescript()."""
    mock_client = _make_mock_client()
    call_count = 0

    def side_effect(messages: Any, tools: Any, **kwargs: Any) -> MagicMock:
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        return _make_response(
          content=None,  # type: ignore[arg-type]
          tool_calls=[_tool_call("run_applescript", '{"script": "tell app \\"Finder\\" to activate"}')],
        )
      return _make_response("AppleScript ran successfully.")

    skill = MacOS()
    skill._client = mock_client
    agent = _make_agent(skill, MockModel(side_effect=side_effect))
    await agent.arun("Run an AppleScript to activate Finder")

    mock_client.run_applescript.assert_awaited_once()
    call_args = mock_client.run_applescript.call_args
    # run_applescript is called positionally: client.run_applescript(script)
    all_values = call_args.args + tuple(call_args.kwargs.values())
    assert any("Finder" in str(v) for v in all_values)

  async def test_run_applescript_absent_when_disabled(self):
    """When enable_applescript=False, run_applescript is not in tools."""
    skill = MacOS(enable_applescript=False)
    tool_names = {t.name for t in skill.tools}
    assert "run_applescript" not in tool_names

  async def test_run_applescript_not_dispatched_when_disabled(self):
    """Model never sees run_applescript when enable_applescript=False."""
    mock_client = _make_mock_client()
    received_tool_names: set[str] = set()

    def side_effect(messages: Any, tools: Any, **kwargs: Any) -> MagicMock:
      for t in tools or []:
        received_tool_names.add(t.get("function", {}).get("name", ""))
      return _make_response("AppleScript is not available.")

    skill = MacOS(enable_applescript=False)
    skill._client = mock_client
    agent = _make_agent(skill, MockModel(side_effect=side_effect))
    await agent.arun("Run an AppleScript")

    assert "run_applescript" not in received_tool_names
    mock_client.run_applescript.assert_not_awaited()


# ---------------------------------------------------------------------------
# File write gating
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
@pytest.mark.integration
@pytest.mark.desktop
class TestFileWriteGating:
  async def test_write_file_dispatched_when_enabled(self):
    """write_file dispatches to client.write_file()."""
    mock_client = _make_mock_client()
    call_count = 0

    def side_effect(messages: Any, tools: Any, **kwargs: Any) -> MagicMock:
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        return _make_response(
          content=None,  # type: ignore[arg-type]
          tool_calls=[_tool_call("write_file", '{"path": "/tmp/test.txt", "content": "hello"}')],
        )
      return _make_response("File written.")

    skill = MacOS()
    skill._client = mock_client
    agent = _make_agent(skill, MockModel(side_effect=side_effect))
    await agent.arun("Write 'hello' to /tmp/test.txt")

    mock_client.write_file.assert_awaited_once()

  async def test_write_file_absent_when_disabled(self):
    """When enable_file_write=False, write_file is not in tools."""
    skill = MacOS(enable_file_write=False)
    tool_names = {t.name for t in skill.tools}
    assert "write_file" not in tool_names

  async def test_read_file_dispatched_when_file_write_disabled(self):
    """read_file is always available, even when enable_file_write=False."""
    mock_client = _make_mock_client()
    call_count = 0

    def side_effect(messages: Any, tools: Any, **kwargs: Any) -> MagicMock:
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        return _make_response(
          content=None,  # type: ignore[arg-type]
          tool_calls=[_tool_call("read_file", '{"path": "/tmp/test.txt"}')],
        )
      return _make_response("File contents: file contents here.")

    skill = MacOS(enable_file_write=False)
    skill._client = mock_client
    agent = _make_agent(skill, MockModel(side_effect=side_effect))
    await agent.arun("Read /tmp/test.txt")

    mock_client.read_file.assert_awaited_once()


# ---------------------------------------------------------------------------
# Bridge connection error handling
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
@pytest.mark.integration
@pytest.mark.desktop
class TestBridgeConnectionError:
  async def test_connect_error_returns_friendly_message(self):
    """When BridgeClient.capture_screen raises a connection error, tool returns friendly string."""
    import httpx

    mock_client = _make_mock_client()
    mock_client.capture_screen = AsyncMock(side_effect=httpx.ConnectError("Connection refused", request=MagicMock()))
    tool_result_content: list[str] = []
    call_count = 0

    def side_effect(messages: Any, tools: Any, **kwargs: Any) -> MagicMock:
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        return _make_response(content=None, tool_calls=[_tool_call("screenshot", "{}")])  # type: ignore[arg-type]
      for msg in messages:
        if hasattr(msg, "role") and msg.role == "tool":
          tool_result_content.append(getattr(msg, "content", ""))
      return _make_response("Bridge is not running.")

    skill = MacOS()
    skill._client = mock_client
    agent = _make_agent(skill, MockModel(side_effect=side_effect))
    await agent.arun("Take a screenshot")

    assert len(tool_result_content) >= 1
    result = tool_result_content[0].lower()
    assert "bridge" in result or "not running" in result or "http://" in result

  async def test_generic_exception_returns_error_string(self):
    """Non-connection exceptions are caught and returned as strings — never raised."""
    mock_client = _make_mock_client()
    mock_client.capture_screen = AsyncMock(side_effect=RuntimeError("Unexpected error"))
    tool_result_content: list[str] = []
    call_count = 0

    def side_effect(messages: Any, tools: Any, **kwargs: Any) -> MagicMock:
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        return _make_response(content=None, tool_calls=[_tool_call("screenshot", "{}")])  # type: ignore[arg-type]
      for msg in messages:
        if hasattr(msg, "role") and msg.role == "tool":
          tool_result_content.append(getattr(msg, "content", ""))
      return _make_response("Something went wrong.")

    skill = MacOS()
    skill._client = mock_client
    agent = _make_agent(skill, MockModel(side_effect=side_effect))
    # Must not raise — exception is caught inside the tool
    await agent.arun("Take a screenshot")

    assert len(tool_result_content) >= 1
    # The tool must return a string, not propagate the exception
    assert isinstance(tool_result_content[0], str)
    assert len(tool_result_content[0]) > 0

  async def test_open_app_connection_error_returns_friendly_message(self):
    """ConnectError on open_app returns 'Bridge is not running' message."""
    import httpx

    mock_client = _make_mock_client()
    mock_client.open_app = AsyncMock(side_effect=httpx.ConnectError("Connection refused", request=MagicMock()))
    tool_result_content: list[str] = []
    call_count = 0

    def side_effect(messages: Any, tools: Any, **kwargs: Any) -> MagicMock:
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        return _make_response(
          content=None,  # type: ignore[arg-type]
          tool_calls=[_tool_call("open_app", '{"name": "Finder"}')],
        )
      for msg in messages:
        if hasattr(msg, "role") and msg.role == "tool":
          tool_result_content.append(getattr(msg, "content", ""))
      return _make_response("Bridge seems offline.")

    skill = MacOS()
    skill._client = mock_client
    agent = _make_agent(skill, MockModel(side_effect=side_effect))
    await agent.arun("Open Finder")

    assert len(tool_result_content) >= 1
    result = tool_result_content[0].lower()
    assert "bridge" in result or "not running" in result


# ---------------------------------------------------------------------------
# System tools
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
@pytest.mark.integration
@pytest.mark.desktop
class TestSystemToolDispatch:
  async def test_send_notification_dispatched_correctly(self):
    """send_notification dispatches to client.send_notification() with correct args."""
    mock_client = _make_mock_client()
    call_count = 0

    def side_effect(messages: Any, tools: Any, **kwargs: Any) -> MagicMock:
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        return _make_response(
          content=None,  # type: ignore[arg-type]
          tool_calls=[_tool_call("send_notification", '{"title": "Hello", "message": "World"}')],
        )
      return _make_response("Notification sent.")

    skill = MacOS()
    skill._client = mock_client
    agent = _make_agent(skill, MockModel(side_effect=side_effect))
    await agent.arun("Send a notification: Hello World")

    mock_client.send_notification.assert_awaited_once()
    call_kwargs = mock_client.send_notification.call_args
    assert call_kwargs.kwargs.get("title") == "Hello"
    assert call_kwargs.kwargs.get("message") == "World"

  async def test_system_info_dispatched_correctly(self):
    """system_info tool dispatches to client.system_info()."""
    mock_client = _make_mock_client()
    call_count = 0

    def side_effect(messages: Any, tools: Any, **kwargs: Any) -> MagicMock:
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        return _make_response(
          content=None,  # type: ignore[arg-type]
          tool_calls=[_tool_call("system_info", "{}")],
        )
      return _make_response("System info retrieved.")

    skill = MacOS()
    skill._client = mock_client
    agent = _make_agent(skill, MockModel(side_effect=side_effect))
    await agent.arun("What are the system details?")

    mock_client.system_info.assert_awaited_once()


# ---------------------------------------------------------------------------
# Tool visibility: all tools visible to model
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
@pytest.mark.integration
@pytest.mark.desktop
class TestToolVisibility:
  async def test_all_default_tools_visible_to_model(self):
    """Agent passes all MacOS skill tools to the model on first call."""
    mock_client = _make_mock_client()
    received_tool_names: set[str] = set()

    def side_effect(messages: Any, tools: Any, **kwargs: Any) -> MagicMock:
      for t in tools or []:
        received_tool_names.add(t.get("function", {}).get("name", ""))
      return _make_response("I have many tools available.")

    skill = MacOS()
    skill._client = mock_client
    agent = _make_agent(skill, MockModel(side_effect=side_effect))
    await agent.arun("What can you do?")

    # Core always-present tools
    assert "screenshot" in received_tool_names
    assert "read_screen" in received_tool_names
    assert "find_text_on_screen" in received_tool_names
    assert "list_running_apps" in received_tool_names
    assert "open_app" in received_tool_names
    assert "quit_app" in received_tool_names
    assert "read_file" in received_tool_names
    assert "list_files" in received_tool_names
    assert "system_info" in received_tool_names
    # Input tools (default enabled)
    assert "click" in received_tool_names
    assert "type_text" in received_tool_names
    assert "press_key" in received_tool_names
    assert "scroll" in received_tool_names
    assert "drag" in received_tool_names
    assert "set_clipboard" in received_tool_names
    # File write tools (default enabled)
    assert "write_file" in received_tool_names
    assert "move_file" in received_tool_names
    # AppleScript (default enabled)
    assert "run_applescript" in received_tool_names

  async def test_read_only_tools_visible_to_model(self):
    """In read-only mode, input and write tools are absent from model's tool list."""
    mock_client = _make_mock_client()
    received_tool_names: set[str] = set()

    def side_effect(messages: Any, tools: Any, **kwargs: Any) -> MagicMock:
      for t in tools or []:
        received_tool_names.add(t.get("function", {}).get("name", ""))
      return _make_response("Read-only mode active.")

    skill = MacOS(enable_input=False, enable_file_write=False, enable_applescript=False)
    skill._client = mock_client
    agent = _make_agent(skill, MockModel(side_effect=side_effect))
    await agent.arun("What can you see?")

    # These must be absent
    assert "click" not in received_tool_names
    assert "type_text" not in received_tool_names
    assert "write_file" not in received_tool_names
    assert "move_file" not in received_tool_names
    assert "set_clipboard" not in received_tool_names
    assert "run_applescript" not in received_tool_names
    # Screen tools must still be present
    assert "screenshot" in received_tool_names
    assert "read_screen" in received_tool_names
    assert "read_file" in received_tool_names


# ---------------------------------------------------------------------------
# Multi-step: screenshot then click
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
@pytest.mark.integration
@pytest.mark.desktop
class TestMultiStepDispatch:
  async def test_screenshot_then_click_sequence(self):
    """Agent can call screenshot on turn 1 and click on turn 2."""
    mock_client = _make_mock_client()
    call_count = 0

    def side_effect(messages: Any, tools: Any, **kwargs: Any) -> MagicMock:
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        return _make_response(
          content=None,  # type: ignore[arg-type]
          tool_calls=[_tool_call("screenshot", "{}", "call_001")],
        )
      if call_count == 2:
        return _make_response(
          content=None,  # type: ignore[arg-type]
          tool_calls=[_tool_call("click", '{"x": 500, "y": 300}', "call_002")],
        )
      return _make_response("I clicked the button after taking a screenshot.")

    skill = MacOS()
    skill._client = mock_client
    agent = _make_agent(skill, MockModel(side_effect=side_effect))
    output = await agent.arun("Take a screenshot then click at 500, 300")

    mock_client.capture_screen.assert_awaited_once()
    mock_client.click.assert_awaited_once()
    assert output.content is not None
    assert len(mock_client.mock_calls) >= 2
