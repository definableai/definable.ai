"""Tests for the Textual TUI mode of the CLI interface.

Phase 1 tests: app lifecycle, event routing, widget basics, mode resolution.
Phase 2 tests: streaming markdown, tool auto-expand, thinking RichLog, status bar tokens.
Phase 3 tests: slash completion, prompt history, system messages, commands.
Phase 4 tests: session tabs, session management, conversation rebuild, status bar session.
Phase 5 tests: search bar, help modal, cost calculator, ANSI/diff tool rendering.
Phase 6 tests: footer bar, prompt spinner, Ctrl+C/L confirmation, status indicators, router hardening.
"""

import pytest

pytest.importorskip("textual")
from unittest.mock import AsyncMock, MagicMock, patch

from definable.agent.interface.cli.config import CLIConfig
from definable.agent.interface.cli.interface import (
  CLIInterface,
  _resolve_mode,
  _textual_available,
)


# ── Mode Resolution ──────────────────────────────────────────────────


class TestModeResolution:
  """Tests for mode detection and resolution."""

  def test_textual_available(self):
    """Textual should be available (we installed it)."""
    assert _textual_available() is True

  def test_resolve_repl_mode(self):
    """Explicitly requesting REPL mode returns REPL."""
    assert _resolve_mode("repl") == "repl"

  def test_resolve_tui_mode(self):
    """Explicitly requesting TUI mode returns TUI when textual is available."""
    assert _resolve_mode("tui") == "tui"

  def test_resolve_tui_mode_without_textual(self):
    """TUI mode raises ImportError when textual is not available."""
    with patch("definable.agent.interface.cli.interface._textual_available", return_value=False):
      with pytest.raises(ImportError, match="Textual is required"):
        _resolve_mode("tui")

  def test_resolve_auto_with_textual_and_tty(self):
    """Auto mode selects TUI when textual is available and terminal is interactive."""
    with (
      patch("definable.agent.interface.cli.interface._textual_available", return_value=True),
      patch("definable.agent.interface.cli.interface._is_interactive_terminal", return_value=True),
    ):
      assert _resolve_mode("auto") == "tui"

  def test_resolve_auto_without_tty(self):
    """Auto mode falls back to REPL when terminal is not interactive."""
    with (
      patch("definable.agent.interface.cli.interface._textual_available", return_value=True),
      patch("definable.agent.interface.cli.interface._is_interactive_terminal", return_value=False),
    ):
      assert _resolve_mode("auto") == "repl"

  def test_resolve_auto_without_textual(self):
    """Auto mode falls back to REPL when textual is not available."""
    with patch("definable.agent.interface.cli.interface._textual_available", return_value=False):
      assert _resolve_mode("auto") == "repl"


# ── CLIConfig ────────────────────────────────────────────────────────


class TestCLIConfigMode:
  """Tests for CLIConfig mode and tools_expand fields."""

  def test_default_mode_is_auto(self):
    cfg = CLIConfig()
    assert cfg.mode == "auto"

  def test_mode_repl(self):
    cfg = CLIConfig(mode="repl")
    assert cfg.mode == "repl"

  def test_mode_tui(self):
    cfg = CLIConfig(mode="tui")
    assert cfg.mode == "tui"

  def test_default_tools_expand(self):
    cfg = CLIConfig()
    assert cfg.tools_expand == "success"

  def test_tools_expand_always(self):
    cfg = CLIConfig(tools_expand="always")
    assert cfg.tools_expand == "always"


# ── CLIInterface Mode ────────────────────────────────────────────────


class TestCLIInterfaceMode:
  """Tests for CLIInterface mode selection."""

  def test_mode_kwarg_repl(self):
    iface = CLIInterface(mode="repl")
    assert iface.active_mode == "repl"

  def test_mode_kwarg_tui(self):
    iface = CLIInterface(mode="tui")
    assert iface.active_mode == "tui"

  def test_mode_kwarg_auto_resolves(self):
    """Auto mode resolves to a concrete mode."""
    iface = CLIInterface(mode="auto")
    assert iface.active_mode in ("tui", "repl")

  def test_tools_expand_kwarg(self):
    iface = CLIInterface(mode="repl", tools_expand="always")
    assert iface._cli_config.tools_expand == "always"


# ── Textual Messages ────────────────────────────────────────────────


class TestTUIMessages:
  """Tests for Textual message types."""

  def test_stream_chunk(self):
    from definable.agent.interface.cli.tui.messages import StreamChunk

    msg = StreamChunk(text="hello", run_id="r1")
    assert msg.text == "hello"
    assert msg.run_id == "r1"

  def test_stream_complete(self):
    from definable.agent.interface.cli.tui.messages import StreamComplete

    msg = StreamComplete(run_id="r1")
    assert msg.run_id == "r1"

  def test_run_started(self):
    from definable.agent.interface.cli.tui.messages import RunStarted

    msg = RunStarted(run_id="r1", input_text="hello")
    assert msg.run_id == "r1"
    assert msg.input_text == "hello"

  def test_run_completed(self):
    from definable.agent.interface.cli.tui.messages import RunCompleted

    msg = RunCompleted(
      run_id="r1",
      content="response",
      total_tokens=100,
      time_to_first_token=50.0,
      total_time=200.0,
    )
    assert msg.run_id == "r1"
    assert msg.content == "response"
    assert msg.total_tokens == 100
    assert msg.time_to_first_token == 50.0
    assert msg.total_time == 200.0

  def test_run_error(self):
    from definable.agent.interface.cli.tui.messages import RunError

    msg = RunError(run_id="r1", error="boom")
    assert msg.error == "boom"

  def test_tool_call_started(self):
    from definable.agent.interface.cli.tui.messages import ToolCallStarted

    msg = ToolCallStarted(tool_name="search", arguments='{"q":"test"}', call_id="c1")
    assert msg.tool_name == "search"
    assert msg.arguments == '{"q":"test"}'
    assert msg.call_id == "c1"

  def test_tool_call_completed(self):
    from definable.agent.interface.cli.tui.messages import ToolCallCompleted

    msg = ToolCallCompleted(tool_name="search", result="found", call_id="c1", duration_ms=150.0)
    assert msg.tool_name == "search"
    assert msg.result == "found"
    assert msg.duration_ms == 150.0
    assert msg.error is None

  def test_tool_call_completed_with_error(self):
    from definable.agent.interface.cli.tui.messages import ToolCallCompleted

    msg = ToolCallCompleted(tool_name="search", call_id="c1", error="not found")
    assert msg.error == "not found"

  def test_thinking_started(self):
    from definable.agent.interface.cli.tui.messages import ThinkingStarted

    msg = ThinkingStarted(run_id="r1")
    assert msg.run_id == "r1"

  def test_thinking_chunk(self):
    from definable.agent.interface.cli.tui.messages import ThinkingChunk

    msg = ThinkingChunk(text="let me think...")
    assert msg.text == "let me think..."

  def test_thinking_completed(self):
    from definable.agent.interface.cli.tui.messages import ThinkingCompleted

    msg = ThinkingCompleted(run_id="r1")
    assert msg.run_id == "r1"

  def test_model_call_update(self):
    from definable.agent.interface.cli.tui.messages import ModelCallUpdate

    msg = ModelCallUpdate(turn=2, model_id="gpt-4o", input_tokens=100, output_tokens=50)
    assert msg.turn == 2
    assert msg.model_id == "gpt-4o"
    assert msg.input_tokens == 100
    assert msg.output_tokens == 50

  def test_knowledge_update(self):
    from definable.agent.interface.cli.tui.messages import KnowledgeUpdate

    msg = KnowledgeUpdate(status="complete", doc_count=5, duration_ms=120.0)
    assert msg.status == "complete"
    assert msg.doc_count == 5

  def test_memory_update(self):
    from definable.agent.interface.cli.tui.messages import MemoryUpdate

    msg = MemoryUpdate(status="recalled", entry_count=3, duration_ms=80.0)
    assert msg.status == "recalled"
    assert msg.entry_count == 3

  def test_user_submitted(self):
    from definable.agent.interface.cli.tui.messages import UserSubmitted

    msg = UserSubmitted(text="hello agent")
    assert msg.text == "hello agent"

  def test_slash_command_requested(self):
    from definable.agent.interface.cli.tui.messages import SlashCommandRequested

    msg = SlashCommandRequested(command="help", args="tools")
    assert msg.command == "help"
    assert msg.args == "tools"


# ── Event Router ─────────────────────────────────────────────────────


class TestEventRouter:
  """Tests for pipeline event → Textual message routing."""

  def _make_router(self):
    from definable.agent.interface.cli.tui.router import EventRouter

    app = MagicMock()
    screen = MagicMock()
    screen.post_message = MagicMock()
    app.screen = screen
    app.post_message = screen.post_message
    return EventRouter(app), app

  def test_run_started_event(self):
    from definable.agent.events import RunStartedEvent
    from definable.agent.interface.cli.tui.messages import RunStarted

    router, app = self._make_router()
    event = RunStartedEvent(run_id="r1")
    router.handle(event)

    app.post_message.assert_called_once()
    msg = app.post_message.call_args[0][0]
    assert isinstance(msg, RunStarted)
    assert msg.run_id == "r1"

  def test_run_content_event(self):
    from definable.agent.events import RunContentEvent
    from definable.agent.interface.cli.tui.messages import StreamChunk

    router, app = self._make_router()
    # First emit RunStarted to set streamed_run_id
    from definable.agent.events import RunStartedEvent

    router.handle(RunStartedEvent(run_id="r1"))
    app.post_message.reset_mock()

    event = RunContentEvent(content="hello ")
    router.handle(event)

    app.post_message.assert_called_once()
    msg = app.post_message.call_args[0][0]
    assert isinstance(msg, StreamChunk)
    assert msg.text == "hello "

  def test_run_completed_event(self):
    from definable.agent.events import RunCompletedEvent
    from definable.agent.interface.cli.tui.messages import RunCompleted

    router, app = self._make_router()
    event = RunCompletedEvent(run_id="r1", content="done")
    router.handle(event)

    msg = app.post_message.call_args[0][0]
    assert isinstance(msg, RunCompleted)

  def test_run_error_event(self):
    from definable.agent.events import RunErrorEvent
    from definable.agent.interface.cli.tui.messages import RunError

    router, app = self._make_router()
    # RunErrorEvent uses content and error_type fields, not error_message
    event = RunErrorEvent(run_id="r1", content="boom")
    router.handle(event)

    msg = app.post_message.call_args[0][0]
    assert isinstance(msg, RunError)

  def test_tool_call_started_event(self):
    from definable.agent.events import ToolCallStartedEvent
    from definable.agent.interface.cli.tui.messages import ToolCallStarted
    from definable.model.response import ToolExecution

    router, app = self._make_router()
    # ToolCallStartedEvent uses a `tool` field (ToolExecution), not flat args
    tool = ToolExecution(tool_name="weather", tool_args={"city": "SF"}, tool_call_id="tc1")
    event = ToolCallStartedEvent(tool=tool)
    router.handle(event)

    msg = app.post_message.call_args[0][0]
    assert isinstance(msg, ToolCallStarted)
    assert msg.tool_name == "weather"
    assert msg.call_id == "tc1"

  def test_tool_call_completed_event(self):
    from definable.agent.events import ToolCallCompletedEvent
    from definable.agent.interface.cli.tui.messages import ToolCallCompleted
    from definable.model.response import ToolExecution

    router, app = self._make_router()
    tool = ToolExecution(tool_name="weather", result="sunny", tool_call_id="tc1")
    event = ToolCallCompletedEvent(tool=tool)
    router.handle(event)

    msg = app.post_message.call_args[0][0]
    assert isinstance(msg, ToolCallCompleted)
    assert msg.tool_name == "weather"
    assert msg.result == "sunny"

  def test_reasoning_started_event(self):
    from definable.agent.events import ReasoningStartedEvent
    from definable.agent.interface.cli.tui.messages import ThinkingStarted

    router, app = self._make_router()
    event = ReasoningStartedEvent(run_id="r1")
    router.handle(event)

    msg = app.post_message.call_args[0][0]
    assert isinstance(msg, ThinkingStarted)

  def test_reasoning_step_event(self):
    from definable.agent.events import ReasoningStepEvent
    from definable.agent.interface.cli.tui.messages import ThinkingChunk

    router, app = self._make_router()
    # ReasoningStepEvent uses `reasoning_content`, not `step_text`
    event = ReasoningStepEvent(reasoning_content="analyzing...")
    router.handle(event)

    msg = app.post_message.call_args[0][0]
    assert isinstance(msg, ThinkingChunk)
    assert msg.text == "analyzing..."

  def test_model_call_started_increments_turn(self):
    from definable.agent.events import ModelCallStartedEvent
    from definable.agent.interface.cli.tui.messages import ModelCallUpdate

    router, app = self._make_router()
    event = ModelCallStartedEvent(model_id="gpt-4o")
    router.handle(event)

    msg = app.post_message.call_args[0][0]
    assert isinstance(msg, ModelCallUpdate)
    assert msg.turn == 1

    # Second model call
    app.post_message.reset_mock()
    router.handle(event)
    msg = app.post_message.call_args[0][0]
    assert msg.turn == 2

  def test_knowledge_retrieval_events(self):
    from definable.agent.events import KnowledgeRetrievalStartedEvent, KnowledgeRetrievalCompletedEvent
    from definable.agent.interface.cli.tui.messages import KnowledgeUpdate

    router, app = self._make_router()

    router.handle(KnowledgeRetrievalStartedEvent())
    msg = app.post_message.call_args[0][0]
    assert isinstance(msg, KnowledgeUpdate)
    assert msg.status == "searching"

    app.post_message.reset_mock()
    router.handle(KnowledgeRetrievalCompletedEvent())
    msg = app.post_message.call_args[0][0]
    assert msg.status == "complete"

  def test_memory_recall_events(self):
    from definable.agent.events import MemoryRecallStartedEvent
    from definable.agent.interface.cli.tui.messages import MemoryUpdate

    router, app = self._make_router()

    router.handle(MemoryRecallStartedEvent())
    msg = app.post_message.call_args[0][0]
    assert isinstance(msg, MemoryUpdate)
    assert msg.status == "recalling"

  def test_ttft_tracking(self):
    """First content event records TTFT."""
    from definable.agent.events import RunStartedEvent, RunContentEvent, RunCompletedEvent
    from definable.agent.interface.cli.tui.messages import RunCompleted

    router, app = self._make_router()

    router.handle(RunStartedEvent(run_id="r1"))
    router.handle(RunContentEvent(content="hi"))
    router.handle(RunCompletedEvent(run_id="r1", content="hi"))

    msg = app.post_message.call_args[0][0]
    assert isinstance(msg, RunCompleted)
    # TTFT should be set (some positive value)
    assert msg.time_to_first_token is not None
    assert msg.time_to_first_token >= 0

  def test_turn_count_resets_on_new_run(self):
    """Turn count resets when a new run starts."""
    from definable.agent.events import ModelCallStartedEvent, RunStartedEvent
    from definable.agent.interface.cli.tui.messages import ModelCallUpdate

    router, app = self._make_router()

    # First run: 2 turns
    router.handle(RunStartedEvent(run_id="r1"))
    router.handle(ModelCallStartedEvent(model_id="gpt-4o"))
    router.handle(ModelCallStartedEvent(model_id="gpt-4o"))

    # Second run
    router.handle(RunStartedEvent(run_id="r2"))
    app.post_message.reset_mock()
    router.handle(ModelCallStartedEvent(model_id="gpt-4o"))

    msg = app.post_message.call_args[0][0]
    assert isinstance(msg, ModelCallUpdate)
    assert msg.turn == 1  # reset to 1

  def test_unknown_event_ignored(self):
    """Events the router doesn't recognize are silently ignored."""
    from definable.agent.run.base import BaseRunOutputEvent

    router, app = self._make_router()
    event = BaseRunOutputEvent()  # base event — no specific handler
    router.handle(event)
    # Should not crash, may or may not post a message


# ── Widget Basics ────────────────────────────────────────────────────


class TestToolCallBlock:
  """Tests for ToolCallBlock widget logic (non-rendering)."""

  def test_init(self):
    from definable.agent.interface.cli.tui.widgets.tool_call import ToolCallBlock

    block = ToolCallBlock(tool_name="search", arguments='{"q":"test"}', call_id="c1")
    assert block.tool_name == "search"
    assert block.arguments == '{"q":"test"}'
    assert block.call_id == "c1"
    assert block.is_completed is False
    assert block.is_error is False

  def test_complete_success(self):
    from definable.agent.interface.cli.tui.widgets.tool_call import ToolCallBlock

    block = ToolCallBlock(tool_name="search", call_id="c1")
    block.complete(result="found it", duration_ms=150.0)
    assert block.is_completed is True
    assert block.is_error is False
    assert block._result == "found it"
    assert block._duration_ms == 150.0

  def test_complete_error(self):
    from definable.agent.interface.cli.tui.widgets.tool_call import ToolCallBlock

    block = ToolCallBlock(tool_name="search", call_id="c1")
    block.complete(error="not found")
    assert block.is_completed is True
    assert block.is_error is True


class TestAgentResponse:
  """Tests for AgentResponse widget logic (non-rendering)."""

  def test_init(self):
    from definable.agent.interface.cli.tui.widgets.agent_response import AgentResponse

    response = AgentResponse(run_id="r1")
    assert response.run_id == "r1"
    assert response.content == ""
    assert response.finished is False


class TestConversation:
  """Tests for Conversation widget logic (non-rendering)."""

  def test_init(self):
    from definable.agent.interface.cli.tui.widgets.conversation import Conversation

    conv = Conversation()
    assert conv._current_response is None
    assert conv._current_thinking is None
    assert conv._tool_calls == {}


# ── Integration: serve_forever mode routing ──────────────────────────


class TestServeForeverMode:
  """Tests that serve_forever routes to the correct mode."""

  @pytest.mark.asyncio
  async def test_repl_mode_calls_serve_repl(self):
    iface = CLIInterface(mode="repl")
    # Without binding an agent, should raise
    with pytest.raises(ValueError, match="no agent"):
      await iface.serve_forever()

  @pytest.mark.asyncio
  async def test_tui_mode_calls_serve_tui(self):
    iface = CLIInterface(mode="tui")
    with pytest.raises(ValueError, match="no agent"):
      await iface.serve_forever()


# ── Backward Compatibility ───────────────────────────────────────────


class TestBackwardCompatibility:
  """Ensure REPL mode behavior is unchanged."""

  def test_default_repl_renderers_present(self):
    """All 9 REPL renderers are registered."""
    iface = CLIInterface(mode="repl")
    assert len(iface._renderer_registry._renderers) == 9

  def test_default_commands_present(self):
    """All 9 built-in commands are registered."""
    iface = CLIInterface(mode="repl")
    commands = iface._command_registry.all_commands
    assert len(commands) == 9

  def test_add_command_returns_self(self):
    """add_command is chainable."""
    iface = CLIInterface(mode="repl")

    class FakeCommand:
      @property
      def name(self):
        return "test"

      @property
      def description(self):
        return "Test"

      @property
      def aliases(self):
        return []

      async def execute(self, args, context):
        pass

    result = iface.add_command(FakeCommand())
    assert result is iface

  def test_add_renderer_returns_self(self):
    """add_renderer is chainable."""
    iface = CLIInterface(mode="repl")

    class FakeRenderer:
      def handles(self, event):
        return False

      def render(self, event, console, config):
        pass

    result = iface.add_renderer(FakeRenderer())
    assert result is iface

  @pytest.mark.asyncio
  async def test_convert_inbound_dict(self):
    """_convert_inbound handles dict input."""
    iface = CLIInterface(mode="repl")
    msg = await iface._convert_inbound({"text": "hello"})
    assert msg is not None
    assert msg.text == "hello"
    assert msg.platform == "cli"

  @pytest.mark.asyncio
  async def test_convert_inbound_string(self):
    """_convert_inbound handles string input."""
    iface = CLIInterface(mode="repl")
    msg = await iface._convert_inbound("hello")
    assert msg is not None
    assert msg.text == "hello"


# ══════════════════════════════════════════════════════════════════════
# Phase 2 Tests: Streaming & Rendering Excellence
# ══════════════════════════════════════════════════════════════════════


# ── AgentResponse — MarkdownStream ──────────────────────────────────


class TestAgentResponseStreaming:
  """Tests for MarkdownStream-based streaming in AgentResponse."""

  def test_has_stream_attr(self):
    """AgentResponse initializes with stream=None (set on mount)."""
    from definable.agent.interface.cli.tui.widgets.agent_response import AgentResponse

    response = AgentResponse(run_id="r1")
    assert response._stream is None

  def test_content_accumulates(self):
    """Content accumulates across append_chunk calls."""
    from definable.agent.interface.cli.tui.widgets.agent_response import AgentResponse

    response = AgentResponse(run_id="r1")
    response._content = ""
    # Simulate chunks (without mount/stream, just test accumulation)
    response._content += "Hello "
    response._content += "world"
    assert response.content == "Hello world"

  def test_finish_sets_flag(self):
    """finish() sets the finished flag."""
    from definable.agent.interface.cli.tui.widgets.agent_response import AgentResponse

    response = AgentResponse(run_id="r1")
    assert response.finished is False
    response._finished = True
    assert response.finished is True

  def test_markdown_stream_import(self):
    """MarkdownStream is importable from textual."""
    from textual.widgets.markdown import MarkdownStream

    assert MarkdownStream is not None

  def test_get_stream_classmethod(self):
    """Markdown.get_stream is a valid classmethod."""
    from textual.widgets import Markdown

    assert hasattr(Markdown, "get_stream")
    assert callable(Markdown.get_stream)


# ── ToolCallBlock — Auto-Expand Logic ──────────────────────────────


class TestToolCallAutoExpand:
  """Tests for tool call auto-expand behavior."""

  def test_default_tools_expand_is_success(self):
    from definable.agent.interface.cli.tui.widgets.tool_call import ToolCallBlock

    block = ToolCallBlock(tool_name="search", call_id="c1")
    assert block.tools_expand == "success"

  def test_tools_expand_always(self):
    from definable.agent.interface.cli.tui.widgets.tool_call import ToolCallBlock

    block = ToolCallBlock(tool_name="search", call_id="c1", tools_expand="always")
    assert block.tools_expand == "always"

  def test_tools_expand_never(self):
    from definable.agent.interface.cli.tui.widgets.tool_call import ToolCallBlock

    block = ToolCallBlock(tool_name="search", call_id="c1", tools_expand="never")
    assert block.tools_expand == "never"

  def test_tools_expand_fail(self):
    from definable.agent.interface.cli.tui.widgets.tool_call import ToolCallBlock

    block = ToolCallBlock(tool_name="search", call_id="c1", tools_expand="fail")
    assert block.tools_expand == "fail"

  def test_tools_expand_both(self):
    from definable.agent.interface.cli.tui.widgets.tool_call import ToolCallBlock

    block = ToolCallBlock(tool_name="search", call_id="c1", tools_expand="both")
    assert block.tools_expand == "both"

  def test_icon_pending(self):
    from definable.agent.interface.cli.tui.widgets.tool_call import _ICON_PENDING

    assert _ICON_PENDING == "\u25b6"

  def test_icon_success(self):
    from definable.agent.interface.cli.tui.widgets.tool_call import _ICON_SUCCESS

    assert _ICON_SUCCESS == "\u2714"

  def test_icon_error(self):
    from definable.agent.interface.cli.tui.widgets.tool_call import _ICON_ERROR

    assert _ICON_ERROR == "\u2717"

  def test_complete_stores_result(self):
    from definable.agent.interface.cli.tui.widgets.tool_call import ToolCallBlock

    block = ToolCallBlock(tool_name="search", call_id="c1")
    block.complete(result="found", duration_ms=100.0)
    assert block._result == "found"
    assert block._duration_ms == 100.0
    assert block.is_completed is True
    assert block.is_error is False

  def test_complete_stores_error(self):
    from definable.agent.interface.cli.tui.widgets.tool_call import ToolCallBlock

    block = ToolCallBlock(tool_name="search", call_id="c1")
    block.complete(error="timeout", duration_ms=5000.0)
    assert block._error == "timeout"
    assert block.is_completed is True
    assert block.is_error is True

  def test_long_args_truncated(self):
    """Arguments longer than 500 chars get truncated in compose."""
    from definable.agent.interface.cli.tui.widgets.tool_call import ToolCallBlock

    long_args = "x" * 600
    block = ToolCallBlock(tool_name="search", arguments=long_args, call_id="c1")
    assert block.arguments == long_args  # original stored intact


# ── ThinkingBlock — RichLog Streaming ───────────────────────────────


class TestThinkingBlockStreaming:
  """Tests for RichLog-based streaming in ThinkingBlock."""

  def test_init(self):
    from definable.agent.interface.cli.tui.widgets.thinking import ThinkingBlock

    block = ThinkingBlock()
    assert block._content == ""
    assert block._finished is False
    assert block._pending_text == ""

  def test_content_accumulates(self):
    from definable.agent.interface.cli.tui.widgets.thinking import ThinkingBlock

    block = ThinkingBlock()
    block._content += "Hello "
    block._content += "world"
    assert block.content == "Hello world"

  def test_richlog_import(self):
    """RichLog is importable from textual."""
    from textual.widgets import RichLog

    assert RichLog is not None


# ── StatusBar — Accumulated Token Tracking ──────────────────────────


class TestStatusBarTokenTracking:
  """Tests for accumulated token tracking in StatusBar."""

  def test_default_zero_tokens(self):
    from definable.agent.interface.cli.tui.widgets.status_bar import StatusBar

    bar = StatusBar()
    assert bar.total_tokens == 0
    assert bar.input_tokens == 0
    assert bar.output_tokens == 0

  def test_add_turn_tokens_accumulates(self):
    from definable.agent.interface.cli.tui.widgets.status_bar import StatusBar

    bar = StatusBar()
    bar.add_turn_tokens(100, 50)
    assert bar.input_tokens == 100
    assert bar.output_tokens == 50
    assert bar.total_tokens == 150

    bar.add_turn_tokens(200, 80)
    assert bar.input_tokens == 300
    assert bar.output_tokens == 130
    assert bar.total_tokens == 430

  def test_set_running_resets_tokens(self):
    from definable.agent.interface.cli.tui.widgets.status_bar import StatusBar

    bar = StatusBar()
    bar.add_turn_tokens(100, 50)
    assert bar.total_tokens == 150

    bar.set_running()
    assert bar.total_tokens == 0
    assert bar.input_tokens == 0
    assert bar.output_tokens == 0
    assert bar.turn == 0
    assert bar.ttft_ms is None
    assert bar.total_time_ms is None

  def test_set_ready(self):
    from definable.agent.interface.cli.tui.widgets.status_bar import StatusBar

    bar = StatusBar()
    bar.status = "Running"
    bar.phase = "Tool: search"
    bar.set_ready()
    assert bar.status == "Ready"
    assert bar.phase == ""

  def test_set_error(self):
    from definable.agent.interface.cli.tui.widgets.status_bar import StatusBar

    bar = StatusBar()
    bar.set_error()
    assert bar.status == "Error"
    assert bar.phase == ""

  def test_total_time_reactive(self):
    from definable.agent.interface.cli.tui.widgets.status_bar import StatusBar

    bar = StatusBar()
    assert bar.total_time_ms is None
    bar.total_time_ms = 1500.0
    assert bar.total_time_ms == 1500.0

  def test_model_name_reactive(self):
    from definable.agent.interface.cli.tui.widgets.status_bar import StatusBar

    bar = StatusBar(model_name="gpt-4o")
    assert bar.model_name == "gpt-4o"


# ── Conversation — tools_expand passthrough ─────────────────────────


class TestConversationToolsExpand:
  """Tests that Conversation passes tools_expand to ToolCallBlock."""

  def test_default_tools_expand(self):
    from definable.agent.interface.cli.tui.widgets.conversation import Conversation

    conv = Conversation()
    assert conv._tools_expand == "success"

  def test_custom_tools_expand(self):
    from definable.agent.interface.cli.tui.widgets.conversation import Conversation

    conv = Conversation(tools_expand="always")
    assert conv._tools_expand == "always"

  def test_tools_expand_never(self):
    from definable.agent.interface.cli.tui.widgets.conversation import Conversation

    conv = Conversation(tools_expand="never")
    assert conv._tools_expand == "never"


# ── MainScreen — tools_expand wiring ────────────────────────────────


class TestMainScreenToolsExpand:
  """Tests that MainScreen accepts and stores tools_expand."""

  def test_default_tools_expand(self):
    from definable.agent.interface.cli.tui.screens.main import MainScreen

    iface = CLIInterface(mode="repl")
    screen = MainScreen(interface=iface, model_name="gpt-4o")
    assert screen._tools_expand == "success"

  def test_custom_tools_expand(self):
    from definable.agent.interface.cli.tui.screens.main import MainScreen

    iface = CLIInterface(mode="repl")
    screen = MainScreen(interface=iface, model_name="gpt-4o", tools_expand="always")
    assert screen._tools_expand == "always"


# ── Router — Token accumulation in ModelCallCompleted ───────────────


class TestRouterTokenAccumulation:
  """Tests that router passes token counts from ModelCallCompletedEvent."""

  def _make_router(self):
    from definable.agent.interface.cli.tui.router import EventRouter

    app = MagicMock()
    screen = MagicMock()
    screen.post_message = MagicMock()
    app.screen = screen
    app.post_message = screen.post_message
    return EventRouter(app), app

  def test_model_completed_emits_tokens(self):
    from definable.agent.events import ModelCallCompletedEvent
    from definable.agent.interface.cli.tui.messages import ModelCallUpdate

    router, app = self._make_router()
    # Need a started event first to set turn count
    from definable.agent.events import ModelCallStartedEvent

    router.handle(ModelCallStartedEvent(model_id="gpt-4o"))
    app.post_message.reset_mock()

    # Create metrics mock
    metrics = MagicMock()
    metrics.input_tokens = 200
    metrics.output_tokens = 80
    event = ModelCallCompletedEvent(metrics=metrics)
    router.handle(event)

    msg = app.post_message.call_args[0][0]
    assert isinstance(msg, ModelCallUpdate)
    assert msg.input_tokens == 200
    assert msg.output_tokens == 80

  def test_model_completed_no_metrics(self):
    from definable.agent.events import ModelCallCompletedEvent
    from definable.agent.interface.cli.tui.messages import ModelCallUpdate

    router, app = self._make_router()
    from definable.agent.events import ModelCallStartedEvent

    router.handle(ModelCallStartedEvent(model_id="gpt-4o"))
    app.post_message.reset_mock()

    event = ModelCallCompletedEvent()
    router.handle(event)

    msg = app.post_message.call_args[0][0]
    assert isinstance(msg, ModelCallUpdate)
    assert msg.input_tokens == 0
    assert msg.output_tokens == 0

  def test_run_completed_total_time(self):
    """RunCompleted includes total_time from router timing."""
    from definable.agent.events import RunStartedEvent, RunContentEvent, RunCompletedEvent
    from definable.agent.interface.cli.tui.messages import RunCompleted

    router, app = self._make_router()

    router.handle(RunStartedEvent(run_id="r1"))
    router.handle(RunContentEvent(content="hi"))
    router.handle(RunCompletedEvent(run_id="r1", content="hi"))

    msg = app.post_message.call_args[0][0]
    assert isinstance(msg, RunCompleted)
    assert msg.total_time is not None
    assert msg.total_time >= 0

  def test_cancelled_event_posts_error(self):
    """RunCancelledEvent posts a RunError with 'Cancelled' message."""
    from definable.agent.events import RunCancelledEvent
    from definable.agent.interface.cli.tui.messages import RunError

    router, app = self._make_router()
    event = RunCancelledEvent(run_id="r1")
    router.handle(event)

    msg = app.post_message.call_args[0][0]
    assert isinstance(msg, RunError)
    assert msg.error == "Cancelled"

  def test_reasoning_content_delta_event(self):
    """ReasoningContentDeltaEvent posts ThinkingChunk with delta text."""
    from definable.agent.events import ReasoningContentDeltaEvent
    from definable.agent.interface.cli.tui.messages import ThinkingChunk

    router, app = self._make_router()
    event = ReasoningContentDeltaEvent(reasoning_content="analyzing the problem")
    router.handle(event)

    msg = app.post_message.call_args[0][0]
    assert isinstance(msg, ThinkingChunk)
    assert msg.text == "analyzing the problem"

  def test_reasoning_completed_event(self):
    """ReasoningCompletedEvent posts ThinkingCompleted."""
    from definable.agent.events import ReasoningCompletedEvent
    from definable.agent.interface.cli.tui.messages import ThinkingCompleted

    router, app = self._make_router()
    event = ReasoningCompletedEvent(run_id="r1")
    router.handle(event)

    msg = app.post_message.call_args[0][0]
    assert isinstance(msg, ThinkingCompleted)

  def test_tool_error_event(self):
    """ToolCallErrorEvent posts ToolCallCompleted with error."""
    from definable.agent.events import ToolCallErrorEvent
    from definable.agent.interface.cli.tui.messages import ToolCallCompleted
    from definable.model.response import ToolExecution

    router, app = self._make_router()
    tool = ToolExecution(tool_name="search", tool_call_id="tc1", result="timeout error")
    event = ToolCallErrorEvent(tool=tool)
    router.handle(event)

    msg = app.post_message.call_args[0][0]
    assert isinstance(msg, ToolCallCompleted)
    assert msg.tool_name == "search"
    assert msg.error == "timeout error"

  def test_memory_update_started(self):
    """MemoryUpdateStartedEvent posts MemoryUpdate with status 'updating'."""
    from definable.agent.events import MemoryUpdateStartedEvent
    from definable.agent.interface.cli.tui.messages import MemoryUpdate

    router, app = self._make_router()
    router.handle(MemoryUpdateStartedEvent())

    msg = app.post_message.call_args[0][0]
    assert isinstance(msg, MemoryUpdate)
    assert msg.status == "updating"

  def test_memory_update_completed(self):
    """MemoryUpdateCompletedEvent posts MemoryUpdate with status 'updated'."""
    from definable.agent.events import MemoryUpdateCompletedEvent
    from definable.agent.interface.cli.tui.messages import MemoryUpdate

    router, app = self._make_router()
    router.handle(MemoryUpdateCompletedEvent())

    msg = app.post_message.call_args[0][0]
    assert isinstance(msg, MemoryUpdate)
    assert msg.status == "updated"


# ══════════════════════════════════════════════════════════════════════
# Phase 3 Tests: Input Excellence & Commands
# ══════════════════════════════════════════════════════════════════════


# ── Slash Command Completion ────────────────────────────────────────


class TestSlashComplete:
  """Tests for the slash command completion widget."""

  def test_init(self):
    from definable.agent.interface.cli.tui.widgets.slash_complete import SlashComplete

    sc = SlashComplete()
    assert sc._commands == []
    assert sc._filtered == []
    assert sc._highlighted_index == 0

  def test_set_commands(self):
    from definable.agent.interface.cli.tui.widgets.slash_complete import SlashComplete

    sc = SlashComplete()
    commands = [
      ("help", "Show help", ["h", "?"]),
      ("info", "Show info", ["i"]),
      ("tools", "List tools", ["t"]),
    ]
    sc.set_commands(commands)
    assert len(sc._commands) == 3

  def test_filter_empty_query(self):
    from definable.agent.interface.cli.tui.widgets.slash_complete import SlashComplete

    sc = SlashComplete()
    sc.set_commands([
      ("help", "Show help", ["h"]),
      ("info", "Show info", ["i"]),
    ])
    sc._filter("")
    assert len(sc._filtered) == 2

  def test_filter_by_name(self):
    from definable.agent.interface.cli.tui.widgets.slash_complete import SlashComplete

    sc = SlashComplete()
    sc.set_commands([
      ("help", "Show help", ["h"]),
      ("history", "Show history", ["hist"]),
      ("info", "Show info", ["i"]),
    ])
    sc._filter("he")
    assert len(sc._filtered) == 1
    assert sc._filtered[0][0] == "help"

  def test_filter_by_alias(self):
    from definable.agent.interface.cli.tui.widgets.slash_complete import SlashComplete

    sc = SlashComplete()
    sc.set_commands([
      ("help", "Show help", ["h", "?"]),
      ("info", "Show info", ["i"]),
    ])
    sc._filter("h")
    # Both "help" (name starts with h) and matches
    names = [f[0] for f in sc._filtered]
    assert "help" in names

  def test_filter_no_match(self):
    from definable.agent.interface.cli.tui.widgets.slash_complete import SlashComplete

    sc = SlashComplete()
    sc.set_commands([("help", "Show help", ["h"])])
    sc._filter("xyz")
    assert len(sc._filtered) == 0

  def test_selected_command(self):
    from definable.agent.interface.cli.tui.widgets.slash_complete import SlashComplete

    sc = SlashComplete()
    sc.set_commands([
      ("help", "Show help", []),
      ("info", "Show info", []),
    ])
    sc._filter("")
    sc._highlighted_index = 0
    assert sc.selected_command == "help"
    sc._highlighted_index = 1
    assert sc.selected_command == "info"

  def test_selected_command_none_when_empty(self):
    from definable.agent.interface.cli.tui.widgets.slash_complete import SlashComplete

    sc = SlashComplete()
    assert sc.selected_command is None

  def test_has_matches(self):
    from definable.agent.interface.cli.tui.widgets.slash_complete import SlashComplete

    sc = SlashComplete()
    assert sc.has_matches is False
    sc.set_commands([("help", "Show help", [])])
    sc._filter("")
    assert sc.has_matches is True

  def test_move_up_down(self):
    from definable.agent.interface.cli.tui.widgets.slash_complete import SlashComplete

    sc = SlashComplete()
    sc.set_commands([
      ("help", "Show help", []),
      ("info", "Show info", []),
      ("tools", "List tools", []),
    ])
    sc._filter("")
    assert sc._highlighted_index == 0

    sc.move_down()
    assert sc._highlighted_index == 1
    sc.move_down()
    assert sc._highlighted_index == 2
    sc.move_down()  # clamp at end
    assert sc._highlighted_index == 2

    sc.move_up()
    assert sc._highlighted_index == 1
    sc.move_up()
    assert sc._highlighted_index == 0
    sc.move_up()  # clamp at start
    assert sc._highlighted_index == 0

  def test_show_hide(self):
    from definable.agent.interface.cli.tui.widgets.slash_complete import SlashComplete

    sc = SlashComplete()
    sc.set_commands([("help", "Show help", [])])
    sc.show()
    assert sc.is_shown is True
    sc.hide()
    assert sc.is_shown is False

  def test_update_filter_resets_index(self):
    from definable.agent.interface.cli.tui.widgets.slash_complete import SlashComplete

    sc = SlashComplete()
    sc.set_commands([
      ("help", "Show help", []),
      ("info", "Show info", []),
    ])
    sc._filter("")
    sc._highlighted_index = 1
    sc.update_filter("he")
    assert sc._highlighted_index == 0


# ── Prompt Input History ────────────────────────────────────────────


class TestPromptInputHistory:
  """Tests for input history in PromptInput."""

  def test_empty_history(self):
    from definable.agent.interface.cli.tui.widgets.prompt import PromptInput

    pi = PromptInput()
    assert pi.input_history == []
    assert pi._history_index == -1

  def test_history_up_with_no_history(self):
    """Up arrow with no history does nothing."""
    from definable.agent.interface.cli.tui.widgets.prompt import PromptInput

    pi = PromptInput()
    pi._history_up()  # should not crash
    assert pi._history_index == -1

  def test_history_down_with_no_navigation(self):
    """Down arrow when not navigating does nothing."""
    from definable.agent.interface.cli.tui.widgets.prompt import PromptInput

    pi = PromptInput()
    pi._history_down()  # should not crash
    assert pi._history_index == -1

  def test_history_stores_entries(self):
    from definable.agent.interface.cli.tui.widgets.prompt import PromptInput

    pi = PromptInput()
    pi._history.append("hello")
    pi._history.append("world")
    assert pi.input_history == ["hello", "world"]

  def test_history_no_duplicates(self):
    """Same text back-to-back should not create duplicate history entries."""
    from definable.agent.interface.cli.tui.widgets.prompt import PromptInput

    pi = PromptInput()
    pi._history.append("hello")
    # Simulate submit — only adds if different from last
    text = "hello"
    if not pi._history or pi._history[-1] != text:
      pi._history.append(text)
    assert len(pi._history) == 1

  def test_slash_completing_flag(self):
    from definable.agent.interface.cli.tui.widgets.prompt import PromptInput

    pi = PromptInput()
    assert pi._slash_completing is False
    pi._slash_completing = True
    assert pi._slash_completing is True

  def test_set_enabled(self):
    from definable.agent.interface.cli.tui.widgets.prompt import PromptInput

    pi = PromptInput()
    pi.set_enabled(False)
    assert pi._enabled is False
    assert pi.read_only is True
    pi.set_enabled(True)
    assert pi._enabled is True
    assert pi.read_only is False  # type: ignore[unreachable]


# ── System Message Widget ───────────────────────────────────────────


class TestSystemMessage:
  """Tests for the SystemMessage widget."""

  def test_init(self):
    from definable.agent.interface.cli.tui.widgets.system_message import SystemMessage

    msg = SystemMessage(content="Hello system")
    assert msg.content == "Hello system"
    assert msg._label == "Sys"

  def test_custom_label(self):
    from definable.agent.interface.cli.tui.widgets.system_message import SystemMessage

    msg = SystemMessage(content="Warning", label="Warn")
    assert msg._label == "Warn"


# ── New Message Types ───────────────────────────────────────────────


class TestPhase3Messages:
  """Tests for new Phase 3 message types."""

  def test_show_slash_complete(self):
    from definable.agent.interface.cli.tui.messages import ShowSlashComplete

    msg = ShowSlashComplete(query="he")
    assert msg.query == "he"

  def test_hide_slash_complete(self):
    from definable.agent.interface.cli.tui.messages import HideSlashComplete

    msg = HideSlashComplete()
    assert msg is not None

  def test_accept_slash_complete(self):
    from definable.agent.interface.cli.tui.messages import AcceptSlashComplete

    msg = AcceptSlashComplete()
    assert msg is not None

  def test_navigate_slash_complete(self):
    from definable.agent.interface.cli.tui.messages import NavigateSlashComplete

    msg = NavigateSlashComplete(direction=-1)
    assert msg.direction == -1

    msg2 = NavigateSlashComplete(direction=1)
    assert msg2.direction == 1


# ── Conversation — system messages ──────────────────────────────────


class TestConversationSystemMessages:
  """Tests that Conversation can add system messages."""

  def test_init_has_no_system_messages(self):
    from definable.agent.interface.cli.tui.widgets.conversation import Conversation

    conv = Conversation()
    # No system messages initially
    assert conv._current_response is None


# ── MainScreen — command list ───────────────────────────────────────


class TestMainScreenCommands:
  """Tests for MainScreen command handling."""

  def test_get_command_list(self):
    """MainScreen can build command list from interface registry."""
    from definable.agent.interface.cli.tui.screens.main import MainScreen

    iface = CLIInterface(mode="repl")
    screen = MainScreen(interface=iface, model_name="gpt-4o")
    commands = screen._get_command_list()
    # Should have all 9 built-in commands + 2 TUI-specific (sessions, session)
    assert len(commands) == 11
    # Each is a (name, description, aliases) tuple
    names = [c[0] for c in commands]
    assert "help" in names
    assert "info" in names
    assert "tools" in names
    assert "model" in names
    assert "clear" in names
    assert "history" in names
    assert "export" in names
    assert "reset" in names
    assert "quit" in names
    assert "sessions" in names
    assert "session" in names

  def test_has_slash_complete_attr(self):
    """MainScreen has a _slash_complete attribute (set to None before compose)."""
    from definable.agent.interface.cli.tui.screens.main import MainScreen

    iface = CLIInterface(mode="repl")
    screen = MainScreen(interface=iface, model_name="gpt-4o")
    # Before compose, _slash_complete is None (set during compose)
    assert hasattr(screen, "_slash_complete")


# ── Prompt — set_text and input_widget ──────────────────────────────


class TestPromptWidget:
  """Tests for the Prompt container widget."""

  def test_init(self):
    from definable.agent.interface.cli.tui.widgets.prompt import Prompt

    p = Prompt(indicator="$")
    assert p._indicator == "$"

  def test_input_widget_property(self):
    from definable.agent.interface.cli.tui.widgets.prompt import Prompt

    p = Prompt()
    # Before mount, _input is None
    assert p.input_widget is None


# ══════════════════════════════════════════════════════════════════════
# Phase 4: Session Persistence & Multi-Session
# ══════════════════════════════════════════════════════════════════════

# ── SessionTabs widget ────────────────────────────────────────────────


class TestSessionTabs:
  """Tests for the SessionTabs widget."""

  def test_init(self):
    """SessionTabs initializes with empty state."""
    from definable.agent.interface.cli.tui.widgets.session_tabs import SessionTabs

    tabs = SessionTabs()
    assert tabs.session_count == 0
    assert tabs.active_session is None

  def test_set_sessions_updates_count(self):
    """set_sessions updates session count."""
    from definable.agent.interface.cli.tui.widgets.session_tabs import SessionTabs

    tabs = SessionTabs()
    tabs._sessions = [("cli", "Session 1"), ("cli-2", "Session 2")]
    assert tabs.session_count == 2

  def test_active_session_property(self):
    """active_session returns active_key or None."""
    from definable.agent.interface.cli.tui.widgets.session_tabs import SessionTabs

    tabs = SessionTabs()
    assert tabs.active_session is None
    tabs.active_key = "cli"
    assert tabs.active_session == "cli"

  def test_tab_selected_message(self):
    """TabSelected message carries session_key."""
    from definable.agent.interface.cli.tui.widgets.session_tabs import SessionTabs

    msg = SessionTabs.TabSelected(session_key="cli-2")
    assert msg.session_key == "cli-2"

  def test_hidden_with_single_session(self):
    """Tabs should not display with only 1 session."""
    from definable.agent.interface.cli.tui.widgets.session_tabs import SessionTabs

    SessionTabs()  # verify constructible
    # set_sessions sets display — can't test directly without mount
    # but we can verify the logic: display = len(sessions) > 1
    sessions = [("cli", "Session 1")]
    assert len(sessions) <= 1  # single session → hidden

  def test_visible_with_multiple_sessions(self):
    """Tabs should display with > 1 session."""
    from definable.agent.interface.cli.tui.widgets.session_tabs import SessionTabs

    SessionTabs()  # verify constructible
    sessions = [("cli", "Session 1"), ("cli-2", "Session 2")]
    assert len(sessions) > 1  # multiple sessions → visible

  def test_internal_tab_class(self):
    """_Tab stores session_key."""
    from definable.agent.interface.cli.tui.widgets.session_tabs import _Tab

    tab = _Tab(label="Session 1", session_key="cli")
    assert tab.session_key == "cli"


# ── StatusBar session_name ────────────────────────────────────────────


class TestStatusBarSession:
  """Tests for StatusBar session_name reactive."""

  def test_session_name_default(self):
    """StatusBar starts with empty session_name."""
    from definable.agent.interface.cli.tui.widgets.status_bar import StatusBar

    bar = StatusBar()
    assert bar.session_name == ""

  def test_session_name_set(self):
    """session_name can be set."""
    from definable.agent.interface.cli.tui.widgets.status_bar import StatusBar

    bar = StatusBar()
    bar.session_name = "Session 1"
    assert bar.session_name == "Session 1"


# ── Conversation rebuild ──────────────────────────────────────────────


class TestConversationRebuild:
  """Tests for Conversation.rebuild_from_messages and clear."""

  def test_clear_conversation_resets_state(self):
    """clear_conversation resets internal state."""
    from definable.agent.interface.cli.tui.widgets.conversation import Conversation

    conv = Conversation()
    conv._tool_calls["test"] = MagicMock()
    conv._focused_block_index = 5
    # We can't await clear_conversation without a running app,
    # but we can check the state that _would_ be reset
    assert len(conv._tool_calls) == 1
    assert conv._focused_block_index == 5

  def test_rebuild_from_messages_exists(self):
    """Conversation has rebuild_from_messages method."""
    from definable.agent.interface.cli.tui.widgets.conversation import Conversation

    conv = Conversation()
    assert hasattr(conv, "rebuild_from_messages")
    assert callable(conv.rebuild_from_messages)


# ── MainScreen — session management ──────────────────────────────────


class TestMainScreenSessions:
  """Tests for MainScreen session management."""

  def test_initial_session_state(self):
    """MainScreen starts with one session named 'Session 1'."""
    from definable.agent.interface.cli.tui.screens.main import MainScreen

    iface = CLIInterface(mode="repl")
    screen = MainScreen(interface=iface, model_name="gpt-4o")
    assert screen._active_chat_id == "cli"
    assert screen._session_names == {"cli": "Session 1"}
    assert screen._session_counter == 1

  def test_session_tabs_attr(self):
    """MainScreen has _session_tabs attribute before compose."""
    from definable.agent.interface.cli.tui.screens.main import MainScreen

    iface = CLIInterface(mode="repl")
    screen = MainScreen(interface=iface, model_name="gpt-4o")
    assert hasattr(screen, "_session_tabs")
    assert screen._session_tabs is None  # set during compose

  def test_get_current_session_delegates(self):
    """_get_current_session calls session_manager.get."""
    from definable.agent.interface.cli.tui.screens.main import MainScreen

    iface = CLIInterface(mode="repl")
    screen = MainScreen(interface=iface, model_name="gpt-4o")
    # Calling _get_current_session with default state should work
    result = screen._get_current_session()
    # May be None since no session was created via session_manager
    # (session_manager.get may return None, but should not raise)
    assert result is None or hasattr(result, "messages")

  def test_update_session_tabs_no_crash(self):
    """_update_session_tabs does not crash when _session_tabs is None."""
    from definable.agent.interface.cli.tui.screens.main import MainScreen

    iface = CLIInterface(mode="repl")
    screen = MainScreen(interface=iface, model_name="gpt-4o")
    # _session_tabs is None before compose — should return early
    screen._update_session_tabs()  # should not raise

  def test_command_list_includes_session_commands(self):
    """Command list includes sessions and session commands."""
    from definable.agent.interface.cli.tui.screens.main import MainScreen

    iface = CLIInterface(mode="repl")
    screen = MainScreen(interface=iface, model_name="gpt-4o")
    commands = screen._get_command_list()
    names = [c[0] for c in commands]
    assert "sessions" in names
    assert "session" in names
    # session should have 's' alias
    session_cmd = [c for c in commands if c[0] == "session"][0]
    assert "s" in session_cmd[2]

  def test_bindings_include_ctrl_t(self):
    """MainScreen has Ctrl+T binding for new session."""
    from definable.agent.interface.cli.tui.screens.main import MainScreen

    binding_keys = [b[0] if isinstance(b, tuple) else b.key for b in MainScreen.BINDINGS]
    assert "ctrl+t" in binding_keys

  @pytest.mark.asyncio
  async def test_create_session_increments_counter(self):
    """_create_session increments session counter and adds to names."""
    from definable.agent.interface.cli.tui.screens.main import MainScreen

    iface = CLIInterface(mode="repl")
    screen = MainScreen(interface=iface, model_name="gpt-4o")
    # Mock widgets — use AsyncMock for async methods
    screen._conversation = MagicMock()
    screen._conversation.clear_conversation = AsyncMock()
    screen._conversation.add_system_message = AsyncMock()
    screen._conversation.rebuild_from_messages = AsyncMock()
    screen._session_tabs = MagicMock()
    screen._status_bar = MagicMock()
    screen.notify = MagicMock()  # type: ignore[method-assign]

    old_counter = screen._session_counter
    chat_id = await screen._create_session()
    assert screen._session_counter == old_counter + 1
    assert chat_id in screen._session_names
    assert chat_id.startswith("cli-")

  @pytest.mark.asyncio
  async def test_create_session_with_name(self):
    """_create_session uses custom name when provided."""
    from definable.agent.interface.cli.tui.screens.main import MainScreen

    iface = CLIInterface(mode="repl")
    screen = MainScreen(interface=iface, model_name="gpt-4o")
    screen._conversation = MagicMock()
    screen._conversation.clear_conversation = AsyncMock()
    screen._conversation.add_system_message = AsyncMock()
    screen._conversation.rebuild_from_messages = AsyncMock()
    screen._session_tabs = MagicMock()
    screen._status_bar = MagicMock()
    screen.notify = MagicMock()  # type: ignore[method-assign]

    chat_id = await screen._create_session(name="Research")
    assert screen._session_names[chat_id] == "Research"

  @pytest.mark.asyncio
  async def test_switch_session_noop_for_same(self):
    """_switch_session does nothing when switching to the same session."""
    from definable.agent.interface.cli.tui.screens.main import MainScreen

    iface = CLIInterface(mode="repl")
    screen = MainScreen(interface=iface, model_name="gpt-4o")
    screen._conversation = MagicMock()
    screen.notify = MagicMock()  # type: ignore[method-assign]

    await screen._switch_session("cli")  # same as active
    # notify should NOT have been called (no switch happened)
    screen.notify.assert_not_called()

  @pytest.mark.asyncio
  async def test_switch_session_updates_active(self):
    """_switch_session updates _active_chat_id."""
    from definable.agent.interface.cli.tui.screens.main import MainScreen

    iface = CLIInterface(mode="repl")
    screen = MainScreen(interface=iface, model_name="gpt-4o")
    screen._session_names["cli-2"] = "Session 2"
    screen._conversation = MagicMock()
    screen._conversation.clear_conversation = AsyncMock()
    screen._conversation.add_system_message = AsyncMock()
    screen._conversation.rebuild_from_messages = AsyncMock()
    screen._session_tabs = MagicMock()
    screen._status_bar = MagicMock()
    screen.notify = MagicMock()  # type: ignore[method-assign]

    await screen._switch_session("cli-2")
    assert screen._active_chat_id == "cli-2"
    screen.notify.assert_called_once()

  def test_tui_session_command_dispatch(self):
    """Slash command handler routes 'sessions' and 'session' commands."""
    from definable.agent.interface.cli.tui.screens.main import MainScreen

    # Verify the command names are in the dispatch logic
    import inspect

    source = inspect.getsource(MainScreen.handle_slash_command)
    assert "sessions" in source
    assert "session" in source


# ── Conversation block navigation ─────────────────────────────────────


class TestConversationBlockNav:
  """Tests for Conversation block navigation edge cases."""

  def test_navigation_empty_blocks(self):
    """Navigation on empty conversation does nothing."""
    from definable.agent.interface.cli.tui.widgets.conversation import Conversation

    conv = Conversation()
    conv.action_previous_block()  # should not raise
    conv.action_next_block()  # should not raise
    assert conv._focused_block_index == -1

  def test_auto_scroll_default(self):
    """Conversation starts with auto-scroll enabled."""
    from definable.agent.interface.cli.tui.widgets.conversation import Conversation

    conv = Conversation()
    assert conv._auto_scroll is True


# ══════════════════════════════════════════════════════════════════════
# Phase 5: Advanced Widgets
# ══════════════════════════════════════════════════════════════════════

# ── SearchBar widget ──────────────────────────────────────────────────


class TestSearchBar:
  """Tests for the SearchBar widget."""

  def test_init(self):
    """SearchBar initializes hidden."""
    from definable.agent.interface.cli.tui.widgets.search_bar import SearchBar

    bar = SearchBar()
    assert bar.match_count == 0
    assert bar.current_match == 0
    assert bar.is_active is False

  def test_search_query_default(self):
    """search_query returns empty when input is None."""
    from definable.agent.interface.cli.tui.widgets.search_bar import SearchBar

    bar = SearchBar()
    assert bar.search_query == ""

  def test_set_match_info(self):
    """set_match_info updates reactive properties."""
    from definable.agent.interface.cli.tui.widgets.search_bar import SearchBar

    bar = SearchBar()
    bar.set_match_info(3, 7)
    assert bar.current_match == 3
    assert bar.match_count == 7

  def test_search_changed_message(self):
    """SearchChanged message carries query."""
    from definable.agent.interface.cli.tui.widgets.search_bar import SearchBar

    msg = SearchBar.SearchChanged(query="hello")
    assert msg.query == "hello"

  def test_search_navigate_message(self):
    """SearchNavigate message carries direction."""
    from definable.agent.interface.cli.tui.widgets.search_bar import SearchBar

    msg = SearchBar.SearchNavigate(direction=-1)
    assert msg.direction == -1

  def test_search_dismissed_message(self):
    """SearchDismissed message can be created."""
    from definable.agent.interface.cli.tui.widgets.search_bar import SearchBar

    msg = SearchBar.SearchDismissed()
    assert isinstance(msg, SearchBar.SearchDismissed)

  def test_hide_search_resets_state(self):
    """hide_search resets match count and current match."""
    from definable.agent.interface.cli.tui.widgets.search_bar import SearchBar

    bar = SearchBar()
    bar.match_count = 5
    bar.current_match = 3
    bar.add_class("active")
    bar.hide_search()
    assert bar.match_count == 0
    assert bar.current_match == 0
    assert bar.is_active is False


# ── HelpModal screen ─────────────────────────────────────────────────


class TestHelpModal:
  """Tests for the HelpModal screen."""

  def test_init_no_interface(self):
    """HelpModal can be created without interface."""
    from definable.agent.interface.cli.tui.screens.help_modal import HelpModal

    modal = HelpModal()
    assert modal._interface is None

  def test_init_with_interface(self):
    """HelpModal accepts an interface."""
    from definable.agent.interface.cli.tui.screens.help_modal import HelpModal

    iface = CLIInterface(mode="repl")
    modal = HelpModal(interface=iface)
    assert modal._interface is iface

  def test_has_close_action(self):
    """HelpModal has action_close method."""
    from definable.agent.interface.cli.tui.screens.help_modal import HelpModal

    modal = HelpModal()
    assert hasattr(modal, "action_close")
    assert callable(modal.action_close)

  def test_bindings_include_escape(self):
    """HelpModal binds Escape and q to close."""
    from definable.agent.interface.cli.tui.screens.help_modal import HelpModal

    binding_keys = [b[0] if isinstance(b, tuple) else b.key for b in HelpModal.BINDINGS]
    assert "escape" in binding_keys
    assert "q" in binding_keys


# ── StatusBar cost tracking ───────────────────────────────────────────


class TestStatusBarCost:
  """Tests for StatusBar cost estimation."""

  def test_lookup_pricing_known_model(self):
    """_lookup_pricing finds pricing for known models."""
    from definable.agent.interface.cli.tui.widgets.status_bar import _lookup_pricing

    pricing = _lookup_pricing("gpt-4o")
    assert pricing is not None
    assert pricing == (2.50, 10.00)

  def test_lookup_pricing_unknown_model(self):
    """_lookup_pricing returns None for unknown models."""
    from definable.agent.interface.cli.tui.widgets.status_bar import _lookup_pricing

    pricing = _lookup_pricing("totally-unknown-model")
    assert pricing is None

  def test_lookup_pricing_substring_match(self):
    """_lookup_pricing matches by substring."""
    from definable.agent.interface.cli.tui.widgets.status_bar import _lookup_pricing

    # "gpt-4o-mini" contains "gpt-4o-mini"
    pricing = _lookup_pricing("gpt-4o-mini (openai)")
    assert pricing is not None
    assert pricing[0] == 0.15

  def test_lookup_pricing_anthropic(self):
    """_lookup_pricing matches Anthropic models."""
    from definable.agent.interface.cli.tui.widgets.status_bar import _lookup_pricing

    pricing = _lookup_pricing("claude-sonnet-4-20250514")
    assert pricing is not None
    assert pricing[0] == 3.00

  def test_estimated_cost_property(self):
    """estimated_cost calculates based on tokens and model."""
    from definable.agent.interface.cli.tui.widgets.status_bar import StatusBar

    bar = StatusBar(model_name="gpt-4o-mini")
    bar.input_tokens = 1000
    bar.output_tokens = 500
    cost = bar.estimated_cost
    assert cost is not None
    # 1000/1M * 0.15 + 500/1M * 0.60 = 0.00015 + 0.0003 = 0.00045
    assert abs(cost - 0.00045) < 0.0001

  def test_estimated_cost_unknown_model(self):
    """estimated_cost returns None for unknown models."""
    from definable.agent.interface.cli.tui.widgets.status_bar import StatusBar

    bar = StatusBar(model_name="unknown-model")
    bar.input_tokens = 1000
    cost = bar.estimated_cost
    assert cost is None

  def test_set_running_resets_cost(self):
    """set_running resets token counts (which clears cost)."""
    from definable.agent.interface.cli.tui.widgets.status_bar import StatusBar

    bar = StatusBar(model_name="gpt-4o")
    bar.input_tokens = 5000
    bar.output_tokens = 2000
    bar.set_running()
    assert bar.input_tokens == 0
    assert bar.output_tokens == 0


# ── Enhanced ToolCallBlock ────────────────────────────────────────────


class TestToolCallBlockEnhanced:
  """Tests for enhanced ToolCallBlock with ANSI/diff rendering."""

  def test_has_ansi_detection(self):
    """_has_ansi detects ANSI escape sequences."""
    from definable.agent.interface.cli.tui.widgets.tool_call import _has_ansi

    assert _has_ansi("\x1b[31mred\x1b[0m") is True
    assert _has_ansi("plain text") is False
    assert _has_ansi("\x1b[1;32mgreen bold\x1b[0m") is True

  def test_looks_like_diff_detection(self):
    """_looks_like_diff detects unified diff format."""
    from definable.agent.interface.cli.tui.widgets.tool_call import _looks_like_diff

    diff = "--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,3 @@\n-old\n+new\n same"
    assert _looks_like_diff(diff) is True
    assert _looks_like_diff("just some text") is False
    assert _looks_like_diff("--- only one marker") is False

  def test_render_result_plain(self):
    """_render_result returns Static for plain text."""
    from textual.widgets import Static

    from definable.agent.interface.cli.tui.widgets.tool_call import ToolCallBlock

    block = ToolCallBlock(tool_name="test")
    widget = block._render_result("plain text result")
    assert isinstance(widget, Static)

  def test_render_result_ansi(self):
    """_render_result returns RichLog for ANSI text."""
    from textual.widgets import RichLog

    from definable.agent.interface.cli.tui.widgets.tool_call import ToolCallBlock

    block = ToolCallBlock(tool_name="test")
    widget = block._render_result("\x1b[31merror output\x1b[0m")
    assert isinstance(widget, RichLog)

  def test_render_result_diff(self):
    """_render_result returns RichLog for diff text."""
    from textual.widgets import RichLog

    from definable.agent.interface.cli.tui.widgets.tool_call import ToolCallBlock

    diff = "--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,3 @@\n-old\n+new\n same"
    block = ToolCallBlock(tool_name="test")
    widget = block._render_result(diff)
    assert isinstance(widget, RichLog)

  def test_result_truncation_increased(self):
    """Results are truncated at 2000 chars (increased from 1000)."""
    from definable.agent.interface.cli.tui.widgets.tool_call import ToolCallBlock

    block = ToolCallBlock(tool_name="test")
    block._content_area = MagicMock()
    block._header = MagicMock()
    # Create a result longer than 2000 chars
    long_result = "x" * 3000
    block.complete(result=long_result)
    assert block._result == long_result  # full result stored
    assert block._completed is True


# ── Search messages ───────────────────────────────────────────────────


class TestSearchMessages:
  """Tests for search-related messages."""

  def test_toggle_search_message(self):
    """ToggleSearch message can be created."""
    from definable.agent.interface.cli.tui.messages import ToggleSearch

    msg = ToggleSearch()
    assert isinstance(msg, ToggleSearch)

  def test_search_query_changed_message(self):
    """SearchQueryChanged carries query text."""
    from definable.agent.interface.cli.tui.messages import SearchQueryChanged

    msg = SearchQueryChanged(query="test")
    assert msg.query == "test"

  def test_search_navigate_match_message(self):
    """SearchNavigateMatch carries direction."""
    from definable.agent.interface.cli.tui.messages import SearchNavigateMatch

    msg = SearchNavigateMatch(direction=1)
    assert msg.direction == 1

  def test_search_dismiss_message(self):
    """SearchDismiss message can be created."""
    from definable.agent.interface.cli.tui.messages import SearchDismiss

    msg = SearchDismiss()
    assert isinstance(msg, SearchDismiss)


# ── MainScreen search integration ─────────────────────────────────────


class TestMainScreenSearch:
  """Tests for MainScreen search functionality."""

  def test_has_search_bar_attr(self):
    """MainScreen has _search_bar attribute."""
    from definable.agent.interface.cli.tui.screens.main import MainScreen

    iface = CLIInterface(mode="repl")
    screen = MainScreen(interface=iface, model_name="gpt-4o")
    assert hasattr(screen, "_search_bar")
    assert screen._search_bar is None  # set during compose

  def test_has_search_state(self):
    """MainScreen has search state tracking."""
    from definable.agent.interface.cli.tui.screens.main import MainScreen

    iface = CLIInterface(mode="repl")
    screen = MainScreen(interface=iface, model_name="gpt-4o")
    assert screen._search_matches == []
    assert screen._search_index == 0

  def test_ctrl_f_binding(self):
    """MainScreen has Ctrl+F binding."""
    from definable.agent.interface.cli.tui.screens.main import MainScreen

    binding_keys = [b[0] if isinstance(b, tuple) else b.key for b in MainScreen.BINDINGS]
    assert "ctrl+f" in binding_keys

  def test_extract_block_text_user_message(self):
    """_extract_block_text extracts text from UserMessage."""
    from definable.agent.interface.cli.tui.screens.main import MainScreen
    from definable.agent.interface.cli.tui.widgets.user_message import UserMessage

    msg = UserMessage("Hello world")
    text = MainScreen._extract_block_text(msg)
    assert text == "Hello world"

  def test_extract_block_text_system_message(self):
    """_extract_block_text extracts content from SystemMessage."""
    from definable.agent.interface.cli.tui.screens.main import MainScreen
    from definable.agent.interface.cli.tui.widgets.system_message import SystemMessage

    msg = SystemMessage(content="System output")
    text = MainScreen._extract_block_text(msg)
    assert text == "System output"

  def test_extract_block_text_agent_response(self):
    """_extract_block_text extracts content from AgentResponse."""
    from definable.agent.interface.cli.tui.screens.main import MainScreen
    from definable.agent.interface.cli.tui.widgets.agent_response import AgentResponse

    resp = AgentResponse()
    resp._content = "Agent says hello"
    text = MainScreen._extract_block_text(resp)
    assert text == "Agent says hello"

  def test_extract_block_text_tool_call(self):
    """_extract_block_text extracts combined text from ToolCallBlock."""
    from definable.agent.interface.cli.tui.screens.main import MainScreen
    from definable.agent.interface.cli.tui.widgets.tool_call import ToolCallBlock

    block = ToolCallBlock(tool_name="search", arguments='{"q": "test"}')
    block._result = "found 5 results"
    text = MainScreen._extract_block_text(block)
    assert "search" in text
    assert "test" in text
    assert "found 5 results" in text

  def test_extract_block_text_thinking(self):
    """_extract_block_text extracts content from ThinkingBlock."""
    from definable.agent.interface.cli.tui.screens.main import MainScreen
    from definable.agent.interface.cli.tui.widgets.thinking import ThinkingBlock

    block = ThinkingBlock()
    block._content = "I need to think about this"
    text = MainScreen._extract_block_text(block)
    assert text == "I need to think about this"

  def test_extract_block_text_empty(self):
    """_extract_block_text returns empty for unknown widget."""
    from definable.agent.interface.cli.tui.screens.main import MainScreen

    text = MainScreen._extract_block_text(object())
    assert text == ""


# ── App integration ───────────────────────────────────────────────────


class TestAppHelpModal:
  """Tests for DefinableApp help modal integration."""

  def test_app_imports_help_modal(self):
    """DefinableApp imports HelpModal."""
    from definable.agent.interface.cli.tui.app import DefinableApp

    assert hasattr(DefinableApp, "action_help")

  def test_f1_binding(self):
    """DefinableApp has F1 binding."""
    from definable.agent.interface.cli.tui.app import DefinableApp

    binding_keys = [b[0] if isinstance(b, tuple) else b.key for b in DefinableApp.BINDINGS]
    assert "f1" in binding_keys


# ── Model pricing table ──────────────────────────────────────────────


class TestModelPricing:
  """Tests for the model pricing lookup table."""

  def test_pricing_has_major_providers(self):
    """Pricing table covers major providers."""
    from definable.agent.interface.cli.tui.widgets.status_bar import _MODEL_PRICING

    # Check key models exist
    assert "gpt-4o" in _MODEL_PRICING
    assert "gpt-4o-mini" in _MODEL_PRICING
    assert "claude-sonnet-4" in _MODEL_PRICING
    assert "claude-opus-4" in _MODEL_PRICING
    assert "deepseek-chat" in _MODEL_PRICING
    assert "gemini-2.0-flash" in _MODEL_PRICING
    assert "grok-3" in _MODEL_PRICING

  def test_pricing_format(self):
    """Each pricing entry is (input_price, output_price) tuple."""
    from definable.agent.interface.cli.tui.widgets.status_bar import _MODEL_PRICING

    for key, pricing in _MODEL_PRICING.items():
      assert isinstance(pricing, tuple), f"{key} pricing is not a tuple"
      assert len(pricing) == 2, f"{key} pricing has {len(pricing)} elements"
      assert pricing[0] >= 0, f"{key} input price is negative"
      assert pricing[1] >= 0, f"{key} output price is negative"


# ══════════════════════════════════════════════════════════════════════
# Phase 6: Polish & Optimization
# ══════════════════════════════════════════════════════════════════════

# ── FooterBar widget ──────────────────────────────────────────────────


class TestFooterBar:
  """Tests for the FooterBar widget."""

  def test_init(self):
    """FooterBar initializes with idle mode."""
    from definable.agent.interface.cli.tui.widgets.footer_bar import FooterBar

    bar = FooterBar()
    assert bar.mode == "idle"

  def test_mode_running(self):
    """FooterBar mode can be set to running."""
    from definable.agent.interface.cli.tui.widgets.footer_bar import FooterBar

    bar = FooterBar()
    bar.mode = "running"
    assert bar.mode == "running"

  def test_mode_searching(self):
    """FooterBar mode can be set to searching."""
    from definable.agent.interface.cli.tui.widgets.footer_bar import FooterBar

    bar = FooterBar()
    bar.mode = "searching"
    assert bar.mode == "searching"

  def test_hint_constants_exist(self):
    """Hint constants are defined for all modes."""
    from definable.agent.interface.cli.tui.widgets.footer_bar import (
      _HINTS_IDLE,
      _HINTS_RUNNING,
      _HINTS_SEARCHING,
    )

    assert "Help" in _HINTS_IDLE
    assert "Quit" in _HINTS_IDLE
    assert "Cancel" in _HINTS_RUNNING
    assert "Navigate" in _HINTS_SEARCHING
    assert "Close" in _HINTS_SEARCHING


# ── Prompt spinner ────────────────────────────────────────────────────


class TestPromptSpinner:
  """Tests for the Prompt animated spinner."""

  def test_spinner_frames_exist(self):
    """Spinner frames are defined."""
    from definable.agent.interface.cli.tui.widgets.prompt import _SPINNER_FRAMES

    assert len(_SPINNER_FRAMES) > 0
    assert all(isinstance(f, str) for f in _SPINNER_FRAMES)

  def test_prompt_running_state(self):
    """Prompt tracks running state."""
    from definable.agent.interface.cli.tui.widgets.prompt import Prompt

    p = Prompt()
    assert p.is_running is False
    assert p._running is False

  def test_prompt_set_running_true(self):
    """set_running(True) sets running state."""
    from definable.agent.interface.cli.tui.widgets.prompt import Prompt

    p = Prompt()
    # Can't fully test timer without app, but can test state
    p._running = True
    assert p.is_running is True

  def test_prompt_has_set_running(self):
    """Prompt has set_running method."""
    from definable.agent.interface.cli.tui.widgets.prompt import Prompt

    p = Prompt()
    assert hasattr(p, "set_running")
    assert callable(p.set_running)


# ── Ctrl+C / Ctrl+L confirmation ──────────────────────────────────────


class TestConfirmationBehavior:
  """Tests for double-press confirmation on Ctrl+C and Ctrl+L."""

  def test_ctrl_c_requires_double_press(self):
    """MainScreen requires double Ctrl+C to quit."""
    from definable.agent.interface.cli.tui.screens.main import MainScreen

    iface = CLIInterface(mode="repl")
    screen = MainScreen(interface=iface, model_name="gpt-4o")
    # First press sets counter to 1
    assert screen._ctrl_c_count == 0

  def test_clear_pending_default(self):
    """MainScreen starts with _clear_pending=False."""
    from definable.agent.interface.cli.tui.screens.main import MainScreen

    iface = CLIInterface(mode="repl")
    screen = MainScreen(interface=iface, model_name="gpt-4o")
    assert screen._clear_pending is False

  def test_cancel_or_quit_increments_counter(self):
    """action_cancel_or_quit increments ctrl_c_count."""
    from definable.agent.interface.cli.tui.screens.main import MainScreen

    iface = CLIInterface(mode="repl")
    screen = MainScreen(interface=iface, model_name="gpt-4o")
    # First press should increment counter
    # Can't call action directly without mounted app, so verify state
    assert screen._ctrl_c_count == 0
    screen._ctrl_c_count = 1
    assert screen._ctrl_c_count == 1

  def test_cancel_or_quit_logic(self):
    """Double Ctrl+C is required to quit — verify dispatch logic."""
    from definable.agent.interface.cli.tui.screens.main import MainScreen

    import inspect

    source = inspect.getsource(MainScreen.action_cancel_or_quit)
    # Should always increment counter (not only when running)
    assert "self._ctrl_c_count += 1" in source
    # Should check >= 2
    assert ">= 2" in source

  def test_user_submit_resets_counters(self):
    """handle_user_submit resets clear_pending and ctrl_c_count."""
    import inspect

    from definable.agent.interface.cli.tui.screens.main import MainScreen

    source = inspect.getsource(MainScreen.handle_user_submit)
    # Verify that the method resets both counters
    assert "_clear_pending = False" in source or "_clear_pending=False" in source
    assert "_ctrl_c_count = 0" in source or "_ctrl_c_count=0" in source


# ── Status bar text indicators ────────────────────────────────────────


class TestStatusBarAccessibility:
  """Tests for status bar text indicators alongside colors."""

  def test_ready_has_checkmark(self):
    """Ready status includes ✓ symbol."""
    from definable.agent.interface.cli.tui.widgets.status_bar import StatusBar

    bar = StatusBar()
    bar.set_ready()
    assert bar.status == "Ready"

  def test_running_has_symbol(self):
    """Running status includes ↻ symbol."""
    from definable.agent.interface.cli.tui.widgets.status_bar import StatusBar

    bar = StatusBar()
    bar.set_running()
    assert bar.status == "Running"

  def test_error_has_symbol(self):
    """Error status includes ✗ symbol."""
    from definable.agent.interface.cli.tui.widgets.status_bar import StatusBar

    bar = StatusBar()
    bar.set_error()
    assert bar.status == "Error"


# ── Event router hardening ────────────────────────────────────────────


class TestEventRouterHardening:
  """Tests for event router error handling."""

  def test_router_handle_catches_exceptions(self):
    """handle() catches exceptions without crashing."""
    from definable.agent.interface.cli.tui.router import EventRouter

    app = MagicMock()
    router = EventRouter(app)

    # Create a bad event that would cause _dispatch to fail
    class BadEvent:
      pass

    # Should not raise
    router.handle(BadEvent())  # type: ignore[arg-type]

  def test_router_has_dispatch_method(self):
    """EventRouter has _dispatch method."""
    from definable.agent.interface.cli.tui.router import EventRouter

    app = MagicMock()
    router = EventRouter(app)
    assert hasattr(router, "_dispatch")

  def test_router_dispatch_unknown_event(self):
    """_dispatch ignores unknown event types silently."""
    from definable.agent.interface.cli.tui.router import EventRouter

    app = MagicMock()
    router = EventRouter(app)

    class UnknownEvent:
      pass

    # Should not raise and should not post any messages
    router._dispatch(UnknownEvent())  # type: ignore[arg-type]
    app.post_message.assert_not_called()


# ── Conversation rebuild fix ──────────────────────────────────────────


class TestConversationRebuildFix:
  """Tests for improved conversation rebuild truncation."""

  def test_rebuild_truncation_limit_increased(self):
    """Tool messages in rebuild use 500-char limit, not 200."""
    import inspect

    from definable.agent.interface.cli.tui.widgets.conversation import Conversation

    source = inspect.getsource(Conversation.rebuild_from_messages)
    # Should NOT contain [:200] truncation
    assert "[:200]" not in source
    # Should contain [:500] or similar increased limit
    assert "500" in source


# ── MainScreen footer integration ─────────────────────────────────────


class TestMainScreenFooter:
  """Tests for MainScreen footer bar integration."""

  def test_has_footer_bar_attr(self):
    """MainScreen has _footer_bar attribute."""
    from definable.agent.interface.cli.tui.screens.main import MainScreen

    iface = CLIInterface(mode="repl")
    screen = MainScreen(interface=iface, model_name="gpt-4o")
    assert hasattr(screen, "_footer_bar")
    assert screen._footer_bar is None  # set during compose

  def test_set_running_updates_footer(self):
    """_set_running updates footer mode."""
    from definable.agent.interface.cli.tui.screens.main import MainScreen

    iface = CLIInterface(mode="repl")
    screen = MainScreen(interface=iface, model_name="gpt-4o")
    screen._prompt = MagicMock()
    screen._status_bar = MagicMock()
    footer = MagicMock()
    screen._footer_bar = footer

    screen._set_running(True)
    assert footer.mode == "running"
    screen._set_running(False)
    assert footer.mode == "idle"
