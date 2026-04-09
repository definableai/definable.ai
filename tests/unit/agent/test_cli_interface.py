"""Unit tests for the CLI interface."""

from __future__ import annotations

import json
import os
import tempfile
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from definable.agent.interface.cli.commands import CommandContext, CommandRegistry
from definable.agent.interface.cli.commands.builtin import (
  ClearCommand,
  ExportCommand,
  HelpCommand,
  HistoryCommand,
  InfoCommand,
  ModelCommand,
  QuitCommand,
  ResetCommand,
  ToolsCommand,
  register_builtins,
)
from definable.agent.interface.cli.config import CLIConfig
from definable.agent.interface.cli.interface import CLIInterface
from definable.agent.interface.cli.output import OutputManager
from definable.agent.interface.cli.renderers import RendererRegistry
from definable.agent.interface.cli.renderers.guardrail import GuardrailRenderer
from definable.agent.interface.cli.renderers.knowledge import KnowledgeRenderer
from definable.agent.interface.cli.renderers.memory import MemoryRenderer
from definable.agent.interface.cli.renderers.model import ModelCallRenderer
from definable.agent.interface.cli.renderers.reasoning import ReasoningRenderer
from definable.agent.interface.cli.renderers.research import DeepResearchRenderer
from definable.agent.interface.cli.renderers.run import RunRenderer
from definable.agent.interface.cli.renderers.streaming import StreamingRenderer
from definable.agent.interface.cli.renderers.sub_agent import SubAgentRenderer
from definable.agent.interface.cli.renderers.tool import ToolCallRenderer
from definable.agent.interface.message import InterfaceMessage
from definable.agent.interface.session import InterfaceSession, SessionManager
from definable.agent.run.agent import (
  KnowledgeRetrievalCompletedEvent,
  KnowledgeRetrievalStartedEvent,
  MemoryRecallCompletedEvent,
  MemoryRecallStartedEvent,
  MemoryUpdateCompletedEvent,
  MemoryUpdateStartedEvent,
  ModelCallCompletedEvent,
  ModelCallStartedEvent,
  ReasoningCompletedEvent,
  ReasoningStartedEvent,
  ReasoningStepEvent,
  RunCancelledEvent,
  RunCompletedEvent,
  RunContentEvent,
  RunErrorEvent,
  RunStartedEvent,
  SubAgentCompletedEvent,
  SubAgentFailedEvent,
  SubAgentKilledEvent,
  SubAgentSpawnedEvent,
  ToolCallCompletedEvent,
  ToolCallStartedEvent,
  DeepResearchStartedEvent,
  DeepResearchProgressEvent,
  DeepResearchCompletedEvent,
)
from definable.model.message import Message
from definable.model.metrics import Metrics
from definable.model.response import ToolExecution


# ---- Helpers ----


def _make_console() -> Console:
  """Create a Console that writes to a StringIO for assertion."""
  buf = StringIO()
  return Console(file=buf, highlight=False, force_terminal=True)


def _get_console_output(console: Console) -> str:
  """Get the text written to a StringIO-backed console."""
  console.file.seek(0)  # type: ignore[union-attr]
  return console.file.read()  # type: ignore[union-attr]


def _make_mock_agent(**overrides):
  """Create a minimal mock Agent for testing."""
  agent = MagicMock()
  agent.agent_name = overrides.get("agent_name", "test-agent")
  agent.model = MagicMock()
  agent.model.id = overrides.get("model_id", "gpt-4o-mini")
  agent.model.provider = overrides.get("provider", "openai")
  agent.tools = overrides.get("tools", [])
  agent._memory = overrides.get("memory", None)
  agent._knowledge = overrides.get("knowledge", None)
  agent._thinking = overrides.get("thinking", None)
  agent._tracing = overrides.get("tracing", None)
  agent._deep_research = overrides.get("deep_research", None)
  agent._guardrails = overrides.get("guardrails", None)
  agent._sub_agents = overrides.get("sub_agents", None)
  agent._model_id = overrides.get("model_id", "gpt-4o-mini")
  agent.pipeline = None
  return agent


def _make_session() -> InterfaceSession:
  """Create a minimal InterfaceSession."""
  return SessionManager().get_or_create(platform="cli", user_id="cli-user", chat_id="cli")


def _make_command_context(agent=None, session=None, output=None, interface=None):
  """Create a CommandContext for testing commands."""
  return CommandContext(
    agent=agent or _make_mock_agent(),
    session=session or _make_session(),
    output=output or OutputManager(config=CLIConfig()),
    interface=interface or MagicMock(),
  )


# ===========================================================
# CLIConfig
# ===========================================================


class TestCLIConfig:
  def test_defaults(self):
    config = CLIConfig()
    assert config.platform == "cli"
    assert config.prompt == ">>> "
    assert config.show_banner is True
    assert config.show_metrics is True
    assert config.show_tool_args is True
    assert config.show_tool_results is True
    assert config.show_thinking is True
    assert config.show_timestamps is False
    assert config.max_content_display == 500
    assert config.markdown_output is True
    assert config.command_prefix == "/"
    assert config.user_id == "cli-user"
    assert config.max_message_length == 100_000
    assert config.max_concurrent_requests == 1

  def test_with_updates(self):
    config = CLIConfig(show_banner=False, prompt="$ ")
    assert config.show_banner is False
    assert config.prompt == "$ "

    updated = config.with_updates(show_banner=True)
    assert isinstance(updated, CLIConfig)
    assert updated.show_banner is True
    assert updated.prompt == "$ "

  def test_frozen(self):
    config = CLIConfig()
    with pytest.raises(AttributeError):
      config.prompt = ">> "  # type: ignore[misc]


# ===========================================================
# RendererRegistry
# ===========================================================


class TestRendererRegistry:
  def test_add_and_dispatch(self):
    registry = RendererRegistry()
    renderer = MagicMock()
    renderer.handles.return_value = True

    registry.add(renderer)
    assert registry.renderer_count == 1

    event = RunStartedEvent()
    console = _make_console()
    config = CLIConfig()
    registry.dispatch(event, console, config)
    renderer.render.assert_called_once_with(event, console, config)

  def test_first_match_wins(self):
    registry = RendererRegistry()
    r1 = MagicMock()
    r1.handles.return_value = True
    r2 = MagicMock()
    r2.handles.return_value = True

    registry.add(r1)
    registry.add(r2)

    event = RunStartedEvent()
    console = _make_console()
    config = CLIConfig()
    registry.dispatch(event, console, config)

    r1.render.assert_called_once()
    r2.render.assert_not_called()

  def test_no_match_silent(self):
    registry = RendererRegistry()
    renderer = MagicMock()
    renderer.handles.return_value = False
    registry.add(renderer)

    event = RunStartedEvent()
    console = _make_console()
    config = CLIConfig()
    registry.dispatch(event, console, config)
    renderer.render.assert_not_called()

  def test_chaining(self):
    registry = RendererRegistry()
    result = registry.add(MagicMock())
    assert result is registry


# ===========================================================
# CommandRegistry
# ===========================================================


class TestCommandRegistry:
  def test_register_and_get(self):
    registry = CommandRegistry()
    cmd = MagicMock()
    cmd.name = "test"
    cmd.aliases = ["t"]

    registry.register(cmd)
    assert registry.get("test") is cmd
    assert registry.get("t") is cmd
    assert registry.get("unknown") is None

  def test_all_commands(self):
    registry = CommandRegistry()
    cmd1 = MagicMock()
    cmd1.name = "a"
    cmd1.aliases = []
    cmd2 = MagicMock()
    cmd2.name = "b"
    cmd2.aliases = ["bb"]

    registry.register(cmd1)
    registry.register(cmd2)
    assert len(registry.all_commands) == 2

  def test_register_builtins(self):
    registry = CommandRegistry()
    register_builtins(registry)
    assert len(registry.all_commands) == 9
    assert registry.get("help") is not None
    assert registry.get("h") is not None
    assert registry.get("?") is not None
    assert registry.get("info") is not None
    assert registry.get("tools") is not None
    assert registry.get("model") is not None
    assert registry.get("clear") is not None
    assert registry.get("history") is not None
    assert registry.get("export") is not None
    assert registry.get("reset") is not None
    assert registry.get("quit") is not None
    assert registry.get("exit") is not None
    assert registry.get("q") is not None


# ===========================================================
# Built-in Commands
# ===========================================================


class TestBuiltinCommands:
  @pytest.mark.asyncio
  async def test_help_command(self):
    ctx = _make_command_context()
    # Attach a command registry to the interface mock
    registry = CommandRegistry()
    register_builtins(registry)
    ctx.interface._command_registry = registry
    ctx.output = OutputManager(config=CLIConfig())
    ctx.output._console = _make_console()

    cmd = HelpCommand()
    await cmd.execute("", ctx)
    output = _get_console_output(ctx.output.console)
    assert "help" in output.lower()

  @pytest.mark.asyncio
  async def test_info_command(self):
    ctx = _make_command_context()
    ctx.output._console = _make_console()

    cmd = InfoCommand()
    await cmd.execute("", ctx)
    output = _get_console_output(ctx.output.console)
    assert "test-agent" in output
    assert "gpt-4o-mini" in output

  @pytest.mark.asyncio
  async def test_info_command_with_features(self):
    agent = _make_mock_agent(memory=True, thinking=True)
    ctx = _make_command_context(agent=agent)
    ctx.output._console = _make_console()

    cmd = InfoCommand()
    await cmd.execute("", ctx)
    output = _get_console_output(ctx.output.console)
    assert "memory" in output
    assert "thinking" in output

  @pytest.mark.asyncio
  async def test_tools_command_no_tools(self):
    ctx = _make_command_context()
    ctx.output._console = _make_console()

    cmd = ToolsCommand()
    await cmd.execute("", ctx)
    output = _get_console_output(ctx.output.console)
    assert "No tools configured" in output

  @pytest.mark.asyncio
  async def test_tools_command_with_tools(self):
    tool = MagicMock()
    tool.name = "search"
    tool.description = "Search the web"
    agent = _make_mock_agent(tools=[tool])
    ctx = _make_command_context(agent=agent)
    ctx.output._console = _make_console()

    cmd = ToolsCommand()
    await cmd.execute("", ctx)
    output = _get_console_output(ctx.output.console)
    assert "search" in output

  @pytest.mark.asyncio
  async def test_model_command(self):
    ctx = _make_command_context()
    ctx.output._console = _make_console()

    cmd = ModelCommand()
    await cmd.execute("", ctx)
    output = _get_console_output(ctx.output.console)
    assert "gpt-4o-mini" in output

  @pytest.mark.asyncio
  async def test_clear_command(self):
    ctx = _make_command_context()
    ctx.output._console = _make_console()

    cmd = ClearCommand()
    await cmd.execute("", ctx)
    # Just verify it doesn't error

  @pytest.mark.asyncio
  async def test_history_command_empty(self):
    ctx = _make_command_context()
    ctx.output._console = _make_console()

    cmd = HistoryCommand()
    await cmd.execute("", ctx)
    output = _get_console_output(ctx.output.console)
    assert "No messages" in output

  @pytest.mark.asyncio
  async def test_history_command_with_messages(self):
    session = _make_session()
    session.messages = [
      Message(role="user", content="hello"),
      Message(role="assistant", content="world"),
    ]
    ctx = _make_command_context(session=session)
    ctx.output._console = _make_console()

    cmd = HistoryCommand()
    await cmd.execute("", ctx)
    output = _get_console_output(ctx.output.console)
    assert "user" in output
    assert "hello" in output

  @pytest.mark.asyncio
  async def test_export_command(self):
    session = _make_session()
    session.messages = [Message(role="user", content="test")]
    ctx = _make_command_context(session=session)
    ctx.output._console = _make_console()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
      path = f.name

    try:
      cmd = ExportCommand()
      await cmd.execute(path, ctx)
      with open(path) as f:
        data = json.load(f)
      assert len(data) == 1
      assert data[0]["role"] == "user"
    finally:
      os.unlink(path)

  @pytest.mark.asyncio
  async def test_reset_command(self):
    session = _make_session()
    session.messages = [Message(role="user", content="test")]
    ctx = _make_command_context(session=session)
    ctx.output._console = _make_console()

    cmd = ResetCommand()
    await cmd.execute("", ctx)
    assert len(session.messages) == 0

  @pytest.mark.asyncio
  async def test_quit_command(self):
    interface = MagicMock()
    interface._running = True
    ctx = _make_command_context(interface=interface)
    ctx.output._console = _make_console()

    cmd = QuitCommand()
    await cmd.execute("", ctx)
    assert interface._running is False


# ===========================================================
# Renderers — handles() and render()
# ===========================================================


class TestRunRenderer:
  def test_handles(self):
    r = RunRenderer()
    assert r.handles(RunStartedEvent()) is True
    assert r.handles(RunCompletedEvent()) is True
    assert r.handles(RunErrorEvent()) is True
    assert r.handles(RunCancelledEvent()) is True
    assert r.handles(ToolCallStartedEvent()) is False

  def test_render_completed_with_metrics(self):
    r = RunRenderer()
    console = _make_console()
    config = CLIConfig(show_metrics=True)
    event = RunCompletedEvent(metrics=Metrics(total_tokens=100))
    r.render(event, console, config)
    output = _get_console_output(console)
    assert "100" in output

  def test_render_completed_metrics_off(self):
    r = RunRenderer()
    console = _make_console()
    config = CLIConfig(show_metrics=False)
    event = RunCompletedEvent(metrics=Metrics(total_tokens=100))
    r.render(event, console, config)
    output = _get_console_output(console)
    assert "100" not in output

  def test_render_error(self):
    r = RunRenderer()
    console = _make_console()
    config = CLIConfig()
    event = RunErrorEvent(content="something broke")
    r.render(event, console, config)
    output = _get_console_output(console)
    assert "something broke" in output

  def test_render_cancelled(self):
    r = RunRenderer()
    console = _make_console()
    config = CLIConfig()
    event = RunCancelledEvent(reason="user cancelled")
    r.render(event, console, config)
    output = _get_console_output(console)
    assert "user cancelled" in output


class TestModelCallRenderer:
  def test_handles(self):
    r = ModelCallRenderer()
    assert r.handles(ModelCallStartedEvent()) is True
    assert r.handles(ModelCallCompletedEvent()) is True
    assert r.handles(RunStartedEvent()) is False

  def test_render_turn_header(self):
    r = ModelCallRenderer()
    console = _make_console()
    config = CLIConfig()
    event = ModelCallStartedEvent(turn=2)
    r.render(event, console, config)
    output = _get_console_output(console)
    assert "Turn 2" in output

  def test_render_turn_1_silent(self):
    r = ModelCallRenderer()
    console = _make_console()
    config = CLIConfig()
    event = ModelCallStartedEvent(turn=1)
    r.render(event, console, config)
    output = _get_console_output(console)
    assert "Turn" not in output

  def test_render_metrics(self):
    r = ModelCallRenderer()
    console = _make_console()
    config = CLIConfig(show_metrics=True)
    event = ModelCallCompletedEvent(metrics=Metrics(input_tokens=50, output_tokens=30))
    r.render(event, console, config)
    output = _get_console_output(console)
    assert "in=50" in output
    assert "out=30" in output


class TestToolCallRenderer:
  def test_handles(self):
    r = ToolCallRenderer()
    assert r.handles(ToolCallStartedEvent()) is True
    assert r.handles(ToolCallCompletedEvent()) is True
    assert r.handles(RunStartedEvent()) is False

  def test_render_started_with_args(self):
    r = ToolCallRenderer()
    console = _make_console()
    config = CLIConfig(show_tool_args=True)
    tool = ToolExecution(tool_name="search", tool_args='{"q": "test"}')  # type: ignore[arg-type]
    event = ToolCallStartedEvent(tool=tool)
    r.render(event, console, config)
    output = _get_console_output(console)
    assert "search" in output
    assert "test" in output

  def test_render_started_args_off(self):
    r = ToolCallRenderer()
    console = _make_console()
    config = CLIConfig(show_tool_args=False)
    tool = ToolExecution(tool_name="search", tool_args='{"q": "secret"}')  # type: ignore[arg-type]
    event = ToolCallStartedEvent(tool=tool)
    r.render(event, console, config)
    output = _get_console_output(console)
    assert "search" in output
    assert "secret" not in output

  def test_render_completed_with_result(self):
    r = ToolCallRenderer()
    console = _make_console()
    config = CLIConfig(show_tool_results=True)
    tool = ToolExecution(tool_name="search", result="found 5 results")
    event = ToolCallCompletedEvent(tool=tool)
    r.render(event, console, config)
    output = _get_console_output(console)
    assert "found 5 results" in output

  def test_render_completed_results_off(self):
    r = ToolCallRenderer()
    console = _make_console()
    config = CLIConfig(show_tool_results=False)
    tool = ToolExecution(tool_name="search", result="found 5 results")
    event = ToolCallCompletedEvent(tool=tool)
    r.render(event, console, config)
    output = _get_console_output(console)
    assert "found 5 results" not in output


class TestStreamingRenderer:
  def test_handles(self):
    r = StreamingRenderer()
    assert r.handles(RunContentEvent()) is True
    assert r.handles(RunStartedEvent()) is True
    assert r.handles(RunCompletedEvent()) is False

  def test_streamed_run_id_tracking(self):
    r = StreamingRenderer()
    assert r.streamed_run_id is None

    # RunStartedEvent resets
    console = _make_console()
    config = CLIConfig()
    r.render(RunStartedEvent(run_id="run-1"), console, config)
    assert r.streamed_run_id is None

    # RunContentEvent sets it
    with patch("builtins.print"):
      r.render(RunContentEvent(content="hello", run_id="run-1"), console, config)
    assert r.streamed_run_id == "run-1"

  def test_reset_clears_state(self):
    r = StreamingRenderer()
    r._streamed_run_id = "run-1"
    r._needs_newline = True
    with patch("builtins.print"):
      r.reset()
    assert r.streamed_run_id is None
    assert r._needs_newline is False


class TestReasoningRenderer:
  def test_handles(self):
    r = ReasoningRenderer()
    assert r.handles(ReasoningStartedEvent()) is True
    assert r.handles(ReasoningStepEvent()) is True
    assert r.handles(ReasoningCompletedEvent()) is True
    assert r.handles(RunStartedEvent()) is False

  def test_render_started(self):
    r = ReasoningRenderer()
    console = _make_console()
    config = CLIConfig(show_thinking=True)
    r.render(ReasoningStartedEvent(), console, config)
    output = _get_console_output(console)
    assert "Thinking" in output

  def test_render_thinking_off(self):
    r = ReasoningRenderer()
    console = _make_console()
    config = CLIConfig(show_thinking=False)
    r.render(ReasoningStartedEvent(), console, config)
    output = _get_console_output(console)
    assert "Thinking" not in output

  def test_render_step(self):
    r = ReasoningRenderer()
    console = _make_console()
    config = CLIConfig(show_thinking=True)
    event = ReasoningStepEvent(reasoning_content="Let me think about this...")
    r.render(event, console, config)
    output = _get_console_output(console)
    assert "Let me think" in output


class TestKnowledgeRenderer:
  def test_handles(self):
    r = KnowledgeRenderer()
    assert r.handles(KnowledgeRetrievalStartedEvent()) is True
    assert r.handles(KnowledgeRetrievalCompletedEvent()) is True
    assert r.handles(RunStartedEvent()) is False

  def test_render_started(self):
    r = KnowledgeRenderer()
    console = _make_console()
    config = CLIConfig()
    r.render(KnowledgeRetrievalStartedEvent(), console, config)
    output = _get_console_output(console)
    assert "knowledge" in output.lower()

  def test_render_completed(self):
    r = KnowledgeRenderer()
    console = _make_console()
    config = CLIConfig()
    event = KnowledgeRetrievalCompletedEvent(documents_found=3, duration_ms=42.0)
    r.render(event, console, config)
    output = _get_console_output(console)
    assert "3" in output
    assert "42" in output


class TestMemoryRenderer:
  def test_handles(self):
    r = MemoryRenderer()
    assert r.handles(MemoryRecallStartedEvent()) is True
    assert r.handles(MemoryRecallCompletedEvent()) is True
    assert r.handles(MemoryUpdateStartedEvent()) is True
    assert r.handles(MemoryUpdateCompletedEvent()) is True
    assert r.handles(RunStartedEvent()) is False

  def test_render_recall_started(self):
    r = MemoryRenderer()
    console = _make_console()
    config = CLIConfig()
    r.render(MemoryRecallStartedEvent(), console, config)
    output = _get_console_output(console)
    assert "memory" in output.lower()

  def test_render_recall_completed(self):
    r = MemoryRenderer()
    console = _make_console()
    config = CLIConfig()
    event = MemoryRecallCompletedEvent(chunks_included=5, duration_ms=10.0)
    r.render(event, console, config)
    output = _get_console_output(console)
    assert "5" in output


class TestDeepResearchRenderer:
  def test_handles(self):
    r = DeepResearchRenderer()
    assert r.handles(DeepResearchStartedEvent()) is True
    assert r.handles(DeepResearchProgressEvent()) is True
    assert r.handles(DeepResearchCompletedEvent()) is True
    assert r.handles(RunStartedEvent()) is False

  def test_render_started(self):
    r = DeepResearchRenderer()
    console = _make_console()
    config = CLIConfig()
    event = DeepResearchStartedEvent(query="quantum computing")
    r.render(event, console, config)
    output = _get_console_output(console)
    assert "quantum" in output.lower()

  def test_render_progress(self):
    r = DeepResearchRenderer()
    console = _make_console()
    config = CLIConfig()
    event = DeepResearchProgressEvent(wave=2, sources_read=10, facts_extracted=5, message="Wave 2")
    r.render(event, console, config)
    output = _get_console_output(console)
    assert "Wave 2" in output

  def test_render_completed(self):
    r = DeepResearchRenderer()
    console = _make_console()
    config = CLIConfig()
    event = DeepResearchCompletedEvent(sources_used=15, waves_executed=3)
    r.render(event, console, config)
    output = _get_console_output(console)
    assert "15" in output


class TestGuardrailRenderer:
  def test_handles(self):
    r = GuardrailRenderer()
    assert r.handles(RunStartedEvent()) is False

  def test_handles_guardrail_events(self):
    from definable.agent.guardrail.events import GuardrailBlockedEvent, GuardrailCheckedEvent

    r = GuardrailRenderer()
    assert r.handles(GuardrailBlockedEvent()) is True
    assert r.handles(GuardrailCheckedEvent()) is True

  def test_render_blocked(self):
    from definable.agent.guardrail.events import GuardrailBlockedEvent

    r = GuardrailRenderer()
    console = _make_console()
    config = CLIConfig()
    event = GuardrailBlockedEvent(guardrail_name="content-filter", reason="PII detected")
    r.render(event, console, config)
    output = _get_console_output(console)
    assert "PII detected" in output


class TestSubAgentRenderer:
  def test_handles(self):
    r = SubAgentRenderer()
    assert r.handles(SubAgentSpawnedEvent()) is True
    assert r.handles(SubAgentCompletedEvent()) is True
    assert r.handles(SubAgentFailedEvent()) is True
    assert r.handles(SubAgentKilledEvent()) is True
    assert r.handles(RunStartedEvent()) is False

  def test_render_spawned(self):
    r = SubAgentRenderer()
    console = _make_console()
    config = CLIConfig()
    event = SubAgentSpawnedEvent(goal="search for information")
    r.render(event, console, config)
    output = _get_console_output(console)
    assert "search for information" in output

  def test_render_failed(self):
    r = SubAgentRenderer()
    console = _make_console()
    config = CLIConfig()
    event = SubAgentFailedEvent(error="timeout")
    r.render(event, console, config)
    output = _get_console_output(console)
    assert "timeout" in output


# ===========================================================
# CLIInterface
# ===========================================================


class TestCLIInterface:
  def test_init_defaults(self):
    iface = CLIInterface()
    assert isinstance(iface.config, CLIConfig)
    assert iface._cli_config.platform == "cli"
    assert iface._renderer_registry.renderer_count == 10
    assert len(iface._command_registry.all_commands) == 9

  def test_init_custom_config(self):
    config = CLIConfig(prompt="$ ", show_banner=False)
    iface = CLIInterface(config=config)
    assert iface._cli_config.prompt == "$ "
    assert iface._cli_config.show_banner is False

  @pytest.mark.asyncio
  async def test_convert_inbound(self):
    iface = CLIInterface()
    msg = await iface._convert_inbound({"text": "hello world"})
    assert msg is not None
    assert msg.text == "hello world"
    assert msg.platform == "cli"
    assert msg.platform_user_id == "cli-user"
    assert msg.platform_chat_id == "cli"

  @pytest.mark.asyncio
  async def test_convert_inbound_string(self):
    iface = CLIInterface()
    msg = await iface._convert_inbound("hello")
    assert msg is not None
    assert msg.text == "hello"

  def test_add_command(self):
    iface = CLIInterface()
    cmd = MagicMock()
    cmd.name = "custom"
    cmd.aliases = ["c"]
    result = iface.add_command(cmd)
    assert result is iface
    assert iface._command_registry.get("custom") is cmd

  def test_add_renderer(self):
    iface = CLIInterface()
    renderer = MagicMock()
    initial_count = iface._renderer_registry.renderer_count
    result = iface.add_renderer(renderer)
    assert result is iface
    assert iface._renderer_registry.renderer_count == initial_count + 1

  @pytest.mark.asyncio
  async def test_handle_command(self):
    iface = CLIInterface()
    iface.agent = _make_mock_agent()
    iface._output._console = _make_console()

    await iface._handle_command("/info")
    output = _get_console_output(iface._output.console)
    assert "test-agent" in output

  @pytest.mark.asyncio
  async def test_handle_unknown_command(self):
    iface = CLIInterface()
    iface.agent = _make_mock_agent()
    iface._output._console = _make_console()

    await iface._handle_command("/nonexistent")
    output = _get_console_output(iface._output.console)
    assert "Unknown command" in output

  def test_event_handler_dispatches(self):
    iface = CLIInterface()
    iface._output._console = _make_console()

    event = RunErrorEvent(content="test error")
    iface._handle_event(event)
    output = _get_console_output(iface._output.console)
    assert "test error" in output


# ===========================================================
# OutputManager
# ===========================================================


class TestOutputManager:
  def test_console_lazy_init(self):
    output = OutputManager(config=CLIConfig())
    assert output._console is None
    console = output.console
    assert console is not None
    assert output.console is console  # same instance

  def test_print_banner(self):
    output = OutputManager(config=CLIConfig(show_banner=True))
    output._console = _make_console()
    agent = _make_mock_agent()
    output.print_banner(agent)
    text = _get_console_output(output.console)
    assert "test-agent" in text
    assert "gpt-4o-mini" in text

  def test_print_banner_off(self):
    output = OutputManager(config=CLIConfig(show_banner=False))
    output._console = _make_console()
    agent = _make_mock_agent()
    output.print_banner(agent)
    text = _get_console_output(output.console)
    assert "test-agent" not in text

  def test_print_response_plain(self):
    output = OutputManager(config=CLIConfig(markdown_output=False))
    output._console = _make_console()
    output.print_response("hello world")
    text = _get_console_output(output.console)
    assert "hello world" in text

  def test_print_response_markdown(self):
    output = OutputManager(config=CLIConfig(markdown_output=True))
    output._console = _make_console()
    output.print_response("**bold text**")
    text = _get_console_output(output.console)
    assert "bold text" in text


# ===========================================================
# Lazy import from interface/__init__.py
# ===========================================================


class TestLazyImport:
  def test_cli_interface_import(self):
    from definable.agent.interface import CLIInterface as LazyImported

    assert LazyImported is CLIInterface

  def test_cli_config_import(self):
    from definable.agent.interface import CLIConfig as LazyImported

    assert LazyImported is CLIConfig

  def test_in_all(self):
    from definable.agent.interface import __all__

    assert "CLIInterface" in __all__
    assert "CLIConfig" in __all__


# ===========================================================
# Streaming flag integration
# ===========================================================


class TestStreamingFlagIntegration:
  @pytest.mark.asyncio
  async def test_send_response_skips_when_streamed(self):
    """If content was streamed via RunContentEvent, _send_response should not double-print."""
    iface = CLIInterface()
    iface._output._console = _make_console()

    # Simulate streaming having occurred
    iface._streaming_renderer._streamed_run_id = "run-1"
    iface._streaming_renderer._needs_newline = False

    msg = InterfaceMessage(
      platform="cli",
      platform_user_id="cli-user",
      platform_chat_id="cli",
      platform_message_id="msg-1",
      text="test",
    )
    from definable.agent.interface.message import InterfaceResponse

    response = InterfaceResponse(content="This was already streamed")

    await iface._send_response(msg, response, {})
    output = _get_console_output(iface._output.console)
    assert "This was already streamed" not in output

  @pytest.mark.asyncio
  async def test_send_response_prints_when_not_streamed(self):
    """If content was NOT streamed, _send_response should print it."""
    iface = CLIInterface(config=CLIConfig(markdown_output=False))
    iface._output._console = _make_console()

    # No streaming occurred
    assert iface._streaming_renderer.streamed_run_id is None

    msg = InterfaceMessage(
      platform="cli",
      platform_user_id="cli-user",
      platform_chat_id="cli",
      platform_message_id="msg-1",
      text="test",
    )
    from definable.agent.interface.message import InterfaceResponse

    response = InterfaceResponse(content="Not streamed response")

    await iface._send_response(msg, response, {})
    output = _get_console_output(iface._output.console)
    assert "Not streamed response" in output


# ===========================================================
# SlashCommandCompleter
# ===========================================================


class TestSlashCommandCompleter:
  """Tests for the prompt_toolkit slash-command completer."""

  def _make_completer(self, prefix="/"):
    from definable.agent.interface.cli.completer import SlashCommandCompleter

    registry = CommandRegistry()
    register_builtins(registry)
    return SlashCommandCompleter(registry, prefix=prefix), registry

  def _get_completions(self, completer, text):
    from prompt_toolkit.document import Document

    doc = Document(text, cursor_position=len(text))
    from prompt_toolkit.completion import CompleteEvent

    return list(completer.get_completions(doc, CompleteEvent()))

  def test_no_completions_for_regular_text(self):
    completer, _ = self._make_completer()
    completions = self._get_completions(completer, "hello world")
    assert completions == []

  def test_prefix_alone_shows_all_commands(self):
    completer, registry = self._make_completer()
    completions = self._get_completions(completer, "/")
    # Should have one entry per command name + one per alias
    names = {c.text for c in completions}
    # All builtin command names must appear
    for cmd in registry.all_commands:
      assert f"/{cmd.name}" in names

  def test_partial_filters(self):
    completer, _ = self._make_completer()
    completions = self._get_completions(completer, "/he")
    names = {c.text for c in completions}
    assert "/help" in names
    # Commands that don't start with "he" should be absent
    assert "/quit" not in names
    assert "/info" not in names

  def test_no_completions_after_space(self):
    completer, _ = self._make_completer()
    completions = self._get_completions(completer, "/export file.json")
    assert completions == []

  def test_aliases_in_completions(self):
    completer, _ = self._make_completer()
    completions = self._get_completions(completer, "/")
    names = {c.text for c in completions}
    # HelpCommand has aliases "h" and "?"
    assert "/h" in names
    assert "/?" in names

  def test_alias_meta_text(self):
    completer, _ = self._make_completer()
    completions = self._get_completions(completer, "/h")
    # Find the /h alias completion
    alias_comp = [c for c in completions if c.text == "/h"]
    assert len(alias_comp) == 1
    # display_meta is FormattedText — convert to plain string for assertion
    meta_text = "".join(t[1] for t in alias_comp[0].display_meta)
    assert "alias of /help" in meta_text

  def test_custom_prefix(self):
    completer, _ = self._make_completer(prefix="!")
    # Should not trigger on "/"
    assert self._get_completions(completer, "/") == []
    # Should trigger on "!"
    completions = self._get_completions(completer, "!")
    assert len(completions) > 0
    # All completions should start with "!"
    for c in completions:
      assert c.text.startswith("!")

  def test_start_position(self):
    completer, _ = self._make_completer()
    completions = self._get_completions(completer, "/he")
    for c in completions:
      # start_position should replace the entire typed text including prefix
      assert c.start_position == -3  # len("/he") == 3

  def test_dynamic_commands_appear(self):
    completer, registry = self._make_completer()
    # Add a new command after completer creation
    cmd = MagicMock()
    cmd.name = "custom"
    cmd.description = "A custom command"
    cmd.aliases = ["cc"]
    registry.register(cmd)

    completions = self._get_completions(completer, "/cu")
    names = {c.text for c in completions}
    assert "/custom" in names

  def test_enable_completions_config_default(self):
    config = CLIConfig()
    assert config.enable_completions is True

  def test_enable_completions_config_override(self):
    config = CLIConfig(enable_completions=False)
    assert config.enable_completions is False
