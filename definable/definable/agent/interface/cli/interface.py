"""CLI interface — interactive terminal for Definable agents.

Supports two modes:
- **tui**: Full Textual TUI with streaming markdown, collapsible tool calls,
  block navigation, and a status bar. Requires ``textual`` (``pip install definable[cli]``).
- **repl**: Simple Rich-based REPL with event rendering. Works everywhere.
- **auto** (default): Uses TUI if textual is installed and the terminal is
  interactive, otherwise falls back to REPL.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
import warnings
from typing import TYPE_CHECKING, Any, List, Optional
from uuid import uuid4

from definable.agent.interface.base import BaseInterface
from definable.agent.interface.cli.commands import BaseCommand, CommandContext, CommandRegistry
from definable.agent.interface.cli.commands.builtin import register_builtins
from definable.agent.interface.cli.config import CLIConfig
from definable.agent.interface.cli.input import InputHandler
from definable.agent.interface.cli.output import OutputManager
from definable.agent.interface.cli.renderers import BaseRenderer, RendererRegistry
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
from definable.agent.interface.message import InterfaceMessage, InterfaceResponse
from definable.utils.log import log_error, log_info

if TYPE_CHECKING:
  from definable.agent.interface.hooks import InterfaceHook
  from definable.agent.interface.identity import IdentityResolver
  from definable.agent.interface.session import SessionManager
  from definable.agent.run.base import BaseRunOutputEvent


def _textual_available() -> bool:
  """Check if textual is importable."""
  try:
    import textual  # noqa: F401

    return True
  except ImportError:
    return False


def _is_interactive_terminal() -> bool:
  """Check if stdin/stdout are connected to an interactive terminal."""
  return hasattr(sys.stdin, "isatty") and sys.stdin.isatty() and hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _resolve_mode(requested: str) -> str:
  """Resolve the display mode.

  Args:
    requested: "auto", "tui", or "repl".

  Returns:
    "tui" or "repl".
  """
  if requested == "repl":
    return "repl"
  if requested == "tui":
    if not _textual_available():
      raise ImportError("Textual is required for TUI mode. Install it with: pip install definable[cli]")
    return "tui"
  # Auto mode: use TUI if available and terminal is interactive
  if _textual_available() and _is_interactive_terminal():
    return "tui"
  return "repl"


class CLIInterface(BaseInterface):
  """Interactive terminal interface for Definable agents.

  Supports two display modes:

  - **tui** — Full Textual TUI with streaming markdown, collapsible tool calls,
    block navigation, and real-time metrics. Requires ``textual``.
  - **repl** — Simple Rich-based REPL with event rendering. Works everywhere.

  The default mode is ``"auto"`` — uses TUI if textual is installed and the
  terminal is interactive, otherwise falls back to REPL.

  Example::

      from definable.agent import Agent
      from definable.agent.interface.cli import CLIInterface

      agent = Agent(model="openai/gpt-4o-mini", tools=[my_tool])

      # Auto-detect best mode
      agent.serve(CLIInterface())

      # Force TUI mode
      agent.serve(CLIInterface(mode="tui"))

      # Force REPL mode (for piped input, CI, etc.)
      agent.serve(CLIInterface(mode="repl"))
  """

  def __init__(
    self,
    *,
    # Mode selection
    mode: str = "auto",
    # CLI-specific
    prompt: str = ">>> ",
    show_banner: bool = True,
    show_metrics: bool = True,
    show_tool_args: bool = True,
    show_tool_results: bool = True,
    show_thinking: bool = True,
    show_timestamps: bool = False,
    max_content_display: int = 500,
    markdown_output: bool = True,
    command_prefix: str = "/",
    enable_completions: bool = True,
    user_id: str = "cli-user",
    tools_expand: str = "success",
    # Base config
    max_session_history: int = 50,
    session_ttl_seconds: int = 3600,
    max_concurrent_requests: int = 1,
    error_message: str = "Sorry, something went wrong. Please try again.",
    typing_indicator: bool = True,
    max_message_length: int = 100_000,
    rate_limit_messages_per_minute: int = 30,
    # BaseInterface params
    session_manager: Optional["SessionManager"] = None,
    hooks: Optional[List["InterfaceHook"]] = None,
    identity_resolver: Optional["IdentityResolver"] = None,
    auth: Optional[object] = None,
    # Deprecated
    config: Optional[CLIConfig] = None,
  ) -> None:
    if config is not None:
      warnings.warn(
        "Passing config= to CLIInterface is deprecated. Pass prompt, show_banner, and other params directly as keyword arguments.",
        DeprecationWarning,
        stacklevel=2,
      )
      resolved_config = config
    else:
      resolved_config = CLIConfig(
        mode=mode,
        prompt=prompt,
        show_banner=show_banner,
        show_metrics=show_metrics,
        show_tool_args=show_tool_args,
        show_tool_results=show_tool_results,
        show_thinking=show_thinking,
        show_timestamps=show_timestamps,
        max_content_display=max_content_display,
        markdown_output=markdown_output,
        command_prefix=command_prefix,
        enable_completions=enable_completions,
        user_id=user_id,
        tools_expand=tools_expand,
        max_session_history=max_session_history,
        session_ttl_seconds=session_ttl_seconds,
        max_concurrent_requests=max_concurrent_requests,
        error_message=error_message,
        typing_indicator=typing_indicator,
        max_message_length=max_message_length,
        rate_limit_messages_per_minute=rate_limit_messages_per_minute,
      )
    super().__init__(
      config=resolved_config,
      session_manager=session_manager,
      hooks=hooks,
      identity_resolver=identity_resolver,
      auth=auth,
    )
    self._cli_config: CLIConfig = self.config  # type: ignore[assignment]
    self._resolved_mode: Optional[str] = None

    # REPL mode components (initialized lazily in _start_receiver)
    self._output = OutputManager(config=self._cli_config)
    self._input: Optional[InputHandler] = None
    self._is_processing = False

    # Renderer registry — streaming renderer is special (tracked for flag)
    self._streaming_renderer = StreamingRenderer()
    self._renderer_registry = RendererRegistry()
    self._renderer_registry.add(self._streaming_renderer)
    self._renderer_registry.add(RunRenderer())
    self._renderer_registry.add(ModelCallRenderer())
    self._renderer_registry.add(ToolCallRenderer())
    self._renderer_registry.add(ReasoningRenderer())
    self._renderer_registry.add(KnowledgeRenderer())
    self._renderer_registry.add(MemoryRenderer())
    self._renderer_registry.add(DeepResearchRenderer())
    self._renderer_registry.add(GuardrailRenderer())
    self._renderer_registry.add(SubAgentRenderer())

    # Command registry
    self._command_registry = CommandRegistry()
    register_builtins(self._command_registry)

    # Event handler ref (for unsubscribe)
    self._event_handler: Optional[object] = None

    # TUI app ref (set in TUI mode)
    self._tui_app: Optional[object] = None

  @property
  def active_mode(self) -> str:
    """The resolved display mode ("tui" or "repl")."""
    if self._resolved_mode is None:
      self._resolved_mode = _resolve_mode(self._cli_config.mode)
    return self._resolved_mode

  # --- Extension API ---

  def add_command(self, command: BaseCommand) -> "CLIInterface":
    """Register a custom slash command.

    Args:
      command: Command instance implementing BaseCommand protocol.

    Returns:
      Self for method chaining.
    """
    self._command_registry.register(command)
    return self

  def add_renderer(self, renderer: BaseRenderer) -> "CLIInterface":
    """Register a custom event renderer (REPL mode only).

    Args:
      renderer: Renderer instance implementing BaseRenderer protocol.

    Returns:
      Self for method chaining.
    """
    self._renderer_registry.add(renderer)
    return self

  # --- BaseInterface abstract methods ---

  async def _start_receiver(self) -> None:
    """Start the interface in the resolved mode."""
    assert self.agent is not None

    if self.active_mode == "tui":
      # TUI mode: app is started in serve_forever(), not here.
      # _start_receiver is only used for REPL mode setup.
      log_info("[cli] TUI mode — receiver managed by Textual app")
      return

    # REPL mode: banner, event subscription, input loop
    self._output.print_banner(self.agent)

    # Subscribe to pipeline event stream
    pipeline = getattr(self.agent, "pipeline", None)
    if pipeline is not None:
      event_stream = getattr(pipeline, "event_stream", None)
      if event_stream is not None:
        self._event_handler = self._handle_event
        event_stream.subscribe(self._event_handler)

    # Build completer
    completer = None
    if self._cli_config.enable_completions:
      try:
        from definable.agent.interface.cli.completer import SlashCommandCompleter

        completer = SlashCommandCompleter(self._command_registry, prefix=self._cli_config.command_prefix)
      except ImportError:
        pass

    # Start input handler
    self._input = InputHandler(
      config=self._cli_config,
      on_input=self._on_user_input,
      on_eof=self._on_eof,
      completer=completer,
    )
    await self._input.start()

  async def _stop_receiver(self) -> None:
    """Stop the interface and unsubscribe from events."""
    # Stop REPL input
    if self._input is not None:
      await self._input.stop()
      self._input = None

    # Unsubscribe from event stream
    if self._event_handler is not None and self.agent is not None:
      pipeline = getattr(self.agent, "pipeline", None)
      if pipeline is not None:
        event_stream = getattr(pipeline, "event_stream", None)
        if event_stream is not None:
          event_stream.unsubscribe(self._event_handler)
      self._event_handler = None

  async def _convert_inbound(self, raw_message: Any) -> Optional[InterfaceMessage]:
    """Convert a raw dict with 'text' to InterfaceMessage."""
    text = raw_message.get("text") if isinstance(raw_message, dict) else str(raw_message)
    return InterfaceMessage(
      text=text,
      platform="cli",
      platform_user_id=self._cli_config.user_id,
      platform_chat_id="cli",
      platform_message_id=str(uuid4()),
    )

  async def _send_response(
    self,
    original_msg: InterfaceMessage,
    response: InterfaceResponse,
    raw_message: Any,
  ) -> None:
    """Send agent response to terminal.

    In TUI mode: response display is handled by widgets via events.
    In REPL mode: prints to Rich console, avoiding double-print.
    """
    if self.active_mode == "tui":
      # TUI mode: widgets handle display via event routing.
      # Nothing to do here — content was streamed via events.
      return

    # REPL mode: finish streaming line
    self._streaming_renderer.finish()

    # Skip if content was already streamed
    if response.content and self._streaming_renderer.streamed_run_id is not None:
      self._output.console.print()
      return

    # Print content that wasn't streamed
    if response.content:
      self._output.print_response(response.content)

  # --- REPL callbacks ---

  async def _on_user_input(self, text: str) -> None:
    """Handle a line of user input (REPL mode)."""
    stripped = text.strip()
    if not stripped:
      return

    prefix = self._cli_config.command_prefix
    if stripped.startswith(prefix) and len(stripped) > len(prefix):
      await self._handle_command(stripped)
      return

    self._is_processing = True
    try:
      await self.handle_platform_message({"text": stripped})
    except KeyboardInterrupt:
      self._output.console.print("\n[dim]Interrupted[/dim]")
    except Exception as e:
      log_error(f"[cli] Error processing message: {e}")
      self._output.console.print(f"[red]Error: {e}[/red]")
    finally:
      self._is_processing = False

  async def _on_eof(self) -> None:
    """Handle EOF (Ctrl+D) — stop the interface."""
    self._output.console.print("\n[dim]Goodbye.[/dim]")
    self._running = False

  # --- Command handling ---

  async def _handle_command(self, text: str) -> None:
    """Parse and execute a slash command."""
    prefix = self._cli_config.command_prefix
    without_prefix = text[len(prefix) :]
    parts = without_prefix.split(maxsplit=1)
    cmd_name = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    command = self._command_registry.get(cmd_name)
    if command is None:
      self._output.console.print(f"[red]Unknown command: {prefix}{cmd_name}[/red] (type /help for commands)")
      return

    assert self.agent is not None
    session = self.session_manager.get_or_create(
      platform="cli",
      user_id=self._cli_config.user_id,
      chat_id="cli",
    )
    context = CommandContext(
      agent=self.agent,
      session=session,
      output=self._output,
      interface=self,
    )

    try:
      await command.execute(args, context)
    except Exception as e:
      self._output.console.print(f"[red]Command error: {e}[/red]")

  # --- Event rendering (REPL mode) ---

  def _handle_event(self, event: "BaseRunOutputEvent") -> None:
    """Dispatch an event to the renderer registry (sync handler)."""
    self._renderer_registry.dispatch(event, self._output.console, self._cli_config)

  # --- Lifecycle overrides ---

  async def serve_forever(self) -> None:
    """Block until the interface is stopped.

    In TUI mode: runs the Textual app (which manages its own event loop).
    In REPL mode: standard polling loop with Ctrl+C handling.
    """
    if self.agent is None:
      raise ValueError("Interface has no agent bound. Call bind(agent) or pass agent= to constructor.")

    if self.active_mode == "tui":
      await self._serve_tui()
    else:
      await self._serve_repl()

  async def _serve_tui(self) -> None:
    """Run the Textual TUI app."""
    from definable.agent.interface.cli.tui.app import DefinableApp

    assert self.agent is not None

    # Initialize BaseInterface state
    self._request_semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
    self._running = True

    app = DefinableApp(agent=self.agent, interface=self)
    self._tui_app = app

    try:
      await app.run_async()
    except (asyncio.CancelledError, KeyboardInterrupt):
      pass
    finally:
      self._running = False
      self._tui_app = None
      with contextlib.suppress(Exception):
        await self.stop()

  async def _serve_repl(self) -> None:
    """Run the REPL mode."""
    if not self._running:
      await self.start()
    try:
      while self._running:
        await asyncio.sleep(0.1)
    except (asyncio.CancelledError, KeyboardInterrupt):
      pass
    finally:
      with contextlib.suppress(Exception):
        await self.stop()
