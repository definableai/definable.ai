"""CLI interface — interactive terminal REPL for Definable agents."""

from __future__ import annotations

import asyncio
import contextlib
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
from definable.utils.log import log_error

if TYPE_CHECKING:
  from definable.agent.interface.hooks import InterfaceHook
  from definable.agent.interface.identity import IdentityResolver
  from definable.agent.interface.session import SessionManager
  from definable.agent.run.base import BaseRunOutputEvent


class CLIInterface(BaseInterface):
  """Interactive terminal interface for Definable agents.

  Provides a rich REPL with real-time event visualization (thinking,
  memory, knowledge, tools) using the Rich library.

  Example::

      from definable.agent import Agent
      from definable.agent.interface.cli import CLIInterface

      agent = Agent(model="openai/gpt-4o-mini", tools=[my_tool])
      agent.serve(CLIInterface())

      # With custom settings:
      agent.serve(CLIInterface(show_thinking=False, markdown_output=False))
  """

  def __init__(
    self,
    *,
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
    """Register a custom event renderer.

    Args:
      renderer: Renderer instance implementing BaseRenderer protocol.

    Returns:
      Self for method chaining.
    """
    self._renderer_registry.add(renderer)
    return self

  # --- BaseInterface abstract methods ---

  async def _start_receiver(self) -> None:
    """Start the REPL: banner, event subscription, input loop."""
    assert self.agent is not None

    # Print startup banner
    self._output.print_banner(self.agent)

    # Subscribe to pipeline event stream (if pipeline exists)
    pipeline = getattr(self.agent, "pipeline", None)
    if pipeline is not None:
      event_stream = getattr(pipeline, "event_stream", None)
      if event_stream is not None:
        self._event_handler = self._handle_event
        event_stream.subscribe(self._event_handler)

    # Build completer (if enabled and prompt_toolkit is available)
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
    """Stop the REPL and unsubscribe from events."""
    # Stop input
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
    """Print agent response to terminal.

    Skips printing if content was already streamed via RunContentEvent.
    """
    # Finish streaming line (add newline if needed)
    self._streaming_renderer.finish()

    # Skip if content was already streamed for this run
    if response.content and self._streaming_renderer.streamed_run_id is not None:
      # Content was streamed — don't double-print
      self._output.console.print()  # Blank line after streamed content
      return

    # Print content that wasn't streamed
    if response.content:
      self._output.print_response(response.content)

  # --- REPL callbacks ---

  async def _on_user_input(self, text: str) -> None:
    """Handle a line of user input."""
    stripped = text.strip()
    if not stripped:
      return

    # Check for slash commands
    prefix = self._cli_config.command_prefix
    if stripped.startswith(prefix) and len(stripped) > len(prefix):
      await self._handle_command(stripped)
      return

    # Normal message — route through BaseInterface pipeline
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
    # Strip prefix and split into command + args
    without_prefix = text[len(prefix) :]
    parts = without_prefix.split(maxsplit=1)
    cmd_name = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    command = self._command_registry.get(cmd_name)
    if command is None:
      self._output.console.print(f"[red]Unknown command: {prefix}{cmd_name}[/red] (type /help for commands)")
      return

    # Build context
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

  # --- Event rendering ---

  def _handle_event(self, event: "BaseRunOutputEvent") -> None:
    """Dispatch an event to the renderer registry (sync handler)."""
    self._renderer_registry.dispatch(event, self._output.console, self._cli_config)

  # --- Lifecycle overrides ---

  async def serve_forever(self) -> None:
    """Block until the interface is stopped.

    Overrides base to handle Ctrl+C gracefully:
    first Ctrl+C during a run cancels the run,
    second Ctrl+C (or Ctrl+C when idle) exits.
    """
    if self.agent is None:
      raise ValueError("Interface has no agent bound. Call bind(agent) or pass agent= to constructor.")
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
