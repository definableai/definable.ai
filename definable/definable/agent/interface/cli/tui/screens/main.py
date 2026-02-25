"""Main screen — primary conversation UI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

from textual import on, work
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Header

from definable.agent.interface.cli.tui.messages import (
  AcceptSlashComplete,
  HideSlashComplete,
  KnowledgeUpdate,
  MemoryUpdate,
  ModelCallUpdate,
  NavigateSlashComplete,
  RunCompleted,
  RunError,
  RunStarted,
  ShowSlashComplete,
  SlashCommandRequested,
  StreamChunk,
  StreamComplete,
  ThinkingChunk,
  ThinkingCompleted,
  ThinkingStarted,
  ToolCallCompleted,
  ToolCallStarted,
  UserSubmitted,
)
from definable.agent.interface.cli.tui.widgets.conversation import Conversation
from definable.agent.interface.cli.tui.widgets.footer_bar import FooterBar
from definable.agent.interface.cli.tui.widgets.prompt import Prompt
from definable.agent.interface.cli.tui.widgets.search_bar import SearchBar
from definable.agent.interface.cli.tui.widgets.session_tabs import SessionTabs
from definable.agent.interface.cli.tui.widgets.slash_complete import SlashComplete
from definable.agent.interface.cli.tui.widgets.status_bar import StatusBar

if TYPE_CHECKING:
  from definable.agent.interface.cli.interface import CLIInterface


class MainScreen(Screen):
  """Primary conversation screen.

  Composes:
  - SessionTabs (hidden by default, shown with multiple sessions)
  - SearchBar (hidden by default, toggled with Ctrl+F)
  - Conversation (scrollable message blocks)
  - SlashComplete (filtered command popup, hidden by default)
  - Prompt (input textarea with animated spinner)
  - FooterBar (context-sensitive keybinding hints)
  - StatusBar (bottom metrics bar)
  """

  BINDINGS = [
    ("ctrl+l", "clear", "Clear"),
    ("ctrl+c", "cancel_or_quit", "Cancel/Quit"),
    ("ctrl+t", "new_session", "New session"),
    ("ctrl+f", "toggle_search", "Search"),
  ]

  DEFAULT_CSS = """
  MainScreen {
    layout: vertical;
  }
  """

  def __init__(
    self,
    interface: "CLIInterface",
    model_name: str = "",
    tools_expand: str = "success",
  ) -> None:
    super().__init__()
    self._interface = interface
    self._model_name = model_name
    self._tools_expand = tools_expand
    self._conversation: Conversation | None = None
    self._prompt: Prompt | None = None
    self._status_bar: StatusBar | None = None
    self._slash_complete: SlashComplete | None = None
    self._session_tabs: SessionTabs | None = None
    self._search_bar: SearchBar | None = None
    self._footer_bar: FooterBar | None = None
    self._is_running = False
    self._clear_pending = False  # double Ctrl+L confirmation
    # Search state
    self._search_matches: List[int] = []  # indices into conversation children
    self._search_index = 0
    self._ctrl_c_count = 0
    # Session management
    self._active_chat_id = "cli"
    self._session_names: Dict[str, str] = {"cli": "Session 1"}
    self._session_counter = 1

  def compose(self) -> ComposeResult:
    yield Header(show_clock=False)
    self._session_tabs = SessionTabs()
    yield self._session_tabs
    self._search_bar = SearchBar()
    yield self._search_bar
    self._conversation = Conversation(tools_expand=self._tools_expand)
    yield self._conversation
    self._slash_complete = SlashComplete()
    yield self._slash_complete
    self._prompt = Prompt(indicator=self._interface._cli_config.prompt)
    yield self._prompt
    self._footer_bar = FooterBar()
    yield self._footer_bar
    self._status_bar = StatusBar(model_name=self._model_name)
    yield self._status_bar

  async def on_mount(self) -> None:
    """Focus the prompt on mount, load commands, show welcome."""
    if self._prompt is not None:
      self._prompt.focus_input()
    # Load available commands into the completion popup
    if self._slash_complete is not None:
      self._slash_complete.set_commands(self._get_command_list())
    # Update session tabs
    self._update_session_tabs()
    # Update status bar with session name
    if self._status_bar is not None:
      self._status_bar.session_name = self._session_names.get(self._active_chat_id, "")
    # Show welcome message for empty conversations
    await self._show_welcome_if_empty()

  def _get_command_list(self) -> List[Tuple[str, str, List[str]]]:
    """Build the list of available commands for completion."""
    commands: List[Tuple[str, str, List[str]]] = []
    registry = getattr(self._interface, "_command_registry", None)
    if registry is not None:
      for cmd in registry.all_commands:
        commands.append((cmd.name, cmd.description, list(cmd.aliases)))
    # Add TUI-specific session commands
    commands.append(("sessions", "List active sessions", []))
    commands.append(("session", "Switch or create a session", ["s"]))
    return commands

  async def _show_welcome_if_empty(self) -> None:
    """Show welcome message if the current session has no messages."""
    session = self._get_current_session()
    has_messages = session is not None and session.messages and len(session.messages) > 0
    if not has_messages and self._conversation is not None:
      agent_name = ""
      if self._interface.agent is not None:
        agent_name = self._interface.agent.agent_name or ""
      greeting = f"**{agent_name}**" if agent_name else "**Definable AI**"
      welcome = f"Welcome to {greeting}.\nType a message to start, or /help for commands."
      await self._conversation.add_system_message(welcome)

  # --- Session management ---

  def _get_current_session(self):
    """Get the current InterfaceSession."""
    return self._interface.session_manager.get(
      platform="cli",
      user_id=self._interface._cli_config.user_id,
      chat_id=self._active_chat_id,
    )

  def _update_session_tabs(self) -> None:
    """Update the session tabs widget."""
    if self._session_tabs is None:
      return
    tabs = []
    for chat_id, name in self._session_names.items():
      tabs.append((chat_id, name))
    self._session_tabs.set_sessions(tabs, active_key=self._active_chat_id)

  async def _switch_session(self, chat_id: str) -> None:
    """Switch to a different session, rebuilding the conversation."""
    if chat_id == self._active_chat_id:
      return

    self._active_chat_id = chat_id

    # Rebuild conversation from session messages
    if self._conversation is not None:
      session = self._interface.session_manager.get(
        platform="cli",
        user_id=self._interface._cli_config.user_id,
        chat_id=chat_id,
      )
      if session is not None and session.messages:
        await self._conversation.rebuild_from_messages(session.messages)
      else:
        await self._conversation.clear_conversation()
        await self._show_welcome_if_empty()

    # Update tabs and status bar
    self._update_session_tabs()
    if self._status_bar is not None:
      self._status_bar.session_name = self._session_names.get(chat_id, "")

    self.notify(f"Switched to {self._session_names.get(chat_id, chat_id)}", timeout=2)

  async def _create_session(self, name: str = "") -> str:
    """Create a new session and switch to it."""
    self._session_counter += 1
    chat_id = f"cli-{self._session_counter}"
    if not name:
      name = f"Session {self._session_counter}"
    self._session_names[chat_id] = name
    await self._switch_session(chat_id)
    return chat_id

  # --- User input handling ---

  @on(UserSubmitted)
  async def handle_user_submit(self, event: UserSubmitted) -> None:
    """Handle user submitting a prompt."""
    self._clear_pending = False  # reset clear confirmation
    self._ctrl_c_count = 0  # reset quit confirmation
    if self._is_running:
      return  # ignore input while agent is running

    text = event.text
    if not text:
      return

    # Add user message to conversation
    if self._conversation is not None:
      await self._conversation.add_user_message(text)

    # Disable input and run agent
    self._set_running(True)
    self._run_agent(text)

  @on(SlashCommandRequested)
  async def handle_slash_command(self, event: SlashCommandRequested) -> None:
    """Handle slash command — execute natively in TUI when possible."""
    # Hide completion popup
    if self._slash_complete is not None:
      self._slash_complete.hide()

    cmd = event.command
    args = event.args

    # TUI-native commands
    if cmd in ("clear", "cls"):
      await self.action_clear()
    elif cmd in ("quit", "exit", "q"):
      self.app.exit()
    elif cmd in ("help", "h", "?"):
      await self._tui_help()
    elif cmd in ("info", "i"):
      await self._tui_info()
    elif cmd in ("tools", "t"):
      await self._tui_tools()
    elif cmd in ("model", "m"):
      await self._tui_model()
    elif cmd in ("history", "hist"):
      await self._tui_history()
    elif cmd in ("reset", "new"):
      await self._tui_reset()
    elif cmd == "export":
      await self._tui_export(args)
    elif cmd == "sessions":
      await self._tui_sessions()
    elif cmd in ("session", "s"):
      await self._tui_session(args)
    else:
      # Unknown — show error in conversation
      if self._conversation is not None:
        await self._conversation.add_system_message(f"Unknown command: /{cmd} (type /help for commands)")

  # --- Session tab handling ---

  @on(SessionTabs.TabSelected)
  async def handle_tab_selected(self, event: SessionTabs.TabSelected) -> None:
    """Switch session when a tab is clicked."""
    await self._switch_session(event.session_key)

  # --- Slash completion handling ---

  @on(ShowSlashComplete)
  async def handle_show_complete(self, event: ShowSlashComplete) -> None:
    """Show or update the slash completion popup."""
    if self._slash_complete is not None:
      if not self._slash_complete.is_shown:
        self._slash_complete.show(event.query)
      else:
        self._slash_complete.update_filter(event.query)

  @on(HideSlashComplete)
  async def handle_hide_complete(self, event: HideSlashComplete) -> None:
    """Hide the slash completion popup."""
    if self._slash_complete is not None:
      self._slash_complete.hide()

  @on(NavigateSlashComplete)
  async def handle_navigate_complete(self, event: NavigateSlashComplete) -> None:
    """Navigate the slash completion popup."""
    if self._slash_complete is not None and self._slash_complete.is_shown:
      if event.direction < 0:
        self._slash_complete.move_up()
      else:
        self._slash_complete.move_down()

  @on(AcceptSlashComplete)
  async def handle_accept_complete(self, event: AcceptSlashComplete) -> None:
    """Accept the currently highlighted slash completion."""
    if self._slash_complete is not None and self._slash_complete.is_shown:
      selected = self._slash_complete.selected_command
      if selected:
        # Set the prompt text to the selected command
        if self._prompt is not None:
          self._prompt.set_text(f"/{selected} ")
        self._slash_complete.hide()

  # --- Search handlers ---

  @on(SearchBar.SearchChanged)
  async def handle_search_changed(self, event: SearchBar.SearchChanged) -> None:
    """Re-run the search when the query changes."""
    self._perform_search(event.query)

  @on(SearchBar.SearchNavigate)
  async def handle_search_navigate(self, event: SearchBar.SearchNavigate) -> None:
    """Navigate to next/previous search match."""
    if not self._search_matches:
      return
    self._search_index = (self._search_index + event.direction) % len(self._search_matches)
    self._scroll_to_match()

  @on(SearchBar.SearchDismissed)
  async def handle_search_dismissed(self, event: SearchBar.SearchDismissed) -> None:
    """Close the search bar and clear highlights."""
    if self._search_bar is not None:
      self._search_bar.hide_search()
    self._clear_search_highlights()
    self._search_matches.clear()
    self._search_index = 0
    if self._prompt is not None:
      self._prompt.focus_input()
    if self._footer_bar is not None:
      self._footer_bar.mode = "running" if self._is_running else "idle"

  def _perform_search(self, query: str) -> None:
    """Search conversation blocks for matching text."""
    self._clear_search_highlights()
    self._search_matches.clear()
    self._search_index = 0

    if not query or self._conversation is None:
      if self._search_bar is not None:
        self._search_bar.set_match_info(0, 0)
      return

    query_lower = query.lower()
    for i, child in enumerate(self._conversation.children):
      text = self._extract_block_text(child)
      if text and query_lower in text.lower():
        self._search_matches.append(i)
        child.add_class("search-match")

    total = len(self._search_matches)
    if self._search_bar is not None:
      self._search_bar.set_match_info(1 if total > 0 else 0, total)

    if total > 0:
      self._scroll_to_match()

  def _scroll_to_match(self) -> None:
    """Scroll to the current search match and highlight it."""
    if not self._search_matches or self._conversation is None:
      return
    idx = self._search_matches[self._search_index]
    children = list(self._conversation.children)
    if idx < len(children):
      # Remove active highlight from all, add to current
      for i in self._search_matches:
        if i < len(children):
          children[i].remove_class("search-active")
      children[idx].add_class("search-active")
      children[idx].scroll_visible()

    if self._search_bar is not None:
      self._search_bar.set_match_info(self._search_index + 1, len(self._search_matches))

  def _clear_search_highlights(self) -> None:
    """Remove search highlight classes from all blocks."""
    if self._conversation is None:
      return
    for child in self._conversation.children:
      child.remove_class("search-match", "search-active")

  @staticmethod
  def _extract_block_text(widget: object) -> str:
    """Extract searchable plain text from a conversation block."""
    # UserMessage, SystemMessage, AgentResponse, ThinkingBlock all have _content or _text
    for attr in ("_text", "_content", "content"):
      val = getattr(widget, attr, None)
      if isinstance(val, str) and val:
        return val
    # ToolCallBlock: combine tool_name + arguments + result
    if hasattr(widget, "tool_name"):
      parts = [getattr(widget, "tool_name", "")]
      args = getattr(widget, "arguments", "")
      if args:
        parts.append(args)
      result = getattr(widget, "_result", "")
      if result:
        parts.append(result)
      error = getattr(widget, "_error", "")
      if error:
        parts.append(error)
      return " ".join(parts)
    return ""

  # --- TUI-native command implementations ---

  async def _tui_help(self) -> None:
    """Show help as a system message in the conversation."""
    lines = ["**Available Commands:**\n"]
    registry = getattr(self._interface, "_command_registry", None)
    if registry is not None:
      for cmd in registry.all_commands:
        aliases = ", ".join(f"/{a}" for a in cmd.aliases) if cmd.aliases else ""
        alias_part = f"  ({aliases})" if aliases else ""
        lines.append(f"  /{cmd.name}{alias_part} \u2014 {cmd.description}")

    # Session commands
    lines.append("  /sessions \u2014 List active sessions")
    lines.append("  /session new [name] \u2014 Create a new session")
    lines.append("  /session <name> \u2014 Switch to a session")

    lines.append("\n**Keyboard Shortcuts:**")
    lines.append("  Enter \u2014 Submit prompt")
    lines.append("  Shift+Enter \u2014 New line")
    lines.append("  Up/Down \u2014 Input history")
    lines.append("  Tab \u2014 Accept command completion")
    lines.append("  Ctrl+L \u2014 Clear conversation")
    lines.append("  Ctrl+T \u2014 New session")
    lines.append("  Ctrl+C \u2014 Cancel / Quit")
    lines.append("  Ctrl+Q \u2014 Quit")
    lines.append("  Alt+Up/Down \u2014 Navigate blocks")

    if self._conversation is not None:
      await self._conversation.add_system_message("\n".join(lines))

  async def _tui_info(self) -> None:
    """Show agent info as a system message."""
    agent = self._interface.agent
    if agent is None:
      return

    lines = ["**Agent Info:**\n"]
    lines.append(f"  Name: {agent.agent_name or 'unnamed'}")
    lines.append(f"  Model: {getattr(agent, '_model_id', None) or getattr(agent.model, 'id', '?')}")
    lines.append(f"  Provider: {getattr(agent.model, 'provider', '?')}")

    tool_count = len(agent.tools) if agent.tools else 0
    lines.append(f"  Tools: {tool_count}")

    features = []
    for feat, attr in [
      ("memory", "_memory"),
      ("knowledge", "_knowledge"),
      ("thinking", "_thinking"),
      ("tracing", "_tracing"),
      ("research", "_deep_research"),
      ("guardrails", "_guardrails"),
      ("sub-agents", "_sub_agents"),
    ]:
      if getattr(agent, attr, None):
        features.append(feat)
    lines.append(f"  Features: {', '.join(features) if features else 'none'}")

    if self._conversation is not None:
      await self._conversation.add_system_message("\n".join(lines))

  async def _tui_tools(self) -> None:
    """Show available tools as a system message."""
    agent = self._interface.agent
    if agent is None:
      return

    tools = agent.tools or []
    if not tools:
      if self._conversation is not None:
        await self._conversation.add_system_message("No tools configured.")
      return

    lines = ["**Available Tools:**\n"]
    for tool in tools:
      name = getattr(tool, "name", "?")
      desc = getattr(tool, "description", "") or ""
      if len(desc) > 80:
        desc = desc[:77] + "..."
      lines.append(f"  {name} \u2014 {desc}")

    if self._conversation is not None:
      await self._conversation.add_system_message("\n".join(lines))

  async def _tui_model(self) -> None:
    """Show model details as a system message."""
    agent = self._interface.agent
    if agent is None:
      return

    model = agent.model
    lines = ["**Model Details:**\n"]
    lines.append(f"  ID: {getattr(model, 'id', '?')}")
    lines.append(f"  Provider: {getattr(model, 'provider', '?')}")

    for attr in ("temperature", "max_tokens", "top_p"):
      val = getattr(model, attr, None)
      if val is not None:
        lines.append(f"  {attr.replace('_', ' ').title()}: {val}")

    if self._conversation is not None:
      await self._conversation.add_system_message("\n".join(lines))

  async def _tui_history(self) -> None:
    """Show conversation history as a system message."""
    agent = self._interface.agent
    if agent is None:
      return

    session = self._interface.session_manager.get_or_create(
      platform="cli",
      user_id=self._interface._cli_config.user_id,
      chat_id=self._active_chat_id,
    )
    messages = session.messages
    if not messages:
      if self._conversation is not None:
        await self._conversation.add_system_message("No messages in session.")
      return

    role_labels = {"system": "SYS", "user": "YOU", "assistant": "AI", "tool": "TOOL"}
    lines = [f"**Conversation History** ({len(messages)} messages):\n"]
    for msg in messages:
      role = msg.role or "?"
      label = role_labels.get(role, role.upper())
      content = str(msg.content or "")
      if len(content) > 200:
        content = content[:197] + "..."
      lines.append(f"  [{label}] {content}")

    if self._conversation is not None:
      await self._conversation.add_system_message("\n".join(lines))

  async def _tui_reset(self) -> None:
    """Reset the session and clear conversation."""
    agent = self._interface.agent
    if agent is None:
      return

    session = self._interface.session_manager.get_or_create(
      platform="cli",
      user_id=self._interface._cli_config.user_id,
      chat_id=self._active_chat_id,
    )
    if session.messages is not None:
      session.messages.clear()
    session.last_run_output = None

    if self._conversation is not None:
      await self._conversation.clear_conversation()

    self.notify("Session reset", timeout=2)

  async def _tui_export(self, args: str) -> None:
    """Export conversation history to JSON."""
    import json

    agent = self._interface.agent
    if agent is None:
      return

    session = self._interface.session_manager.get_or_create(
      platform="cli",
      user_id=self._interface._cli_config.user_id,
      chat_id=self._active_chat_id,
    )
    messages = session.messages
    if not messages:
      if self._conversation is not None:
        await self._conversation.add_system_message("No messages to export.")
      return

    path = args.strip() or "chat_history.json"
    try:
      data = [m.to_dict() for m in messages]
      with open(path, "w") as f:
        json.dump(data, f, indent=2)
      if self._conversation is not None:
        await self._conversation.add_system_message(f"Exported {len(messages)} messages to {path}")
    except Exception as e:
      if self._conversation is not None:
        await self._conversation.add_system_message(f"Export failed: {e}")

  async def _tui_sessions(self) -> None:
    """List all active sessions."""
    lines = ["**Active Sessions:**\n"]
    for chat_id, name in self._session_names.items():
      marker = " \u25c0" if chat_id == self._active_chat_id else ""
      session = self._interface.session_manager.get(
        platform="cli",
        user_id=self._interface._cli_config.user_id,
        chat_id=chat_id,
      )
      msg_count = len(session.messages) if session and session.messages else 0
      lines.append(f"  {name} ({msg_count} messages){marker}")
    lines.append("\n  Use /session new [name] to create, /session <name> to switch.")
    if self._conversation is not None:
      await self._conversation.add_system_message("\n".join(lines))

  async def _tui_session(self, args: str) -> None:
    """Switch to or create a session."""
    args = args.strip()
    if not args:
      await self._tui_sessions()
      return

    # /session new [name]
    if args.startswith("new"):
      name = args[3:].strip()
      await self._create_session(name)
      return

    # /session <name> — find by name
    for chat_id, name in self._session_names.items():
      if name.lower() == args.lower() or chat_id == args:
        await self._switch_session(chat_id)
        return

    # Not found — offer to create
    if self._conversation is not None:
      await self._conversation.add_system_message(f"Session '{args}' not found. Use /session new {args} to create it.")

  # --- Agent execution ---

  @work(exclusive=True, thread=False)
  async def _run_agent(self, text: str) -> None:
    """Run the agent in a background worker."""
    try:
      # Use the active chat_id for the agent call
      await self._interface.handle_platform_message({"text": text, "chat_id": self._active_chat_id})
    except Exception as e:
      self.app.post_message(RunError(run_id="", error=str(e)))
    finally:
      self._set_running(False)

  def _set_running(self, running: bool) -> None:
    """Update UI state for running/idle."""
    self._is_running = running
    self._ctrl_c_count = 0
    if self._prompt is not None:
      self._prompt.set_running(running)
    if self._status_bar is not None:
      if running:
        self._status_bar.set_running()
      else:
        self._status_bar.set_ready()
    if self._footer_bar is not None:
      self._footer_bar.mode = "running" if running else "idle"

  # --- Pipeline event handlers ---

  @on(RunStarted)
  async def handle_run_started(self, event: RunStarted) -> None:
    """New agent run started — prepare response area."""
    if self._conversation is not None:
      await self._conversation.start_response(run_id=event.run_id)
    if self._status_bar is not None:
      self._status_bar.set_running()

  @on(StreamChunk)
  async def handle_stream_chunk(self, event: StreamChunk) -> None:
    """Streaming content chunk from agent."""
    if self._conversation is not None:
      await self._conversation.append_to_response(event.text)

  @on(StreamComplete)
  async def handle_stream_complete(self, event: StreamComplete) -> None:
    """Agent finished streaming content."""
    if self._conversation is not None:
      await self._conversation.finish_response()

  @on(RunCompleted)
  async def handle_run_completed(self, event: RunCompleted) -> None:
    """Agent run completed — show final metrics."""
    if self._conversation is not None:
      # If content wasn't streamed (non-streaming arun), display it now
      if event.content and self._conversation._current_response is None:
        await self._conversation.start_response(run_id=event.run_id)
        await self._conversation.append_to_response(event.content)
      await self._conversation.finish_response()

    if self._status_bar is not None:
      # Use pipeline-reported totals if available, otherwise keep accumulated
      if event.total_tokens > 0:
        self._status_bar.total_tokens = event.total_tokens
      self._status_bar.ttft_ms = event.time_to_first_token
      self._status_bar.total_time_ms = event.total_time
      self._status_bar.set_ready()

    self._set_running(False)
    if self._prompt is not None:
      self._prompt.focus_input()

  @on(RunError)
  async def handle_run_error(self, event: RunError) -> None:
    """Agent run errored."""
    if self._conversation is not None:
      await self._conversation.finish_response()

    if self._status_bar is not None:
      self._status_bar.set_error()

    self._set_running(False)
    if self._prompt is not None:
      self._prompt.focus_input()
    self.notify(f"Error: {event.error}", severity="error", timeout=5)

  # --- Thinking events ---

  @on(ThinkingStarted)
  async def handle_thinking_started(self, event: ThinkingStarted) -> None:
    if self._conversation is not None:
      await self._conversation.start_thinking()
    if self._status_bar is not None:
      self._status_bar.status = "Thinking"

  @on(ThinkingChunk)
  async def handle_thinking_chunk(self, event: ThinkingChunk) -> None:
    if self._conversation is not None:
      await self._conversation.append_to_thinking(event.text)

  @on(ThinkingCompleted)
  async def handle_thinking_completed(self, event: ThinkingCompleted) -> None:
    if self._conversation is not None:
      await self._conversation.finish_thinking()
    if self._status_bar is not None:
      self._status_bar.status = "Running"

  # --- Tool call events ---

  @on(ToolCallStarted)
  async def handle_tool_started(self, event: ToolCallStarted) -> None:
    if self._conversation is not None:
      await self._conversation.add_tool_call(
        tool_name=event.tool_name,
        arguments=event.arguments,
        call_id=event.call_id,
      )
    if self._status_bar is not None:
      self._status_bar.phase = f"Tool: {event.tool_name}"

  @on(ToolCallCompleted)
  async def handle_tool_completed(self, event: ToolCallCompleted) -> None:
    if self._conversation is not None:
      self._conversation.complete_tool_call(
        call_id=event.call_id,
        result=event.result,
        error=event.error,
        duration_ms=event.duration_ms,
      )
    if self._status_bar is not None:
      self._status_bar.phase = ""

  # --- Model call events ---

  @on(ModelCallUpdate)
  async def handle_model_call(self, event: ModelCallUpdate) -> None:
    if self._status_bar is not None:
      self._status_bar.turn = event.turn
      # Accumulate token counts from each model call
      if event.input_tokens > 0 or event.output_tokens > 0:
        self._status_bar.add_turn_tokens(event.input_tokens, event.output_tokens)

  # --- Knowledge & memory events ---

  @on(KnowledgeUpdate)
  async def handle_knowledge(self, event: KnowledgeUpdate) -> None:
    if self._status_bar is not None:
      if event.status == "searching":
        self._status_bar.phase = "Searching knowledge\u2026"
      elif event.status == "complete":
        count = event.doc_count
        ms = event.duration_ms
        self._status_bar.phase = f"Found {count} docs ({ms:.0f}ms)" if ms else ""
      else:
        self._status_bar.phase = ""

  @on(MemoryUpdate)
  async def handle_memory(self, event: MemoryUpdate) -> None:
    if self._status_bar is not None:
      if event.status == "recalling":
        self._status_bar.phase = "Recalling memory\u2026"
      elif event.status == "recalled":
        count = event.entry_count
        self._status_bar.phase = f"Recalled {count} entries" if count else ""
      elif event.status == "updating":
        self._status_bar.phase = "Updating memory\u2026"
      elif event.status == "updated":
        self._status_bar.phase = ""
      else:
        self._status_bar.phase = ""

  # --- Actions ---

  async def action_clear(self) -> None:
    """Clear the conversation (double Ctrl+L to confirm)."""
    if self._clear_pending:
      self._clear_pending = False
      if self._conversation is not None:
        await self._conversation.clear_conversation()
      self.notify("Conversation cleared", timeout=2)
    else:
      self._clear_pending = True
      self.notify("Press Ctrl+L again to confirm clear", timeout=3)

  async def action_new_session(self) -> None:
    """Create a new session via Ctrl+T."""
    await self._create_session()

  def action_toggle_search(self) -> None:
    """Toggle the conversation search bar (Ctrl+F)."""
    if self._search_bar is None:
      return
    if self._search_bar.is_active:
      self._search_bar.hide_search()
      self._clear_search_highlights()
      self._search_matches.clear()
      if self._prompt is not None:
        self._prompt.focus_input()
      if self._footer_bar is not None:
        self._footer_bar.mode = "running" if self._is_running else "idle"
    else:
      self._search_bar.show_search()
      if self._footer_bar is not None:
        self._footer_bar.mode = "searching"

  def action_cancel_or_quit(self) -> None:
    """Always require double Ctrl+C to quit."""
    self._ctrl_c_count += 1
    if self._ctrl_c_count >= 2:
      self.app.exit()
    elif self._is_running:
      self.notify("Cancelling\u2026 press Ctrl+C again to force quit", severity="warning", timeout=3)
    else:
      self.notify("Press Ctrl+C again to quit", timeout=3)

  async def _show_help(self) -> None:
    """Show help — delegates to _tui_help for inline display."""
    await self._tui_help()
