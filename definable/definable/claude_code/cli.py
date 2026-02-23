"""Interactive CLI for ClaudeCodeAgent — faithful Claude Code CLI experience.

Provides a terminal interface that mirrors the Claude Code CLI visual language:
streaming content, thinking blocks, human-readable tool calls, HITL approval
panels, metrics, and a bottom status bar via prompt_toolkit.

Usage::

    from definable.claude_code import ClaudeCodeAgent, run_cli

    agent = ClaudeCodeAgent(
        model="claude-sonnet-4-6",
        instructions="Senior Python developer.",
        confirm_tools=["Bash", "Write", "Edit"],
    )
    run_cli(agent)

    # Or with customization:
    from definable.claude_code import ClaudeCodeCLI

    cli = ClaudeCodeCLI(agent=agent, show_thinking=False)
    await cli.run()
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Set

from definable.agent.events import (
  KnowledgeRetrievalCompletedEvent,
  KnowledgeRetrievalStartedEvent,
  MemoryRecallCompletedEvent,
  MemoryRecallStartedEvent,
  MemoryUpdateStartedEvent,
  ReasoningContentDeltaEvent,
  RunCompletedEvent,
  RunContentEvent,
  RunErrorEvent,
  RunPausedEvent,
  RunStartedEvent,
  ToolCallCompletedEvent,
  ToolCallStartedEvent,
)

if TYPE_CHECKING:
  from definable.agent.events import RunOutputEvent
  from definable.claude_code.agent import ClaudeCodeAgent


# ---------------------------------------------------------------------------
# Tool display helpers
# ---------------------------------------------------------------------------

_TOOL_VERBS = {
  "Read": "Reading",
  "Write": "Writing",
  "Edit": "Editing",
  "Bash": "Running",
  "Glob": "Searching",
  "Grep": "Searching",
  "WebFetch": "Fetching",
  "WebSearch": "Searching",
  "Task": "Spawning",
  "TodoWrite": "Updating tasks",
  "TaskCreate": "Creating task",
  "TaskUpdate": "Updating task",
  "TaskList": "Listing tasks",
  "TaskGet": "Getting task",
}

_TOOL_PRIMARY_ARG = {
  "Read": "file_path",
  "Write": "file_path",
  "Edit": "file_path",
  "Bash": "command",
  "Glob": "pattern",
  "Grep": "pattern",
  "WebFetch": "url",
  "WebSearch": "query",
  "TaskCreate": "subject",
  "TaskUpdate": "taskId",
  "TaskGet": "taskId",
}


def _truncate(text: str, max_len: int) -> str:
  """Truncate text with ellipsis if it exceeds max_len."""
  if len(text) <= max_len:
    return text
  return text[: max_len - 3] + "..."


def _format_tool_header(tool_name: str, tool_args: Optional[dict] = None) -> str:
  """Build a human-readable tool header.

  Examples:
    _format_tool_header("Read", {"file_path": "/src/auth.py"})
      -> "Reading /src/auth.py"
    _format_tool_header("Bash", {"command": "npm test"})
      -> "Running `npm test`"
    _format_tool_header("mcp__definable__search", {"query": "auth"})
      -> "mcp:definable:search"
  """
  # Strip mcp__server__tool prefix
  display_name = tool_name
  if tool_name.startswith("mcp__"):
    parts = tool_name.split("__")
    display_name = ":".join(parts[1:]) if len(parts) > 2 else tool_name[5:]

  verb = _TOOL_VERBS.get(tool_name)
  primary_key = _TOOL_PRIMARY_ARG.get(tool_name)

  if verb and primary_key and tool_args and primary_key in tool_args:
    primary_val = str(tool_args[primary_key])
    if tool_name == "Bash":
      return f"{verb} `{primary_val}`"
    return f"{verb} {primary_val}"

  if verb and (not tool_args or not primary_key):
    return verb

  # Unknown tool — show name(key=value, ...) or just name
  if tool_args:
    pairs = ", ".join(f"{k}={v}" for k, v in list(tool_args.items())[:3])
    return f"{display_name}({pairs})"
  return display_name


# ---------------------------------------------------------------------------
# Rich tool renderers — specialized display for structured tool calls
# ---------------------------------------------------------------------------

_STATUS_ICONS = {
  "completed": "[green]\u2713[/green]",
  "in_progress": "[yellow]\u25c6[/yellow]",
  "pending": "[dim]\u25cb[/dim]",
  "deleted": "[red]\u2717[/red]",
}

_TASK_TOOLS = {"TodoWrite", "TaskCreate", "TaskUpdate"}

_MAX_PANEL_TASKS = 12


@dataclass
class _TrackedTask:
  """A task tracked by the sticky panel."""

  id: str
  subject: str
  status: str = "pending"
  active_form: str = ""


def _format_tool_result(content: Any, max_lines: int = 8, max_chars: int = 500) -> str:
  """Format tool result output with line truncation.

  If >10 lines: first 5 + '... (N more lines)' + last 2.
  Truncate total chars to max_chars.
  """
  if content is None:
    return ""
  text = str(content)
  if not text.strip():
    return ""

  lines = text.splitlines()
  if len(lines) > 10:
    visible = lines[:5] + [f"\u2026 ({len(lines) - 7} more lines)"] + lines[-2:]
    text = "\n".join(visible)
  elif len(lines) > max_lines:
    text = "\n".join(lines[:max_lines])

  return _truncate(text, max_chars)


def _format_token_count(tokens: int) -> str:
  """Format token count: 2500 -> '2.5k', 150 -> '150'."""
  if tokens >= 1000:
    return f"{tokens / 1000:.1f}k"
  return str(tokens)


def _format_cost(value: float) -> str:
  """Format cost with smart precision."""
  if value == 0:
    return "0.00"
  if value >= 0.01:
    return f"{value:.4f}"
  import math

  return f"{value:.{max(4, 4 - int(math.log10(abs(value))))}f}"


# ---------------------------------------------------------------------------
# Completers
# ---------------------------------------------------------------------------

_SLASH_COMMANDS = [
  ("help", "Show available commands"),
  ("info", "Show agent settings"),
  ("status", "Show session statistics"),
  ("tools", "Show registered tools"),
  ("model", "Show model configuration"),
  ("session", "Show session details"),
  ("history", "Show recent prompts"),
  ("clear", "Clear the terminal"),
  ("quit", "Exit the CLI"),
]


class _SlashCommandCompleter:
  """Completer for slash commands."""

  def get_completions(self, document: Any, complete_event: Any) -> Any:
    from prompt_toolkit.completion import Completion

    text = document.text_before_cursor
    if not text.startswith("/"):
      return

    word = text[1:]
    for cmd, desc in _SLASH_COMMANDS:
      if cmd.startswith(word):
        yield Completion(cmd, start_position=-len(word), display=cmd, display_meta=desc)


class _SmartPathCompleter:
  """Path completer that works anywhere in the text, not just at the start."""

  def __init__(self) -> None:
    from prompt_toolkit.completion import PathCompleter

    self._path_completer = PathCompleter(expanduser=True)

  def get_completions(self, document: Any, complete_event: Any) -> Any:
    from prompt_toolkit.completion import Completion
    from prompt_toolkit.document import Document as PTDocument

    text = document.text_before_cursor
    # Find path-like tokens: ./ ../ / ~/
    match = re.search(r"(?:^|[\s])([.~/][\S]*?)$", text)
    if not match:
      return

    path_start = match.group(1)
    fake_doc = PTDocument(path_start, len(path_start))

    for completion in self._path_completer.get_completions(fake_doc, complete_event):
      yield Completion(
        completion.text,
        start_position=completion.start_position,
        display=completion.display,
        display_meta=completion.display_meta,
      )


# ---------------------------------------------------------------------------
# History file persistence
# ---------------------------------------------------------------------------


def _get_history_dir() -> Path:
  """Get ~/.definable/ directory, creating it if needed."""
  d = Path.home() / ".definable"
  d.mkdir(parents=True, exist_ok=True)
  return d


def _save_to_history_file(command: str, history_file: Path, max_entries: int = 5000) -> None:
  """Append a command to the persistent history file for manual review."""
  try:
    entries: list[tuple[str, str]] = []
    if history_file.exists():
      lines = history_file.read_text(encoding="utf-8").splitlines()
      for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
          entries.append((lines[i], lines[i + 1]))

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entries.append((f"# {timestamp}", f"+{command}"))
    entries = entries[-max_entries:]

    history_file.write_text("\n".join(f"{ts}\n{cmd}" for ts, cmd in entries) + "\n", encoding="utf-8")
  except Exception:
    pass


# ---------------------------------------------------------------------------
# ClaudeCodeCLI
# ---------------------------------------------------------------------------


@dataclass
class ClaudeCodeCLI:
  """Interactive terminal REPL for ClaudeCodeAgent.

  Drives ``arun_stream()`` and ``continue_run_stream()`` in a loop,
  rendering events in the Claude Code CLI visual style.

  Args:
    agent: The ClaudeCodeAgent instance to drive.
    show_banner: Whether to print a startup banner.
    show_thinking: Whether to render reasoning/thinking blocks.
    show_tool_args: Whether to show full tool arguments as secondary detail.
    show_tool_results: Whether to show tool results.
    show_metrics: Whether to show token/cost/duration after each run.
    max_content_display: Max characters for truncated displays.
  """

  agent: "ClaudeCodeAgent"
  show_banner: bool = True
  show_thinking: bool = True
  show_tool_args: bool = True
  show_tool_results: bool = True
  show_metrics: bool = True
  max_content_display: int = 500

  # Internal
  _session: Any = field(default=None, repr=False)
  _console: Any = field(default=None, repr=False)
  _running: bool = field(default=False, repr=False)
  _streamed_any_content: bool = field(default=False, repr=False)

  # Session tracking
  _total_runs: int = field(default=0, repr=False)
  _total_tokens: int = field(default=0, repr=False)
  _total_cost: float = field(default=0.0, repr=False)
  _total_duration: float = field(default=0.0, repr=False)
  _prompts_history: List[str] = field(default_factory=list, repr=False)

  # HITL session grants — tools auto-approved for the session
  _session_grants: Set[str] = field(default_factory=set, repr=False)

  # History file path
  _history_file: Optional[Path] = field(default=None, repr=False)

  # Live spinner for "agent is working" state
  _spinner: Any = field(default=None, repr=False)
  _spinner_label: str = field(default="", repr=False)

  # Task panel state
  _tracked_tasks: List[_TrackedTask] = field(default_factory=list, repr=False)
  _task_counter: int = field(default=0, repr=False)
  _panel_line_count: int = field(default=0, repr=False)

  def __post_init__(self) -> None:
    from rich.console import Console

    self._console = Console(highlight=False)
    self._history_file = _get_history_dir() / "cli_history.txt"

  # ---------------------------------------------------------------------------
  # Main REPL
  # ---------------------------------------------------------------------------

  async def run(self) -> None:
    """Main REPL loop. Blocks until the user exits."""
    self._running = True

    if self.show_banner:
      self._print_banner()
      self._console.rule(style="dim")

    while self._running:
      try:
        line = await self._read_input()
      except EOFError:
        self._print_session_summary()
        self._console.print("\n[dim]Goodbye.[/dim]")
        break
      except KeyboardInterrupt:
        self._console.print("\n[yellow]Use /quit to exit.[/yellow]")
        continue

      if line is None:
        self._print_session_summary()
        self._console.print("\n[dim]Goodbye.[/dim]")
        break

      stripped = line.strip()
      if not stripped:
        continue

      # Persist to history file
      if self._history_file:
        _save_to_history_file(stripped, self._history_file)

      if stripped.startswith("/"):
        should_exit = self._handle_command(stripped)
        if should_exit:
          break
        continue

      self._prompts_history.append(stripped)
      await self._run_prompt(stripped)

  # ---------------------------------------------------------------------------
  # Input — prompt_toolkit with bottom toolbar, completers, styled prompt
  # ---------------------------------------------------------------------------

  async def _read_input(self) -> Optional[str]:
    """Read a line of input asynchronously.

    Uses prompt_toolkit with a bottom status bar, tab completion for paths
    and slash commands, and a styled "You:" prompt.
    """
    try:
      from prompt_toolkit import PromptSession
      from prompt_toolkit.completion import merge_completers
      from prompt_toolkit.formatted_text import FormattedText, HTML
      from prompt_toolkit.history import InMemoryHistory

      if self._session is None:
        # Build completers
        completers: list[Any] = []
        try:
          cmd_completer = _SlashCommandCompleter()
          path_completer = _SmartPathCompleter()
          # merge_completers expects proper Completer subclasses;
          # wrap with a shim if needed
          from prompt_toolkit.completion import Completer as BaseCompleter

          class _Wrap(BaseCompleter):
            def __init__(self, inner: Any) -> None:
              self._inner = inner

            def get_completions(self, document: Any, complete_event: Any) -> Any:
              return self._inner.get_completions(document, complete_event)

          completers = [_Wrap(cmd_completer), _Wrap(path_completer)]
        except Exception:
          pass

        merged = merge_completers(completers) if completers else None

        def _toolbar() -> HTML:
          model = getattr(self.agent, "model", "?")
          mode = "confirm" if getattr(self.agent, "confirm_tools", None) else "auto"
          runs = self._total_runs
          tokens = _format_token_count(self._total_tokens) if self._total_tokens else "0"
          task_info = ""
          if self._tracked_tasks:
            done = sum(1 for t in self._tracked_tasks if t.status == "completed")
            total = len(self._tracked_tasks)
            task_info = f" \u00b7 tasks {done}/{total}"
          return HTML(
            f'<style fg="ansired"><b>\u25b6\u25b6</b> {mode}</style>'
            f'<style fg="ansibrightblack"> \u00b7 {model} \u00b7 {runs} runs \u00b7 {tokens} tokens{task_info} \u00b7 /help</style>'
          )

        self._session = PromptSession(
          history=InMemoryHistory(),
          bottom_toolbar=_toolbar,
          completer=merged,
          complete_while_typing=False,
        )

      prompt_text = FormattedText([("ansibrightcyan bold", "You:"), ("", " ")])
      return await self._session.prompt_async(prompt_text)
    except ImportError:
      loop = asyncio.get_running_loop()
      try:
        return await loop.run_in_executor(None, lambda: input("You: "))
      except EOFError:
        return None

  # ---------------------------------------------------------------------------
  # Prompt execution (streaming + HITL)
  # ---------------------------------------------------------------------------

  async def _run_prompt(self, prompt: str) -> None:
    """Execute a prompt through the agent with streaming + HITL."""
    self._streamed_any_content = False
    self._total_runs += 1
    self._start_spinner("Preparing")
    try:
      async for event in self.agent.arun_stream(prompt):
        self._render_event(event)

        if isinstance(event, RunPausedEvent):
          await self._handle_hitl_loop(event)
          return

    except KeyboardInterrupt:
      self._stop_spinner()
      self._console.print("\n[dim]Interrupted[/dim]")
    except Exception as exc:
      self._stop_spinner()
      self._console.print(f"\n[red]Error: {exc}[/red]")
    finally:
      self._stop_spinner()
      if self._streamed_any_content:
        print(flush=True)
      self._console.rule(style="dim")

  async def _handle_hitl_loop(self, paused_event: RunPausedEvent) -> None:
    """Handle HITL pause: prompt user, then continue_run_stream in a loop."""
    self._resolve_requirements(paused_event)

    try:
      async for event in self.agent.continue_run_stream(run_output=paused_event):
        self._render_event(event)

        if isinstance(event, RunPausedEvent):
          await self._handle_hitl_loop(event)
          return
    except KeyboardInterrupt:
      self._console.print("\n[dim]Interrupted[/dim]")
    except Exception as exc:
      self._console.print(f"\n[red]Error during resume: {exc}[/red]")

  def _resolve_requirements(self, paused: RunPausedEvent) -> None:
    """Prompt the user to resolve each requirement in a paused event."""
    if not paused.requirements:
      return

    for req in paused.requirements:
      # AskUserQuestion — interactive question rendering
      tool_name = req.tool_execution.tool_name if req.tool_execution else None
      if tool_name == "AskUserQuestion":
        self._prompt_ask_user_question(req)
        continue

      if req.needs_confirmation:
        # Check session grants for auto-approval
        if tool_name and tool_name in self._session_grants:
          req.confirm()
        else:
          self._prompt_confirmation(req)
      elif req.needs_user_input:
        self._prompt_user_input(req)
      elif req.needs_external_execution:
        self._prompt_external_execution(req)

  def _prompt_confirmation(self, req: Any) -> None:
    """Render a 3-choice tool approval panel."""
    from rich.panel import Panel

    tool_name = req.tool_execution.tool_name if req.tool_execution else "unknown"
    tool_args = req.tool_execution.tool_args if req.tool_execution else {}
    header = _format_tool_header(tool_name, tool_args)

    body = f"[bold]{header}[/bold]"
    if tool_args:
      try:
        args_json = json.dumps(tool_args, ensure_ascii=False, indent=2)
      except (TypeError, ValueError):
        args_json = str(tool_args)
      body += f"\n[dim]{args_json}[/dim]"

    self._console.print(Panel(body, title="Allow tool?", border_style="yellow", expand=False))

    self._console.print("  [bold]1.[/bold] Yes")
    self._console.print(f"  [bold]2.[/bold] Yes, and don't ask again for '[cyan]{tool_name}[/cyan]' this session")
    self._console.print("  [bold]3.[/bold] No")

    try:
      answer = input("  Choice [1/2/3]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
      answer = "3"

    if answer in ("1", "y", "yes"):
      req.confirm()
    elif answer == "2":
      req.confirm()
      self._session_grants.add(tool_name)
      self._console.print(f"  [dim]Auto-approving '{tool_name}' for the rest of this session.[/dim]")
    else:
      req.reject()

  def _prompt_user_input(self, req: Any) -> None:
    """Prompt the user for each input field with cyan border."""
    from rich.panel import Panel

    tool_name = req.tool_execution.tool_name if req.tool_execution else "unknown"
    fields = req.user_input_schema or []

    body = f"Tool: [bold]{tool_name}[/bold]"
    for f in fields:
      question = getattr(f, "description", None) or getattr(f, "name", "input")
      body += f"\n  {question}"

    self._console.print(Panel(body, title="Input Required", border_style="cyan", expand=False))

    for f in fields:
      label = getattr(f, "name", "value")
      try:
        value = input(f"  {label}: ").strip()
      except (EOFError, KeyboardInterrupt):
        value = ""
      f.value = value

    if req.tool_execution:
      req.tool_execution.answered = True

  def _prompt_external_execution(self, req: Any) -> None:
    """Prompt the user to provide external execution result."""
    from rich.panel import Panel

    tool_name = req.tool_execution.tool_name if req.tool_execution else "unknown"
    body = f"Tool: [bold]{tool_name}[/bold]\nExecute externally and paste the result."

    self._console.print(Panel(body, title="External Execution Required", border_style="magenta", expand=False))

    try:
      result_text = input("  Result: ").strip()
    except (EOFError, KeyboardInterrupt):
      result_text = ""

    req.set_external_execution_result(result_text)

  def _prompt_ask_user_question(self, req: Any) -> None:
    """Render AskUserQuestion interactively — Claude Code visual style."""
    from definable.claude_code.types import parse_ask_user_input

    raw_input = req.tool_execution.tool_args if req.tool_execution else {}
    parsed = parse_ask_user_input(raw_input)
    answers: dict[str, str] = {}
    questions = parsed.questions
    total = len(questions)

    for qi, q_item in enumerate(questions):
      # Progress bar for multi-question flows
      if total > 1:
        chips: list[str] = []
        for j, q in enumerate(questions):
          label = q.header or f"Q{j + 1}"
          if j < qi:
            chips.append(f"[green]\u25a0 {label}[/green]")
          elif j == qi:
            chips.append(f"[bold cyan]\u25b6 {label}[/bold cyan]")
          else:
            chips.append(f"[dim]\u25a1 {label}[/dim]")
        self._console.print(f"\n  {'  '.join(chips)}")

      # Question text
      self._console.print(f"\n  [bold]{q_item.question}[/bold]\n")

      # Options list — two-line format: number + bold label, indented dim description
      options = list(q_item.options)
      other_idx = len(options) + 1
      if options:
        for i, opt in enumerate(options, 1):
          self._console.print(f"    [bold]{i}.[/bold]  [bold]{opt.label}[/bold]")
          if opt.description:
            self._console.print(f"        [dim]{opt.description}[/dim]")
        self._console.print(f"    [bold]{other_idx}.[/bold]  [dim]Type something.[/dim]")
        self._console.print()

      # Hint line
      if q_item.multi_select:
        self._console.print("  [dim]Enter to select \u00b7 Comma-separate for multiple \u00b7 Type number or answer directly[/dim]")
      else:
        self._console.print("  [dim]Enter to select \u00b7 Type number or answer directly[/dim]")

      # Input loop with validation
      answer = self._read_ask_user_response(options, other_idx, q_item.multi_select)
      answers[q_item.question] = answer

    # Store answers on the requirement so continue_run can extract them
    if req.tool_execution:
      req.tool_execution.tool_args["_answers"] = answers
      req.tool_execution.answered = True

  def _read_ask_user_response(
    self,
    options: list[Any],
    other_idx: int,
    multi_select: bool,
  ) -> str:
    """Read and validate user input for an AskUserQuestion option list.

    Returns the answer string (option label(s) or free text).
    """
    while True:
      try:
        raw = input("  > ").strip()
      except (EOFError, KeyboardInterrupt):
        continue

      if not raw:
        continue

      # Try numeric parse
      if options:
        result = self._parse_ask_user_response(raw, options, other_idx, multi_select)
        if result is not None:
          return result

      # Free text fallback (when no options or non-numeric input)
      if not options:
        return raw

      # If input wasn't valid numeric, treat as free text
      if not raw.replace(",", "").replace(" ", "").isdigit():
        return raw

      # Invalid numeric — re-prompt
      max_n = other_idx
      self._console.print(f"  [red]Please enter a number between 1 and {max_n}[/red]")

  def _parse_ask_user_response(
    self,
    raw: str,
    options: list[Any],
    other_idx: int,
    multi_select: bool,
  ) -> Optional[str]:
    """Parse numeric or comma-separated input into option labels.

    Returns None if the input is invalid and should re-prompt.
    """
    # Check if input looks numeric
    cleaned = raw.replace(",", " ").split()
    nums: list[int] = []
    for part in cleaned:
      try:
        nums.append(int(part))
      except ValueError:
        return None  # Not numeric — caller treats as free text

    if not nums:
      return None

    if not multi_select and len(nums) > 1:
      self._console.print("  [red]Only one selection allowed[/red]")
      return None

    max_n = other_idx
    labels: list[str] = []
    for n in nums:
      if n < 1 or n > max_n:
        return None  # Out of range — will re-prompt
      if n == other_idx:
        # "Other" — prompt for free text
        try:
          text = input("  Your answer: ").strip()
        except (EOFError, KeyboardInterrupt):
          text = ""
        if text:
          labels.append(text)
      else:
        labels.append(options[n - 1].label)

    if not labels:
      return None

    return ", ".join(labels) if multi_select else labels[0]

  # ---------------------------------------------------------------------------
  # Spinner — animated "thinking" indicator while waiting for content
  # ---------------------------------------------------------------------------

  def _start_spinner(self, label: str = "Thinking") -> None:
    """Start an animated spinner to show the agent is working."""
    self._stop_spinner()  # ensure clean state
    from rich.spinner import Spinner
    from rich.live import Live

    self._spinner_label = label
    spinner = Spinner("dots", text=f"[dim cyan]{label}...[/dim cyan]", style="cyan")
    live = Live(spinner, console=self._console, refresh_per_second=12, transient=True)
    live.start()
    self._spinner = live

  def _update_spinner(self, label: str) -> None:
    """Update spinner label without restarting it."""
    if self._spinner is None:
      self._start_spinner(label)
      return
    from rich.spinner import Spinner

    self._spinner_label = label
    spinner = Spinner("dots", text=f"[dim cyan]{label}...[/dim cyan]", style="cyan")
    self._spinner.update(spinner)

  def _stop_spinner(self) -> None:
    """Stop the spinner if it's running."""
    if self._spinner is not None:
      self._spinner.stop()
      self._spinner = None
      self._spinner_label = ""

  # ---------------------------------------------------------------------------
  # Sticky task panel — ANSI cursor control
  # ---------------------------------------------------------------------------

  def _clear_task_panel(self) -> None:
    """Erase the task panel from the terminal using ANSI cursor control."""
    if self._panel_line_count == 0 or not sys.stdout.isatty():
      return
    # Move cursor up N lines then clear everything below
    sys.stdout.write(f"\033[{self._panel_line_count}A\033[J")
    sys.stdout.flush()
    self._panel_line_count = 0

  def _render_task_panel(self) -> None:
    """Render the sticky task panel at the current cursor position."""
    if not self._tracked_tasks or not sys.stdout.isatty():
      return

    from rich.console import Console

    buf = StringIO()
    panel_console = Console(file=buf, force_terminal=True, width=60)

    panel_console.rule("Tasks", style="dim")

    display_tasks = self._tracked_tasks[:_MAX_PANEL_TASKS]
    overflow = len(self._tracked_tasks) - _MAX_PANEL_TASKS

    for task in display_tasks:
      icon = _STATUS_ICONS.get(task.status, "[dim]\u25cb[/dim]")
      if task.status == "completed":
        panel_console.print(f"    {icon} [strike]{task.subject}[/strike]")
      else:
        panel_console.print(f"    {icon} {task.subject}")
      if task.active_form and task.status == "in_progress":
        panel_console.print(f"      [dim]{task.active_form}[/dim]")

    if overflow > 0:
      panel_console.print(f"    [dim]+{overflow} more[/dim]")

    output = buf.getvalue()
    sys.stdout.write(output)
    sys.stdout.flush()
    self._panel_line_count = output.count("\n")

  def _track_todo_write(self, args: dict) -> None:
    """Replace all tracked tasks from a TodoWrite payload."""
    todos = args.get("todos", [])
    self._tracked_tasks.clear()
    self._task_counter = 0
    for item in todos:
      self._task_counter += 1
      subject = item.get("content") or item.get("subject") or "?"
      status = item.get("status", "pending")
      active_form = item.get("activeForm", "")
      self._tracked_tasks.append(
        _TrackedTask(
          id=str(self._task_counter),
          subject=subject,
          status=status,
          active_form=active_form,
        )
      )

  def _track_task_create(self, args: dict) -> None:
    """Append a new task from a TaskCreate payload."""
    self._task_counter += 1
    subject = args.get("subject", "?")
    active_form = args.get("activeForm", "")
    self._tracked_tasks.append(
      _TrackedTask(
        id=str(self._task_counter),
        subject=subject,
        active_form=active_form,
      )
    )

  def _track_task_update(self, args: dict) -> None:
    """Update an existing task from a TaskUpdate payload."""
    task_id = args.get("taskId", "")
    for task in self._tracked_tasks:
      if task.id == task_id:
        if "status" in args:
          task.status = args["status"]
        if "subject" in args:
          task.subject = args["subject"]
        if "activeForm" in args:
          task.active_form = args["activeForm"]
        break

  def _track_task_tool(self, name: str, args: dict) -> None:
    """Route a task tool call to the appropriate tracker."""
    if name == "TodoWrite":
      self._track_todo_write(args)
    elif name == "TaskCreate":
      self._track_task_create(args)
    elif name == "TaskUpdate":
      self._track_task_update(args)

  # ---------------------------------------------------------------------------
  # Event rendering
  # ---------------------------------------------------------------------------

  def _render_event(self, event: "RunOutputEvent") -> None:
    """Dispatch a single event to the appropriate renderer."""
    if isinstance(event, RunStartedEvent):
      # Reset task panel for new run
      self._clear_task_panel()
      self._tracked_tasks.clear()
      self._task_counter = 0
      self._panel_line_count = 0
      self._update_spinner("Thinking")

    elif isinstance(event, KnowledgeRetrievalStartedEvent):
      self._update_spinner("Searching knowledge")

    elif isinstance(event, KnowledgeRetrievalCompletedEvent):
      n = getattr(event, "documents_found", 0)
      if n:
        self._stop_spinner()
        self._console.print(f"  [dim blue]Retrieved {n} document{'s' if n != 1 else ''}[/dim blue]")
      self._update_spinner("Thinking")

    elif isinstance(event, MemoryRecallStartedEvent):
      self._update_spinner("Recalling memory")

    elif isinstance(event, MemoryRecallCompletedEvent):
      n = getattr(event, "chunks_included", 0)
      if n:
        self._stop_spinner()
        self._console.print(f"  [dim yellow]Recalled {n} memory chunk{'s' if n != 1 else ''}[/dim yellow]")
      self._update_spinner("Thinking")

    elif isinstance(event, MemoryUpdateStartedEvent):
      self._update_spinner("Saving memory")

    elif isinstance(event, RunContentEvent):
      content = event.content
      if content and isinstance(content, str):
        if not self._streamed_any_content:
          self._stop_spinner()
          self._clear_task_panel()
          print(flush=True)  # visual spacing before first content
          self._streamed_any_content = True
        print(content, end="", flush=True)

    elif isinstance(event, ReasoningContentDeltaEvent):
      if self.show_thinking and event.reasoning_content:
        self._stop_spinner()
        self._clear_task_panel()
        if self._streamed_any_content:
          print(flush=True)  # newline before thinking interrupts content
        self._console.print(f"  [dim cyan]\u25c6 {event.reasoning_content}[/dim cyan]")

    elif isinstance(event, ToolCallStartedEvent):
      self._stop_spinner()
      self._clear_task_panel()
      if self._streamed_any_content:
        print(flush=True)  # newline before tool call interrupts content
      tool = event.tool
      if tool:
        name = tool.tool_name or "unknown"
        args = tool.tool_args

        # AskUserQuestion — skip rendering here; the interactive
        # prompt in _prompt_ask_user_question handles it after the pause.
        if name == "AskUserQuestion":
          return

        header = _format_tool_header(name, args)
        self._console.print(f"  [magenta]\u25cf {header}[/magenta]")

        # Task tools — track state and render panel
        if name in _TASK_TOOLS and args:
          self._track_task_tool(name, args)
          self._render_task_panel()
          self._start_spinner(_TOOL_VERBS.get(name, name))
          return

        # Show full JSON args as secondary detail when multiple args
        if self.show_tool_args and args:
          primary_key = _TOOL_PRIMARY_ARG.get(name)
          extra_args = {k: v for k, v in args.items() if k != primary_key} if primary_key else args
          if extra_args and len(args) > 1:
            try:
              detail = json.dumps(extra_args, ensure_ascii=False, separators=(",", ":"))
            except (TypeError, ValueError):
              detail = str(extra_args)
            detail = _truncate(detail, self.max_content_display)
            self._console.print(f"    [dim]{detail}[/dim]")

        # Show spinner while tool executes
        self._start_spinner(_TOOL_VERBS.get(name, name))

    elif isinstance(event, ToolCallCompletedEvent):
      self._stop_spinner()
      self._clear_task_panel()

      # AskUserQuestion result — already handled by the interactive prompt.
      tool = event.tool
      if tool and (tool.tool_name or "") == "AskUserQuestion":
        return

      if self.show_tool_results and event.content:
        result_str = _format_tool_result(event.content, max_chars=self.max_content_display)
        if result_str:
          for line in result_str.splitlines():
            self._console.print(f"    [dim]{line}[/dim]")
      self._render_task_panel()

    elif isinstance(event, RunCompletedEvent):
      self._stop_spinner()
      self._clear_task_panel()
      if self._streamed_any_content:
        print(flush=True)
        self._streamed_any_content = False
      if event.metrics:
        # Accumulate session stats
        self._total_tokens += getattr(event.metrics, "total_tokens", 0)
        cost = getattr(event.metrics, "cost", None)
        if cost is not None:
          self._total_cost += cost
        duration = getattr(event.metrics, "duration", None)
        if duration is not None:
          self._total_duration += duration
      if self.show_metrics and event.metrics:
        self._render_metrics(event.metrics)
      # Final persistent panel — render and reset line count so it won't be cleared
      self._render_task_panel()
      self._panel_line_count = 0

    elif isinstance(event, RunErrorEvent):
      from rich.panel import Panel

      self._stop_spinner()
      self._clear_task_panel()
      if self._streamed_any_content:
        print(flush=True)
        self._streamed_any_content = False
      error_text = event.content or "Unknown error"
      self._console.print(Panel(error_text, title="Error", border_style="red", expand=False))

    elif isinstance(event, RunPausedEvent):
      self._stop_spinner()
      self._clear_task_panel()
      if self._streamed_any_content:
        print(flush=True)
        self._streamed_any_content = False

  def _render_metrics(self, metrics: Any) -> None:
    """Render a compact metrics line: 2.5k tokens  ·  $0.0450  ·  15.0s"""
    parts: List[str] = []

    total = getattr(metrics, "total_tokens", 0)
    if total:
      parts.append(f"{_format_token_count(total)} tokens")

    cost = getattr(metrics, "cost", None)
    if cost is not None:
      parts.append(f"${cost:.4f}")

    duration = getattr(metrics, "duration", None)
    if duration is not None:
      parts.append(f"{duration:.1f}s")

    if parts:
      sep = " \u00b7 "
      self._console.print(f"\n  [dim green]{sep.join(parts)}[/dim green]")

  # ---------------------------------------------------------------------------
  # Slash commands
  # ---------------------------------------------------------------------------

  def _handle_command(self, text: str) -> bool:
    """Handle a slash command. Returns True if the REPL should exit."""
    parts = text.lstrip("/").split()
    cmd = parts[0].lower() if parts else ""

    if cmd in ("quit", "exit", "q"):
      self._print_session_summary()
      self._console.print("[dim]Goodbye.[/dim]")
      return True

    if cmd == "help":
      self._print_help()
      return False

    if cmd == "clear":
      self._console.clear()
      self._tracked_tasks.clear()
      self._panel_line_count = 0
      return False

    if cmd == "info":
      self._print_info()
      return False

    if cmd == "status":
      self._print_status()
      return False

    if cmd == "tools":
      self._print_tools()
      return False

    if cmd == "model":
      self._print_model()
      return False

    if cmd == "session":
      self._print_session()
      return False

    if cmd == "history":
      n = 10
      if len(parts) > 1:
        with __import__("contextlib").suppress(ValueError):
          n = int(parts[1])
      self._print_history(n)
      return False

    self._console.print(f"[red]Unknown command: /{cmd}[/red] (type /help for commands)")
    return False

  def _print_help(self) -> None:
    """Print categorized help."""
    self._console.print()
    self._console.print("  [bold yellow]Commands:[/bold yellow]")
    self._console.print("    [bold cyan]/help[/bold cyan]       Show this help")
    self._console.print("    [bold cyan]/info[/bold cyan]       Show agent settings")
    self._console.print("    [bold cyan]/status[/bold cyan]     Show session statistics")
    self._console.print("    [bold cyan]/tools[/bold cyan]      Show registered tools")
    self._console.print("    [bold cyan]/model[/bold cyan]      Show model configuration")
    self._console.print("    [bold cyan]/session[/bold cyan]    Show session details")
    self._console.print("    [bold cyan]/history[/bold cyan]    Show recent prompts")
    self._console.print("    [bold cyan]/clear[/bold cyan]      Clear the terminal")
    self._console.print("    [bold cyan]/quit[/bold cyan]       Exit the CLI")
    self._console.print()
    self._console.print("  [bold yellow]Tips:[/bold yellow]")
    self._console.print("    [dim]\u2022 UP/DOWN arrows to navigate history[/dim]")
    self._console.print("    [dim]\u2022 TAB for path and command completion[/dim]")
    self._console.print("    [dim]\u2022 Ctrl+C to interrupt current run[/dim]")
    self._console.print()

  def _print_info(self) -> None:
    """Print agent configuration as a key-value table."""
    from rich.table import Table

    agent = self.agent
    table = Table(show_header=False, box=None, padding=(0, 2), expand=False)
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")

    table.add_row("Model", str(agent.model))
    table.add_row("Name", agent.agent_name or "-")
    table.add_row("Session", agent.session_id or "-")

    features = []
    if agent.memory:
      features.append("memory")
    if agent.knowledge:
      features.append("knowledge")
    if agent.thinking_budget_tokens:
      features.append(f"thinking({agent.thinking_budget_tokens})")
    if agent.confirm_tools:
      features.append(f"hitl({','.join(agent.confirm_tools)})")
    table.add_row("Features", ", ".join(features) if features else "-")

    tool_count = 0
    if agent.tools:
      tool_count += len(agent.tools)
    if agent.toolkits:
      tool_count += len(agent.toolkits)
    table.add_row("Tools/Toolkits", str(tool_count) if tool_count else "-")

    if agent.max_turns:
      table.add_row("Max Turns", str(agent.max_turns))
    if agent.max_budget_usd:
      table.add_row("Max Budget", f"${agent.max_budget_usd}")

    self._console.print(table)

  def _print_status(self) -> None:
    """Print cumulative session statistics."""
    from rich.table import Table

    table = Table(show_header=False, box=None, padding=(0, 2), expand=False)
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")

    table.add_row("Runs", str(self._total_runs))
    table.add_row("Total tokens", f"{self._total_tokens:,}" if self._total_tokens else "0")
    table.add_row("Session cost", f"${_format_cost(self._total_cost)} (estimated)" if self._total_cost else "-")
    if self._total_duration:
      table.add_row("Total time", f"{self._total_duration:.1f}s")
    table.add_row("Model", str(self.agent.model))

    if self._session_grants:
      table.add_row("Auto-approved", ", ".join(sorted(self._session_grants)))

    self._console.print(table)

  def _print_tools(self) -> None:
    """List registered custom tools and MCP toolkits."""
    agent = self.agent
    has_any = False

    if agent.tools:
      has_any = True
      self._console.print("\n  [bold yellow]Custom Tools:[/bold yellow]")
      for t in agent.tools:
        name = getattr(t, "name", str(t))
        desc = getattr(t, "description", "") or ""
        desc_str = f" [dim]- {_truncate(desc, 60)}[/dim]" if desc else ""
        self._console.print(f"    [cyan]{name}[/cyan]{desc_str}")

    if agent.toolkits:
      has_any = True
      self._console.print("\n  [bold yellow]Toolkits:[/bold yellow]")
      for tk in agent.toolkits:
        name = getattr(tk, "name", type(tk).__name__)
        desc = getattr(tk, "description", "") or ""
        desc_str = f" [dim]- {_truncate(desc, 60)}[/dim]" if desc else ""
        self._console.print(f"    [cyan]{name}[/cyan]{desc_str}")

    if agent.skills:
      has_any = True
      self._console.print("\n  [bold yellow]Skills:[/bold yellow]")
      for sk in agent.skills:
        name = getattr(sk, "name", str(sk))
        desc = getattr(sk, "description", "") or ""
        desc_str = f" [dim]- {_truncate(desc, 60)}[/dim]" if desc else ""
        self._console.print(f"    [cyan]{name}[/cyan]{desc_str}")

    if not has_any:
      self._console.print("  [dim]No custom tools, toolkits, or skills registered.[/dim]")

    self._console.print()

  def _print_model(self) -> None:
    """Show model, permission mode, and thinking budget."""
    from rich.table import Table

    agent = self.agent
    table = Table(show_header=False, box=None, padding=(0, 2), expand=False)
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")

    table.add_row("Model", str(agent.model))
    table.add_row("Permission mode", getattr(agent, "permission_mode", "?"))

    if agent.thinking_budget_tokens:
      table.add_row("Thinking budget", f"{agent.thinking_budget_tokens:,} tokens")

    if agent.max_turns:
      table.add_row("Max turns", str(agent.max_turns))
    if agent.max_budget_usd:
      table.add_row("Max budget", f"${agent.max_budget_usd}")

    self._console.print(table)

  def _print_session(self) -> None:
    """Show session ID and conversation state."""
    from rich.table import Table

    agent = self.agent
    table = Table(show_header=False, box=None, padding=(0, 2), expand=False)
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")

    table.add_row("Session ID", agent.session_id or "(auto)")
    table.add_row("Continue conversation", str(getattr(agent, "continue_conversation", False)))
    table.add_row("Agent name", agent.agent_name or "-")
    table.add_row("Prompts this session", str(len(self._prompts_history)))

    self._console.print(table)

  def _print_history(self, n: int = 10) -> None:
    """Show last N prompts from this session."""
    if not self._prompts_history:
      self._console.print("  [dim]No prompts yet.[/dim]")
      return

    recent = self._prompts_history[-n:]
    self._console.print()
    for i, prompt in enumerate(recent, start=max(1, len(self._prompts_history) - n + 1)):
      self._console.print(f"  [dim]{i}.[/dim] {_truncate(prompt, 80)}")
    self._console.print()

  # ---------------------------------------------------------------------------
  # Session summary (on exit)
  # ---------------------------------------------------------------------------

  def _print_session_summary(self) -> None:
    """Print a compact session summary on exit."""
    if self._total_runs == 0:
      return

    self._console.print()
    self._console.rule("[bold]Session Summary[/bold]", style="dim")
    parts = [f"Runs: {self._total_runs}"]
    if self._total_tokens:
      parts.append(f"Tokens: {self._total_tokens:,}")
    if self._total_cost:
      parts.append(f"Cost: ${_format_cost(self._total_cost)} (estimated)")
    if self._total_duration:
      parts.append(f"Time: {self._total_duration:.1f}s")
    sep = " \u00b7 "
    self._console.print(f"  {sep.join(parts)}")
    self._console.rule(style="dim")

  # ---------------------------------------------------------------------------
  # Banner
  # ---------------------------------------------------------------------------

  def _print_banner(self) -> None:
    """Print a startup banner matching the Claude Code panel style."""
    from rich.panel import Panel
    from rich.table import Table

    agent = self.agent

    table = Table(show_header=False, box=None, padding=(0, 2), expand=False)
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")

    table.add_row("Model", str(agent.model))

    if agent.instructions:
      instr = _truncate(agent.instructions, 80)
      table.add_row("Instructions", instr)

    features = []
    if agent.memory:
      features.append("memory")
    if agent.knowledge:
      features.append("knowledge")
    if agent.thinking_budget_tokens:
      features.append("thinking")
    if features:
      table.add_row("Features", ", ".join(features))

    if agent.confirm_tools:
      table.add_row("HITL", ", ".join(agent.confirm_tools))

    # Tool/toolkit count
    tool_count = 0
    if agent.tools:
      tool_count += len(agent.tools)
    if agent.toolkits:
      tool_count += len(agent.toolkits)
    if tool_count:
      table.add_row("Tools", str(tool_count))

    table.add_row("", "[dim]Type /help for commands \u00b7 TAB to complete \u00b7 Ctrl+C to interrupt[/dim]")

    self._console.print(
      Panel(
        table,
        title="[bold]ClaudeCodeAgent[/bold]",
        expand=False,
      )
    )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def run_cli(agent: "ClaudeCodeAgent", **kwargs: Any) -> None:
  """Run the interactive CLI for a ClaudeCodeAgent. Blocks until exit.

  Args:
    agent: The ClaudeCodeAgent to drive.
    **kwargs: Passed to ClaudeCodeCLI constructor (show_thinking, etc.).
  """
  cli = ClaudeCodeCLI(agent=agent, **kwargs)
  asyncio.run(cli.run())
