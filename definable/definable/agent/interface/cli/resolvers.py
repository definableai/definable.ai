"""CLI resolvers for HITL permission prompts and agent questions.

Permission prompts use an arrow-key navigable menu (like Claude Code).
Question prompts use numbered options with arrow-key selection.
"""

from __future__ import annotations

import asyncio
import sys
import termios
import tty
from typing import List, Tuple

from rich.console import Console

from definable.agent.hitl.types import (
  Answer,
  PermissionDecision,
  PermissionRequest,
  PermissionResponse,
  Question,
)

_console = Console()


# ---------------------------------------------------------------------------
# Arrow-key selector
# ---------------------------------------------------------------------------


def _read_key() -> str:
  """Read a single keypress from stdin (raw mode). Returns key name."""
  fd = sys.stdin.fileno()
  old = termios.tcgetattr(fd)
  try:
    tty.setraw(fd)
    ch = sys.stdin.read(1)
    if ch == "\x1b":
      seq = sys.stdin.read(2)
      if seq == "[A":
        return "up"
      if seq == "[B":
        return "down"
      return "escape"
    if ch in ("\r", "\n"):
      return "enter"
    if ch == "\x03":
      return "ctrl-c"
    return ch
  finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _render_menu(options: List[Tuple[str, str]], selected: int, header: str) -> List[str]:
  """Build the menu lines. Each option is (label, description)."""
  lines: list[str] = []
  lines.append(f"  \033[1;33m{header}\033[0m")
  for i, (label, desc) in enumerate(options):
    cursor = "\033[36m❯\033[0m" if i == selected else " "
    if i == selected:
      line = f"  {cursor} \033[1m{label}\033[0m"
    else:
      line = f"  {cursor} \033[2m{label}\033[0m"
    if desc:
      line += f"  \033[2m{desc}\033[0m"
    lines.append(line)
  return lines


def _select_menu(options: List[Tuple[str, str]], header: str) -> int:
  """Interactive arrow-key menu. Returns selected index."""
  selected = 0
  total_lines = len(options) + 1  # header + options

  # Initial render
  lines = _render_menu(options, selected, header)
  sys.stdout.write("\n".join(lines) + "\n")
  sys.stdout.flush()

  while True:
    key = _read_key()

    if key == "up":
      selected = (selected - 1) % len(options)
    elif key == "down":
      selected = (selected + 1) % len(options)
    elif key == "enter":
      # Move cursor below the menu before returning
      return selected
    elif key == "ctrl-c":
      # Default to last option (deny)
      return len(options) - 1
    elif key.isdigit():
      idx = int(key) - 1
      if 0 <= idx < len(options):
        selected = idx
        return selected
    else:
      continue

    # Re-render: move cursor up, clear lines, redraw
    sys.stdout.write(f"\033[{total_lines}A")  # Move up
    for _ in range(total_lines):
      sys.stdout.write("\033[2K\n")  # Clear each line
    sys.stdout.write(f"\033[{total_lines}A")  # Move back up

    lines = _render_menu(options, selected, header)
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Permission resolver
# ---------------------------------------------------------------------------


def _truncate(text: str, max_len: int = 120) -> str:
  return text if len(text) <= max_len else text[:max_len] + "..."


async def cli_permission_resolver(request: PermissionRequest) -> PermissionResponse:
  """Arrow-key permission prompt in the terminal."""
  args_str = _truncate(str(request.tool_args)) if request.tool_args else "{}"

  _console.print()
  _console.print(f"  [magenta]{request.tool_name}[/magenta]({args_str})")

  options = [
    ("Allow", "allow this time"),
    ("Always allow", f"never ask again for {request.tool_name}"),
    ("Deny", "block this tool call"),
  ]

  idx = await asyncio.to_thread(_select_menu, options, "Do you want to proceed?")

  if idx == 0:
    return PermissionResponse(decision=PermissionDecision.allow_once)
  elif idx == 1:
    return PermissionResponse(decision=PermissionDecision.allow_always)
  else:
    return PermissionResponse(decision=PermissionDecision.deny, feedback=f"User denied {request.tool_name}")


# ---------------------------------------------------------------------------
# Question resolver
# ---------------------------------------------------------------------------


async def cli_question_resolver(questions: List[Question]) -> List[Answer]:
  """Arrow-key question prompt in the terminal."""
  answers: List[Answer] = []
  for q in questions:
    _console.print()
    if q.header:
      _console.print(f"  [bold cyan]{q.header}[/bold cyan]")

    if q.options:
      options = [(opt.label, opt.description or "") for opt in q.options]
      if q.allow_custom:
        options.append(("Custom answer...", "type your own"))

      idx = await asyncio.to_thread(_select_menu, options, q.text)

      if q.allow_custom and idx == len(q.options):
        _console.print("  [dim]Type your answer:[/dim]", end=" ")
        raw = (await asyncio.to_thread(input, "")).strip()
        answers.append(Answer(question_text=q.text, custom_text=raw))
      else:
        label = q.options[idx].label
        if q.allow_multiple:
          answers.append(Answer(question_text=q.text, selected=[label]))
        else:
          answers.append(Answer(question_text=q.text, selected=[label]))
    else:
      _console.print(f"  {q.text}")
      raw = (await asyncio.to_thread(input, "  > ")).strip()
      answers.append(Answer(question_text=q.text, custom_text=raw))

  return answers
