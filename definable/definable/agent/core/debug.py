"""TurnSnapshot — per-turn debug payload emitted via TurnStarted event.

Subscribers consume this for live debugging: context size, available tools,
memory state, and (after the model responds) what decision the model made.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TurnSnapshot:
  """Snapshot of harness state at the start of each loop iteration."""

  turn: int
  context_tokens: int
  messages: int
  available_tools: list[str] = field(default_factory=list)
  memory_files: list[str] = field(default_factory=list)
  last_decision: str | None = None
  reasoning_text: str | None = None
