"""Auto-build a one-line-per-file index when no INDEX.md is curated."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.agent.memory.file import FileMemory


def build_index(memory: FileMemory, *, max_line: int = 120) -> str:
  """Return a markdown bullet list — one line per memory file.

  Each line is the file's first non-empty line, trimmed to `max_line`
  characters. INDEX.md itself is skipped if present.
  """
  lines: list[str] = []
  for name in memory.names():
    if name == "INDEX":
      continue
    text = memory.read(name)
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    summary = first[:max_line]
    lines.append(f"- {name}: {summary}" if summary else f"- {name}")
  return "\n".join(lines) if lines else "(no memories)"
