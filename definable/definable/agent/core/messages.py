"""Message assembly — system prompt + memory index + skill descriptions + user input.

No conversation history is loaded. Events are the source of truth for run
state — callers can persist them externally if a long-term log is wanted.
"""

from __future__ import annotations

from typing import Any, Sequence

from definable.model.message import Message


def build_messages(
  *,
  instructions: str | None,
  memory_index: str | None,
  skill_descriptions: Sequence[str] | None,
  user_input: str,
  media: Any | None = None,  # noqa: ARG001  # accepted for forward-compat; wired in Phase 7
) -> list[Message]:
  """Assemble the message list passed to the model.

  System message layout (sections appear only when their input is provided)::

      <instructions>

      # Available Memory
      <memory_index>

      # Available Skills
      <skill_descriptions joined by newlines>

  If `instructions`, `memory_index`, and `skill_descriptions` are all empty,
  no system message is produced. The user message is always present.
  """
  messages: list[Message] = []
  system = _build_system(instructions, memory_index, skill_descriptions)
  if system:
    messages.append(Message(role="system", content=system))
  messages.append(Message(role="user", content=user_input))
  return messages


def _build_system(
  instructions: str | None,
  memory_index: str | None,
  skill_descriptions: Sequence[str] | None,
) -> str:
  parts: list[str] = []
  if instructions:
    parts.append(instructions.strip())
  if memory_index:
    parts.append(f"# Available Memory\n{memory_index.strip()}")
  if skill_descriptions:
    joined = "\n".join(s.strip() for s in skill_descriptions if s and s.strip())
    if joined:
      parts.append(f"# Available Skills\n{joined}")
  return "\n\n".join(parts)
