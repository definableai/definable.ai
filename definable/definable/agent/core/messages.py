"""Message assembly — system prompt + memory index + skill descriptions + user input.

No conversation history is loaded. Events are the source of truth for run
state — callers can persist them externally if a long-term log is wanted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
  from definable.model.message import Message


def build_messages(
  *,
  instructions: str | None,
  memory_index: str | None,
  skill_descriptions: Sequence[str] | None,
  user_input: str,
  media: Any | None = None,
) -> list[Message]:
  """Assemble the message list passed to the model.

  System message layout:

      <instructions>
      [optional: # Available Memory <memory_index>]
      [optional: # Available Skills <skill_descriptions>]

  User message: `user_input` (+ media attachments).

  Phase 2: signature only. Implementation in Phase 6.
  """
  raise NotImplementedError("Phase 6")
