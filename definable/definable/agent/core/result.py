"""RunResult — terminal state of an agent run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
  from definable.model.message import Message


@dataclass(frozen=True)
class RunResult:
  """Final result returned from agent.arun().

  - `content`: the model's terminal text response (None if max_turns hit)
  - `parsed`: structured output if `output_schema` was passed, else None
  - `messages`: full message list at end of run (system, user, assistant, tool)
  - `turns`: how many model-call iterations occurred
  - `exit_reason`: why the loop terminated
  """

  content: str | None
  parsed: Any | None = None
  messages: list[Message] = field(default_factory=list)
  turns: int = 0
  exit_reason: Literal["natural", "max_turns", "error", "aborted"] = "natural"
