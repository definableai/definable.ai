"""The agent harness — read top to bottom.

Pseudocode:

    while turn < max_turns:
      emit TurnStarted + snapshot
      response = await llm.ainvoke(...) or stream chunks
      emit ModelResponded
      if no tool_calls:
        emit RunCompleted; return
      dispatch tools in parallel; append tool messages
      loop

Implementation lands in Phase 7.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from definable.agent.core.events import EventBus
  from definable.agent.core.result import RunResult
  from definable.agent.core.tools import ToolRegistry
  from definable.model.base import Model
  from definable.model.message import Message


async def run(
  *,
  llm: Model,
  messages: list[Message],
  tools: ToolRegistry,
  events: EventBus,
  memory: Any | None = None,
  stream: bool = False,
  max_turns: int = 50,
  output_schema: Any | None = None,
  run_id: str,
) -> RunResult:
  """Drive the agent loop until natural completion or max_turns.

  Emits all observability events through `events`. Returns the terminal
  RunResult once the model produces a final answer (no tool calls) or
  max_turns is exceeded.

  `memory` is FileMemory | None — typed loosely until Phase 11 lands the class.

  Phase 2: signature only. Implementation in Phase 7.
  """
  raise NotImplementedError("Phase 7")
