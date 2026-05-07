"""The agent harness — read top to bottom.

::

    while turn < max_turns:
      emit TurnStarted + snapshot
      response = await llm.ainvoke(...) or stream chunks
      emit ModelResponded
      if no tool_calls:
        emit RunCompleted; return
      dispatch tools in parallel; append tool messages
      loop

"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from definable.agent.core.debug import TurnSnapshot
from definable.agent.core.events import (
  EventBus,
  ModelResponded,
  RunCompleted,
  RunErrored,
  StreamChunkEvent,
  ToolCall,
  TurnStarted,
)
from definable.agent.core.result import RunResult
from definable.agent.core.tools import ToolRegistry
from definable.model.message import Message
from definable.model.response import ModelResponse, ToolExecution

if TYPE_CHECKING:
  from definable.model.base import Model


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

  Mutates `messages` in place — the caller can inspect the full transcript
  on the returned RunResult.messages.
  """
  tool_dicts = _build_tool_dicts(tools)
  turn = 0

  while turn < max_turns:
    turn += 1

    events.emit(
      TurnStarted(
        run_id=run_id,
        timestamp=time.time(),
        snapshot=TurnSnapshot(
          turn=turn,
          context_tokens=_estimate_tokens(messages),
          messages=len(messages),
          available_tools=tools.names(),
          memory_files=_memory_list(memory),
        ),
      )
    )

    assistant_message = Message(role="assistant")
    try:
      response = await _call_model(
        llm=llm,
        messages=messages,
        assistant_message=assistant_message,
        tool_dicts=tool_dicts,
        output_schema=output_schema,
        stream=stream,
        events=events,
        run_id=run_id,
      )
    except Exception as e:
      events.emit(RunErrored(run_id=run_id, timestamp=time.time(), error=str(e) or e.__class__.__name__, turns=turn))
      return RunResult(content=None, messages=messages, turns=turn, exit_reason="error")

    if assistant_message.content is None and response.content is not None:
      assistant_message.content = response.content
    # Attach raw tool_calls so downstream tool messages reference a valid prior call.
    if response.tool_calls and not assistant_message.tool_calls:
      assistant_message.tool_calls = list(response.tool_calls)
    messages.append(assistant_message)

    tool_calls = _extract_tool_calls(response.tool_executions or [], response.tool_calls or [])
    events.emit(
      ModelResponded(
        run_id=run_id,
        timestamp=time.time(),
        content=_content_str(response.content),
        tool_calls=tool_calls,
      )
    )

    if not tool_calls:
      content = _content_str(response.content)
      events.emit(RunCompleted(run_id=run_id, timestamp=time.time(), content=content, turns=turn))
      return RunResult(content=content, parsed=response.parsed, messages=messages, turns=turn)

    results = await tools.dispatch_parallel(tool_calls, events=events, run_id=run_id)
    for r in results:
      messages.append(
        Message(
          role="tool",
          tool_call_id=r.call.id,
          tool_name=r.call.name,
          content=str(r.output) if r.success else f"Error: {r.error}",
        )
      )

  events.emit(RunCompleted(run_id=run_id, timestamp=time.time(), content=None, turns=turn, exit_reason="max_turns"))
  return RunResult(content=None, messages=messages, turns=turn, exit_reason="max_turns")


# ---- helpers -------------------------------------------------------------


async def _call_model(
  *,
  llm: Model,
  messages: list[Message],
  assistant_message: Message,
  tool_dicts: list[dict[str, Any]] | None,
  output_schema: Any | None,
  stream: bool,
  events: EventBus,
  run_id: str,
) -> ModelResponse:
  """Either ainvoke or ainvoke_stream — returns a unified ModelResponse."""
  if not stream:
    return await llm.ainvoke(
      messages=messages,
      assistant_message=assistant_message,
      tools=tool_dicts,
      response_format=output_schema,
    )

  content_parts: list[str] = []
  tool_executions: list[ToolExecution] = []
  parsed: Any | None = None

  async for chunk in llm.ainvoke_stream(
    messages=messages,
    assistant_message=assistant_message,
    tools=tool_dicts,
    response_format=output_schema,
  ):
    if chunk.content:
      content_parts.append(_content_str(chunk.content) or "")
      events.emit(StreamChunkEvent(run_id=run_id, timestamp=time.time(), kind="content", data=_content_str(chunk.content) or ""))
    reasoning = getattr(chunk, "reasoning_content", None)
    if reasoning:
      events.emit(StreamChunkEvent(run_id=run_id, timestamp=time.time(), kind="reasoning", data=reasoning))
    if chunk.tool_executions:
      tool_executions.extend(chunk.tool_executions)
    if chunk.parsed is not None:
      parsed = chunk.parsed

  return ModelResponse(
    content="".join(content_parts) or None,
    tool_executions=tool_executions,
    parsed=parsed,
  )


def _build_tool_dicts(tools: ToolRegistry) -> list[dict[str, Any]] | None:
  """Wrap each Function in the provider-required envelope.

  OpenAI / Anthropic / Mistral / etc all expect:
      {"type": "function", "function": {<JSON schema>}}
  """
  fns = tools.all()
  if not fns:
    return None
  return [{"type": "function", "function": fn.to_dict()} for fn in fns]


def _extract_tool_calls(executions: list[ToolExecution], raw_tool_calls: list[dict[str, Any]]) -> list[ToolCall]:
  """Convert provider tool-call data into harness-internal ToolCall.

  Prefers `ModelResponse.tool_executions` (already-parsed structured form);
  falls back to `ModelResponse.tool_calls` raw dicts (OpenAI/Anthropic
  wire format) if executions are empty.
  """
  calls: list[ToolCall] = []
  for te in executions:
    if te.tool_name is None:
      continue
    calls.append(ToolCall(id=te.tool_call_id or "", name=te.tool_name, args=dict(te.tool_args or {})))
  if calls:
    return calls
  # Fallback: parse the raw OpenAI-shaped tool_calls dicts.
  import json

  for tc in raw_tool_calls:
    fn = tc.get("function") or {}
    name = fn.get("name")
    if not name:
      continue
    raw_args = fn.get("arguments")
    args: dict[str, Any] = {}
    if isinstance(raw_args, str):
      try:
        parsed = json.loads(raw_args)
        if isinstance(parsed, dict):
          args = parsed
      except json.JSONDecodeError:
        args = {}
    elif isinstance(raw_args, dict):
      args = raw_args
    calls.append(ToolCall(id=tc.get("id", ""), name=name, args=args))
  return calls


def _content_str(content: Any | None) -> str | None:
  """Coerce a Message.content (str | list | None) into a str | None."""
  if content is None:
    return None
  if isinstance(content, str):
    return content
  return str(content)


def _estimate_tokens(messages: list[Message]) -> int:
  """Cheap context-size heuristic — ~4 chars per token across all message content."""
  total = 0
  for m in messages:
    c = m.content
    if isinstance(c, str):
      total += len(c)
    elif c is not None:
      total += len(str(c))
  return total // 4


def _memory_list(memory: Any | None) -> list[str]:
  if memory is None:
    return []
  fn = getattr(memory, "names", None)
  if callable(fn):
    result = fn()
    if isinstance(result, list):
      return [str(x) for x in result]
  return []
