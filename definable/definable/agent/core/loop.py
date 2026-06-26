"""The agent harness — read top to bottom.

::

    emit AgentBegin
    while turn < max_turns:
      before_model hooks
      response = model call           (emits reasoning/content step events)
      after_model hooks
      append assistant message
      if no tool_calls:
        emit AgentEnd; return
      dispatch tools                  (before/after_tool hooks + tool step events)
      append tool messages
    emit AgentEnd (max_turns)

Hooks may mutate context or raise AbortRun / SkipTool — see hooks.py.
Events are observe-only — see events.py.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Sequence

from definable.agent.core.events import (
  AgentBegin,
  AgentEnd,
  AgentError,
  EventBus,
  StepBegin,
  StepDelta,
  StepEnd,
  ToolCall,
)
from definable.agent.core.hooks import AbortRun, Hook, ModelHookContext
from definable.agent.core.result import RunResult
from definable.agent.core.tools import ToolRegistry
from definable.model.message import Message
from definable.model.response import ModelResponse, ToolExecution

if TYPE_CHECKING:
  from definable.model.base import Model
  from definable.model.metrics import Metrics


async def run(
  *,
  llm: Model,
  messages: list[Message],
  tools: ToolRegistry,
  events: EventBus,
  hooks: Sequence[Hook] = (),
  memory: Any | None = None,  # noqa: ARG001  # accepted for API stability; not used by the loop
  stream: bool = False,
  max_turns: int = 50,
  output_schema: Any | None = None,
  run_id: str,
) -> RunResult:
  """Drive the agent loop until natural completion or max_turns.

  Mutates `messages` in place — the caller can inspect the full transcript
  on the returned RunResult.messages.
  """
  events.emit(AgentBegin(run_id=run_id, timestamp=time.time()))
  tool_dicts = _build_tool_dicts(tools)
  agg_usage: dict[str, int] = {}
  turn = 0

  while turn < max_turns:
    turn += 1

    # ---- before_model ----
    mctx = ModelHookContext(run_id=run_id, turn=turn, messages=messages, tools=tool_dicts)
    try:
      for h in hooks:
        await h.before_model(mctx)
    except AbortRun:
      return _abort(events, run_id, messages, turn, agg_usage)

    try:
      response = await _call_model(
        llm=llm,
        messages=mctx.messages,
        tool_dicts=mctx.tools,
        output_schema=output_schema,
        stream=stream,
        events=events,
        run_id=run_id,
        turn=turn,
      )
    except Exception as e:
      events.emit(AgentError(run_id=run_id, timestamp=time.time(), error=str(e) or e.__class__.__name__, turns=turn))
      return RunResult(content=None, messages=messages, turns=turn, exit_reason="error")

    # ---- after_model ----
    mctx.response = response
    try:
      for h in hooks:
        await h.after_model(mctx)
    except AbortRun:
      return _abort(events, run_id, messages, turn, agg_usage)
    response = mctx.response or response

    _accumulate(agg_usage, response.response_usage)

    assistant_message = Message(role="assistant")
    if response.content is not None:
      assistant_message.content = response.content
    # Attach raw tool_calls so downstream tool messages reference a valid prior call.
    if response.tool_calls:
      assistant_message.tool_calls = list(response.tool_calls)
    messages.append(assistant_message)

    tool_calls = _extract_tool_calls(response.tool_executions or [], response.tool_calls or [])

    if not tool_calls:
      content = _content_str(response.content)
      events.emit(AgentEnd(run_id=run_id, timestamp=time.time(), content=content, turns=turn, usage=agg_usage or None))
      return RunResult(content=content, parsed=response.parsed, messages=messages, turns=turn)

    results = await tools.dispatch(tool_calls, events=events, run_id=run_id, turn=turn, hooks=hooks)
    for r in results:
      messages.append(
        Message(
          role="tool",
          tool_call_id=r.call.id,
          tool_name=r.call.name,
          content=str(r.output) if r.success else f"Error: {r.error}",
        )
      )
    if any(r.aborted for r in results):
      return _abort(events, run_id, messages, turn, agg_usage)

  events.emit(AgentEnd(run_id=run_id, timestamp=time.time(), content=None, turns=turn, usage=agg_usage or None, exit_reason="max_turns"))
  return RunResult(content=None, messages=messages, turns=turn, exit_reason="max_turns")


# ---- helpers -------------------------------------------------------------


def _abort(events: EventBus, run_id: str, messages: list[Message], turn: int, usage: dict[str, int]) -> RunResult:
  """Terminate the run because a hook raised AbortRun."""
  events.emit(AgentEnd(run_id=run_id, timestamp=time.time(), content=None, turns=turn, usage=usage or None, exit_reason="aborted"))
  return RunResult(content=None, messages=messages, turns=turn, exit_reason="aborted")


async def _call_model(
  *,
  llm: Model,
  messages: list[Message],
  tool_dicts: list[dict[str, Any]] | None,
  output_schema: Any | None,
  stream: bool,
  events: EventBus,
  run_id: str,
  turn: int,
) -> ModelResponse:
  """Either ainvoke or ainvoke_stream — emits reasoning/content step events,
  returns a unified ModelResponse."""
  cid = f"{run_id}:{turn}:content"
  rid = f"{run_id}:{turn}:reasoning"

  if not stream:
    response = await llm.ainvoke(messages=messages, tools=tool_dicts, response_format=output_schema)
    reasoning = getattr(response, "reasoning_content", None)
    if reasoning:
      events.emit(StepBegin(run_id=run_id, timestamp=time.time(), id=rid, type="reasoning", turn=turn))
      events.emit(StepEnd(run_id=run_id, timestamp=time.time(), id=rid, type="reasoning", data=reasoning))
    events.emit(StepBegin(run_id=run_id, timestamp=time.time(), id=cid, type="content", turn=turn))
    events.emit(
      StepEnd(
        run_id=run_id,
        timestamp=time.time(),
        id=cid,
        type="content",
        data=_content_str(response.content),
        usage=_usage_dict(response.response_usage),
      )
    )
    return response

  content_parts: list[str] = []
  reasoning_parts: list[str] = []
  tool_executions: list[ToolExecution] = []
  raw_tool_calls: list[dict[str, Any]] = []
  usage: Metrics | None = None
  parsed: Any | None = None
  content_open = False
  reasoning_open = False

  async for chunk in llm.ainvoke_stream(messages=messages, tools=tool_dicts, response_format=output_schema):
    reasoning = getattr(chunk, "reasoning_content", None)
    if reasoning:
      if not reasoning_open:
        events.emit(StepBegin(run_id=run_id, timestamp=time.time(), id=rid, type="reasoning", turn=turn))
        reasoning_open = True
      reasoning_parts.append(reasoning)
      events.emit(StepDelta(run_id=run_id, timestamp=time.time(), id=rid, type="reasoning", data=reasoning))
    if chunk.content:
      if reasoning_open:  # content marks the end of the reasoning step
        events.emit(StepEnd(run_id=run_id, timestamp=time.time(), id=rid, type="reasoning", data="".join(reasoning_parts)))
        reasoning_open = False
      if not content_open:
        events.emit(StepBegin(run_id=run_id, timestamp=time.time(), id=cid, type="content", turn=turn))
        content_open = True
      frag = _content_str(chunk.content) or ""
      content_parts.append(frag)
      events.emit(StepDelta(run_id=run_id, timestamp=time.time(), id=cid, type="content", data=frag))
    # Streamed tool calls are raw provider fragments — collect, assemble once below.
    if chunk.tool_calls:
      raw_tool_calls.extend(chunk.tool_calls)
    if chunk.tool_executions:
      tool_executions.extend(chunk.tool_executions)
    # Usage arrives on a late chunk; keep the latest (never sum — providers re-report).
    if chunk.response_usage is not None:
      usage = chunk.response_usage
    if chunk.parsed is not None:
      parsed = chunk.parsed

  if reasoning_open:
    events.emit(StepEnd(run_id=run_id, timestamp=time.time(), id=rid, type="reasoning", data="".join(reasoning_parts)))
  # The content step is the per-model-call boundary — emit it even on a
  # tool-only turn (empty content) so it carries the call's usage.
  if not content_open:
    events.emit(StepBegin(run_id=run_id, timestamp=time.time(), id=cid, type="content", turn=turn))
  events.emit(StepEnd(run_id=run_id, timestamp=time.time(), id=cid, type="content", data="".join(content_parts) or None, usage=_usage_dict(usage)))

  return ModelResponse(
    content="".join(content_parts) or None,
    reasoning_content="".join(reasoning_parts) or None,
    tool_calls=llm.parse_tool_calls(raw_tool_calls) if raw_tool_calls else [],
    tool_executions=tool_executions,
    response_usage=usage,
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


def _usage_dict(metrics: Metrics | None) -> dict[str, int] | None:
  """Project provider-reported usage into a flat dict for events.

  Returns None when the provider did not surface usage so consumers can
  distinguish "zero tokens" from "unknown".
  """
  if metrics is None:
    return None
  out: dict[str, int] = {}
  for key in ("input_tokens", "output_tokens", "total_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens"):
    value = getattr(metrics, key, 0)
    if value:
      out[key] = int(value)
  return out or None


def _accumulate(total: dict[str, int], metrics: Metrics | None) -> None:
  """Sum one model call's usage into the run total (sum across CALLS, not deltas)."""
  per_call = _usage_dict(metrics)
  if not per_call:
    return
  for key, value in per_call.items():
    total[key] = total.get(key, 0) + value
