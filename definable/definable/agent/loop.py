"""Unified async-generator agentic loop.

Single implementation for streaming and non-streaming modes.
Yields RunOutputEvent instances throughout execution.
"""

import asyncio
from dataclasses import dataclass, field
from time import time
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Dict, Optional

from definable.agent.cancellation import CancellationToken
from definable.model.message import Message
from definable.model.response import ToolExecution
from definable.agent.events import (
  BaseRunOutputEvent,
  CompressionCompletedEvent,
  CompressionStartedEvent,
  ModelCallCompletedEvent,
  ModelCallStartedEvent,
  RunCompletedEvent,
  RunContentEvent,
  RunContext,
  RunErrorEvent,
  RunOutputEvent,
  RunPausedEvent,
  ToolCallCompletedEvent,
  ToolCallStartedEvent,
)
from definable.agent.run.requirement import RunRequirement
from definable.tool.function import Function
from definable.utils.log import log_debug, log_warning
from definable.utils.tools import get_function_call_for_tool_call

if TYPE_CHECKING:
  from definable.agent.compression.manager import CompressionManager
  from definable.agent.config import AgentConfig
  from definable.agent.guardrail.base import Guardrails
  from definable.model.base import Model
  from definable.model.metrics import Metrics
  from definable.model.response import ModelResponse


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
  """Result of a single tool execution within the loop."""

  tool_call_id: Optional[str] = None
  tool_name: str = ""
  result: Optional[str] = None
  error: Optional[str] = None
  should_stop: bool = False
  is_paused: bool = False
  requirement: Optional[RunRequirement] = None
  tool_execution: Optional[ToolExecution] = None
  events: list[BaseRunOutputEvent] = field(default_factory=list)


@dataclass
class ToolBatchResult:
  """Result of executing a batch of tool calls (parallel + sequential)."""

  results: list[ToolResult]
  events: list[BaseRunOutputEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AgentLoop
# ---------------------------------------------------------------------------


class AgentLoop:
  """Unified async-generator agentic loop.

  Both ``arun()`` (non-streaming) and ``arun_stream()`` (streaming) call
  ``AgentLoop.run()`` which yields ``RunOutputEvent`` instances.

  * ``arun()`` collects events and builds a ``RunOutput``.
  * ``arun_stream()`` yields events directly to the caller.
  """

  def __init__(
    self,
    *,
    model: "Model",
    tools: Dict[str, Function],
    messages: list[Message],
    context: RunContext,
    config: "AgentConfig",
    streaming: bool = False,
    cancellation_token: Optional[CancellationToken] = None,
    compression_manager: Optional["CompressionManager"] = None,
    guardrails: Optional["Guardrails"] = None,
    emit_fn: Callable[[BaseRunOutputEvent], None],
    agent_id: str,
    agent_name: str,
  ) -> None:
    self._model = model
    self._tools = tools
    self._messages = messages
    self._context = context
    self._config = config
    self._streaming = streaming
    self._cancellation_token = cancellation_token
    self._compression_manager = compression_manager
    self._guardrails = guardrails
    self._emit_fn = emit_fn
    self._agent_id = agent_id
    self._agent_name = agent_name

    # Precompute tool dicts for model API (OpenAI format)
    self._tools_dicts: Optional[list[dict]] = [{"type": "function", "function": t.to_dict()} for t in tools.values()] if tools else None

    # Accumulated state during the loop
    self._all_tool_executions: list[ToolExecution] = []
    self._turn: int = 0
    self._tool_retry_counts: Dict[str, int] = {}  # tool_call_id → retry count

  # ------------------------------------------------------------------
  # Public API
  # ------------------------------------------------------------------

  async def run(self) -> AsyncGenerator[RunOutputEvent, None]:
    """The unified loop. Yields events as they occur.

    When ``streaming=True``, yields ``RunContentEvent`` deltas during
    the model call. When ``streaming=False``, uses non-streaming model calls.
    Tool dispatch, HITL, and all other logic is shared.
    """
    tool_round = 0
    max_tool_rounds = self._config.max_tool_rounds
    final_content: Optional[str] = None
    final_parsed: Any = None
    total_metrics: Optional["Metrics"] = None

    try:
      while True:
        # 1. Cancellation check
        if self._cancellation_token:
          self._cancellation_token.raise_if_cancelled()

        # 2. Increment round, check max_tool_rounds
        tool_round += 1
        if tool_round > max_tool_rounds:
          log_warning(f"Agent loop hit max_tool_rounds={max_tool_rounds}. Forcing stop to prevent infinite tool-call loop.")
          content, metrics, parsed = await self._force_final_answer()
          final_content = content
          final_parsed = parsed
          if metrics is not None:
            total_metrics = metrics if total_metrics is None else total_metrics + metrics
          break

        # 3. Compression check
        if self._compression_manager is not None:
          if await self._compression_manager.ashould_compress(self._messages, self._tools_dicts, model=self._model):
            uncompressed_count = len([m for m in self._messages if m.role == "tool" and m.compressed_content is None])
            yield CompressionStartedEvent(
              run_id=self._context.run_id,
              session_id=self._context.session_id,
              agent_id=self._agent_id,
              agent_name=self._agent_name,
              tool_results_count=uncompressed_count,
            )
            compress_start = time()
            await self._compression_manager.acompress(self._messages)
            compress_duration = (time() - compress_start) * 1000
            stats = self._compression_manager.stats
            yield CompressionCompletedEvent(
              run_id=self._context.run_id,
              session_id=self._context.session_id,
              agent_id=self._agent_id,
              agent_name=self._agent_name,
              tool_results_compressed=stats.get("tool_results_compressed", 0),
              original_size=stats.get("original_size", 0),
              compressed_size=stats.get("compressed_size", 0),
              duration_ms=compress_duration,
            )

        # 4. Model call (streaming or non-streaming)
        started_evt = self._make_model_call_started()
        yield started_evt

        if self._streaming:
          # Streaming: yield RunContentEvent deltas inline
          content, tool_calls, metrics, parsed = "", [], None, None
          accumulated_content = ""
          accumulated_tool_calls: list[dict] = []
          accumulated_metrics: Optional["Metrics"] = None

          assistant_message = Message(role="assistant")
          async for chunk in self._model.ainvoke_stream(
            messages=self._messages,
            assistant_message=assistant_message,
            tools=self._tools_dicts,
            response_format=self._context.output_schema,
          ):
            if hasattr(chunk, "content") and chunk.content:
              accumulated_content += chunk.content
              yield RunContentEvent(
                run_id=self._context.run_id,
                session_id=self._context.session_id,
                agent_id=self._agent_id,
                agent_name=self._agent_name,
                content=chunk.content,
              )
            if hasattr(chunk, "tool_calls") and chunk.tool_calls:
              accumulated_tool_calls = _merge_tool_call_deltas(accumulated_tool_calls, chunk.tool_calls)
            if hasattr(chunk, "response_usage") and chunk.response_usage is not None:
              if accumulated_metrics is None:
                accumulated_metrics = chunk.response_usage
              else:
                accumulated_metrics = accumulated_metrics + chunk.response_usage
            # Capture parsed from final chunk if provider set it
            if hasattr(chunk, "parsed") and chunk.parsed is not None:
              parsed = chunk.parsed

          # Parse structured output from accumulated content if not already parsed from chunks
          if parsed is None and self._context.output_schema is not None and accumulated_content:
            import json

            from pydantic import BaseModel

            response_format = self._context.output_schema
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
              try:
                parsed_data = json.loads(accumulated_content)
                parsed = response_format.model_validate(parsed_data)
              except (json.JSONDecodeError, Exception):
                pass  # Non-critical: content is still available as string

          # Add assistant message to history
          assistant_msg = Message(
            role="assistant",
            content=accumulated_content or None,
            tool_calls=accumulated_tool_calls or None,
          )
          if accumulated_metrics is not None:
            assistant_msg.metrics = accumulated_metrics
            total_metrics = accumulated_metrics if total_metrics is None else total_metrics + accumulated_metrics
          self._messages.append(assistant_msg)

          content = accumulated_content
          tool_calls = accumulated_tool_calls
          metrics = accumulated_metrics
        else:
          # Non-streaming
          content, tool_calls, metrics, parsed = await self._call_model()

        completed_evt = self._make_model_call_completed(content, tool_calls, metrics)
        yield completed_evt

        if metrics is not None and not self._streaming:
          total_metrics = metrics if total_metrics is None else total_metrics + metrics

        # 5. If no tool calls -> done
        if not tool_calls:
          final_content = content
          final_parsed = parsed
          break

        # 6. Parallel tool dispatch
        batch = await self._execute_tools(tool_calls)
        for event in batch.events:
          yield event  # type: ignore[misc]

        # 7. Check stop_after_tool_call
        if any(r.should_stop for r in batch.results):
          final_content = content
          final_parsed = parsed
          break

        # 8. Check HITL pause
        paused = [r for r in batch.results if r.is_paused]
        if paused:
          requirements = [r.requirement for r in paused if r.requirement is not None]
          paused_tools = [r.tool_execution for r in paused if r.tool_execution is not None]
          yield RunPausedEvent(
            run_id=self._context.run_id,
            session_id=self._context.session_id,
            agent_id=self._agent_id,
            agent_name=self._agent_name,
            requirements=requirements,
            tools=paused_tools,
          )
          return  # Caller must use continue_run()

        # 9. Append tool results to messages (already done in _execute_tools)
        # Loop continues

      # Yield RunCompleted with final content
      yield RunCompletedEvent(
        run_id=self._context.run_id,
        session_id=self._context.session_id,
        agent_id=self._agent_id,
        agent_name=self._agent_name,
        content=final_content,
        parsed=final_parsed,
        metrics=total_metrics,
      )

    except Exception as e:
      yield RunErrorEvent(
        run_id=self._context.run_id,
        session_id=self._context.session_id,
        agent_id=self._agent_id,
        agent_name=self._agent_name,
        error_type=type(e).__name__,
        content=str(e),
      )
      raise

  async def run_streaming(self) -> AsyncGenerator[RunOutputEvent, None]:
    """Alias for ``run()`` — streaming is controlled by the ``streaming`` constructor flag."""
    async for event in self.run():
      yield event

  # ------------------------------------------------------------------
  # Model call event helpers
  # ------------------------------------------------------------------

  def _make_model_call_started(self) -> ModelCallStartedEvent:
    self._turn += 1
    return ModelCallStartedEvent(
      run_id=self._context.run_id,
      session_id=self._context.session_id,
      agent_id=self._agent_id,
      agent_name=self._agent_name,
      turn=self._turn,
      messages=list(self._messages),
      tool_definitions=self._tools_dicts,
      response_format=self._context.output_schema,
      model_id=self._model.id,
      model_provider=getattr(self._model, "provider", "") or "",
    )

  def _make_model_call_completed(self, content: str, tool_calls: list[dict], metrics: Optional["Metrics"]) -> ModelCallCompletedEvent:
    return ModelCallCompletedEvent(
      run_id=self._context.run_id,
      session_id=self._context.session_id,
      agent_id=self._agent_id,
      agent_name=self._agent_name,
      turn=self._turn,
      content=content or None,
      tool_calls=tool_calls or None,
      metrics=metrics,
      model_id=self._model.id,
    )

  # ------------------------------------------------------------------
  # Model calls
  # ------------------------------------------------------------------

  async def _call_model(self) -> tuple[str, list[dict], Optional["Metrics"], Any]:
    """Non-streaming model call with retry. Returns (content, tool_calls, metrics, parsed)."""
    response = await self._call_model_with_retry()

    # Add assistant message to conversation history
    assistant_msg = Message(
      role="assistant",
      content=response.content,
      tool_calls=response.tool_calls or None,
    )
    if response.response_usage is not None:
      assistant_msg.metrics = response.response_usage
    self._messages.append(assistant_msg)

    return (
      response.content or "",
      response.tool_calls or [],
      response.response_usage,
      response.parsed,
    )

  async def _call_model_with_retry(self) -> "ModelResponse":
    """Call model with retry on transient errors (exponential backoff)."""
    max_retries = self._config.max_retries if self._config.retry_transient_errors else 0
    backoff_base = self._config.retry_backoff_base

    for attempt in range(max_retries + 1):
      try:
        assistant_message = Message(role="assistant")
        return await self._model.ainvoke(
          messages=self._messages,
          assistant_message=assistant_message,
          tools=self._tools_dicts,
          response_format=self._context.output_schema,
        )
      except Exception as e:
        is_transient = isinstance(e, (ConnectionError, TimeoutError, OSError))
        if not self._config.retry_transient_errors or not is_transient:
          raise
        if attempt >= max_retries:
          raise
        delay = min(backoff_base * (2**attempt), 60.0)
        log_debug(f"Transient error (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {delay:.1f}s")
        await asyncio.sleep(delay)

    # Unreachable, but keeps type checkers happy
    raise RuntimeError("Exhausted retries")  # pragma: no cover

  async def _force_final_answer(self) -> tuple[str, Optional["Metrics"], Any]:
    """Inject stop message and call model without tools for a final answer."""
    self._messages.append(
      Message(
        role="user",
        content=(
          f"[SYSTEM] Tool-call limit reached ({self._config.max_tool_rounds} rounds). "
          "You MUST provide your final answer NOW. Do NOT call any more tools. "
          "Summarize what you accomplished and any remaining issues."
        ),
      )
    )

    started_evt = self._make_model_call_started()
    self._emit_fn(started_evt)

    assistant_msg = Message(role="assistant")
    final_response = await self._model.ainvoke(
      messages=self._messages,
      assistant_message=assistant_msg,
      tools=None,
      response_format=self._context.output_schema,
    )
    self._messages.append(Message(role="assistant", content=final_response.content))

    completed_evt = self._make_model_call_completed(final_response.content or "", [], final_response.response_usage)
    self._emit_fn(completed_evt)

    return final_response.content or "", final_response.response_usage, final_response.parsed

  # ------------------------------------------------------------------
  # Tool dispatch
  # ------------------------------------------------------------------

  async def _execute_tools(self, tool_calls: list[dict]) -> ToolBatchResult:
    """Execute tool calls — parallel by default, sequential when flagged."""
    parallel_calls: list[dict] = []
    sequential_calls: list[dict] = []

    for tc in tool_calls:
      fn_name = tc.get("function", {}).get("name", "")
      fn = self._tools.get(fn_name)
      if fn and fn.sequential:
        sequential_calls.append(tc)
      else:
        parallel_calls.append(tc)

    all_events: list[BaseRunOutputEvent] = []
    all_results: list[ToolResult] = []

    # Execute parallel tools via asyncio.gather
    if parallel_calls:
      tasks = [self._execute_single_tool(tc) for tc in parallel_calls]
      results = await asyncio.gather(*tasks, return_exceptions=True)
      for i, r in enumerate(results):
        if isinstance(r, BaseException):
          tc = parallel_calls[i]
          fn_name = tc.get("function", {}).get("name", "unknown")
          tr = ToolResult(
            tool_call_id=tc.get("id"),
            tool_name=fn_name,
            error=str(r),
          )
          # Add tool error message to conversation
          self._messages.append(
            Message(
              role="tool",
              content=f"Error: {r}",
              tool_call_id=tc.get("id"),
              name=fn_name,
            )
          )
          all_results.append(tr)
        else:
          all_results.append(r)
          all_events.extend(r.events)

    # Execute sequential tools in order
    for tc in sequential_calls:
      if self._cancellation_token:
        self._cancellation_token.raise_if_cancelled()
      result = await self._execute_single_tool(tc)
      all_results.append(result)
      all_events.extend(result.events)

    return ToolBatchResult(results=all_results, events=all_events)

  async def _execute_single_tool(self, tool_call: dict) -> ToolResult:
    """Execute one tool call with HITL checks, guardrails, events."""
    function_call = get_function_call_for_tool_call(tool_call, self._tools)
    fn_name = tool_call.get("function", {}).get("name", "unknown")
    fn = self._tools.get(fn_name)
    events: list[BaseRunOutputEvent] = []

    # Build ToolExecution for tracking
    tool_execution = ToolExecution(
      tool_call_id=tool_call.get("id"),
      tool_name=fn_name,
      tool_args=function_call.arguments if function_call else None,
    )

    # Emit ToolCallStarted
    started_event = ToolCallStartedEvent(
      run_id=self._context.run_id,
      session_id=self._context.session_id,
      agent_id=self._agent_id,
      agent_name=self._agent_name,
      tool=tool_execution,
    )
    events.append(started_event)

    # ---- HITL: requires_confirmation ----
    if fn and fn.requires_confirmation:
      tool_execution.requires_confirmation = True
      requirement = RunRequirement(tool_execution)
      return ToolResult(
        tool_call_id=tool_call.get("id"),
        tool_name=fn_name,
        is_paused=True,
        requirement=requirement,
        tool_execution=tool_execution,
        events=events,
      )

    # ---- HITL: requires_user_input ----
    if fn and fn.requires_user_input:
      tool_execution.requires_user_input = True
      tool_execution.user_input_schema = fn.user_input_schema
      requirement = RunRequirement(tool_execution)
      return ToolResult(
        tool_call_id=tool_call.get("id"),
        tool_name=fn_name,
        is_paused=True,
        requirement=requirement,
        tool_execution=tool_execution,
        events=events,
      )

    # ---- HITL: external_execution ----
    if fn and fn.external_execution:
      tool_execution.external_execution_required = True
      requirement = RunRequirement(tool_execution)
      return ToolResult(
        tool_call_id=tool_call.get("id"),
        tool_name=fn_name,
        is_paused=True,
        requirement=requirement,
        tool_execution=tool_execution,
        events=events,
      )

    # ---- Tool guardrails ----
    if self._guardrails and hasattr(self._guardrails, "tool") and self._guardrails.tool:
      # Guardrail check is done by the agent — we get the result via the guardrails object
      # For now, guardrails are handled at the agent level and passed as blocked tool results
      pass

    # ---- Execute (with ToolRetry support) ----
    if function_call:
      try:
        result_obj = await function_call.aexecute()
        tool_execution.result = str(result_obj.result) if result_obj.status == "success" else str(result_obj.error)
        tool_execution.tool_call_error = result_obj.status == "failure"
      except Exception as exc:
        # Lazy import to avoid circular dependency
        from definable.agent.pipeline.tool_retry import ToolRetry as _ToolRetry

        if not isinstance(exc, _ToolRetry):
          raise
        retry = exc
        # ToolRetry: send feedback to model, track retry count
        call_id = tool_call.get("id", "")
        count = self._tool_retry_counts.get(call_id, 0) + 1
        self._tool_retry_counts[call_id] = count

        if count > retry.max_retries:
          tool_execution.result = f"[RETRY EXHAUSTED] Tool '{fn_name}' failed after {retry.max_retries} retries: {retry.message}"
          tool_execution.tool_call_error = True
        else:
          tool_execution.result = f"[RETRY] {retry.message}"
          tool_execution.tool_call_error = False
    else:
      tool_execution.result = f"Tool '{fn_name}' not found"
      tool_execution.tool_call_error = True

    self._all_tool_executions.append(tool_execution)

    # Emit ToolCallCompleted
    completed_event = ToolCallCompletedEvent(
      run_id=self._context.run_id,
      session_id=self._context.session_id,
      agent_id=self._agent_id,
      agent_name=self._agent_name,
      tool=tool_execution,
      content=tool_execution.result,
    )
    events.append(completed_event)

    # Add tool result message to conversation
    self._messages.append(
      Message(
        role="tool",
        content=tool_execution.result,
        tool_call_id=tool_call.get("id"),
        name=fn_name,
      )
    )

    return ToolResult(
      tool_call_id=tool_call.get("id"),
      tool_name=fn_name,
      result=tool_execution.result if not tool_execution.tool_call_error else None,
      error=tool_execution.result if tool_execution.tool_call_error else None,
      should_stop=bool(fn and fn.stop_after_tool_call),
      events=events,
      tool_execution=tool_execution,
    )

  # ------------------------------------------------------------------
  # Accessors
  # ------------------------------------------------------------------

  @property
  def messages(self) -> list[Message]:
    """Current message list (mutated during the loop)."""
    return self._messages

  @property
  def tool_executions(self) -> list[ToolExecution]:
    """All tool executions accumulated during the loop."""
    return self._all_tool_executions


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _merge_tool_call_deltas(existing: list[dict], new_deltas: list[Any]) -> list[dict]:
  """Merge streaming tool call deltas into accumulated tool calls."""
  for delta in new_deltas:
    # Handle both dict and object formats
    if hasattr(delta, "index"):
      index = delta.index
    elif isinstance(delta, dict):
      index = delta.get("index", 0)
    else:
      index = 0

    # Ensure list is long enough
    while index >= len(existing):
      existing.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})

    # Get delta values
    if isinstance(delta, dict):
      delta_id = delta.get("id", "")
      delta_type = delta.get("type", "")
      delta_func = delta.get("function", {})
      delta_name = delta_func.get("name", "") if isinstance(delta_func, dict) else ""
      delta_args = delta_func.get("arguments", "") if isinstance(delta_func, dict) else ""
    else:
      delta_id = getattr(delta, "id", "") or ""
      delta_type = getattr(delta, "type", "") or ""
      delta_func = getattr(delta, "function", None)
      delta_name = getattr(delta_func, "name", "") or "" if delta_func else ""
      delta_args = getattr(delta_func, "arguments", "") or "" if delta_func else ""

    # Merge fields
    if delta_id:
      existing[index]["id"] = delta_id
    if delta_type:
      existing[index]["type"] = delta_type
    if delta_name:
      existing[index]["function"]["name"] += delta_name
    if delta_args:
      existing[index]["function"]["arguments"] += delta_args

  return existing
