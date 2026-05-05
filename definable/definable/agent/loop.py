"""Unified async-generator agentic loop.

Single implementation for streaming and non-streaming modes.
Yields RunOutputEvent instances throughout execution.
"""

import asyncio
from dataclasses import dataclass, field, replace as _dc_replace
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
  ReasoningCompletedEvent,
  ReasoningContentDeltaEvent,
  ReasoningStartedEvent,
  RunCompletedEvent,
  RunContentEvent,
  RunContext,
  RunErrorEvent,
  RunOutputEvent,
  ToolCallCompletedEvent,
  ToolCallStartedEvent,
  ToolContentEvent,
)
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
    native_thinking: bool = False,
    cancellation_token: Optional[CancellationToken] = None,
    compression_manager: Optional["CompressionManager"] = None,
    guardrails: Optional["Guardrails"] = None,
    emit_fn: Callable[[BaseRunOutputEvent], None],
    agent_id: str,
    agent_name: str,
    deferred_tool_manager: Optional[Any] = None,
    permission_service: Optional[Any] = None,
  ) -> None:
    self._model = model
    self._tools = tools
    self._messages = messages
    self._context = context
    self._config = config
    self._streaming = streaming
    self._native_thinking = native_thinking
    self._cancellation_token = cancellation_token
    self._compression_manager = compression_manager
    self._compress_tool_results = compression_manager is not None and compression_manager.compress_tool_results
    self._guardrails = guardrails
    self._emit_fn = emit_fn
    self._agent_id = agent_id
    self._agent_name = agent_name
    self._deferred_tool_manager = deferred_tool_manager
    self._permission_service = permission_service

    # Precompute tool dicts for model API (OpenAI format)
    self._tools_dicts: Optional[list[dict]] = [{"type": "function", "function": t.to_dict()} for t in tools.values()] if tools else None

    # Accumulated state during the loop
    self._all_tool_executions: list[ToolExecution] = []
    self._turn: int = 0
    self._tool_retry_counts: Dict[str, int] = {}  # tool_call_id → retry count
    self._native_reasoning_content: Optional[str] = None  # accumulated native thinking content

  # ------------------------------------------------------------------
  # Public API
  # ------------------------------------------------------------------

  async def run(self) -> AsyncGenerator[RunOutputEvent, None]:
    """The unified loop. Yields events as they occur.

    When ``streaming=True``, yields ``RunContentEvent`` deltas during
    the model call. When ``streaming=False``, uses non-streaming model calls.
    Tool dispatch, permissions, and all other logic is shared.
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
          accumulated_reasoning = ""
          accumulated_tool_calls: list[dict] = []
          accumulated_metrics: Optional["Metrics"] = None
          reasoning_started_emitted = False
          reasoning_completed_emitted = False

          assistant_message = Message(role="assistant")
          async for chunk in self._model.ainvoke_stream(
            messages=self._messages,
            assistant_message=assistant_message,
            tools=self._tools_dicts,
            response_format=self._context.output_schema,
            compress_tool_results=self._compress_tool_results,
          ):
            # Native thinking: stream reasoning_content deltas as events
            if self._native_thinking and hasattr(chunk, "reasoning_content") and chunk.reasoning_content:
              if not reasoning_started_emitted:
                reasoning_started_emitted = True
                yield ReasoningStartedEvent(
                  run_id=self._context.run_id,
                  session_id=self._context.session_id,
                  agent_id=self._agent_id,
                  agent_name=self._agent_name,
                )
              accumulated_reasoning += chunk.reasoning_content
              yield ReasoningContentDeltaEvent(
                run_id=self._context.run_id,
                session_id=self._context.session_id,
                agent_id=self._agent_id,
                agent_name=self._agent_name,
                reasoning_content=chunk.reasoning_content,
              )
            if hasattr(chunk, "content") and chunk.content:
              # If we were streaming reasoning and now get content, close reasoning
              if reasoning_started_emitted and not reasoning_completed_emitted:
                reasoning_completed_emitted = True
                yield ReasoningCompletedEvent(
                  run_id=self._context.run_id,
                  session_id=self._context.session_id,
                  agent_id=self._agent_id,
                  agent_name=self._agent_name,
                )
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

          # Close reasoning if it was started but never completed (e.g. tool calls follow)
          if reasoning_started_emitted and not reasoning_completed_emitted:
            yield ReasoningCompletedEvent(
              run_id=self._context.run_id,
              session_id=self._context.session_id,
              agent_id=self._agent_id,
              agent_name=self._agent_name,
            )

          # Store accumulated reasoning content
          if accumulated_reasoning:
            self._native_reasoning_content = accumulated_reasoning

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
          if accumulated_reasoning:
            assistant_msg.reasoning_content = accumulated_reasoning
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

        # Emit native thinking events for non-streaming mode
        if not self._streaming and self._native_thinking and self._native_reasoning_content:
          yield ReasoningStartedEvent(
            run_id=self._context.run_id,
            session_id=self._context.session_id,
            agent_id=self._agent_id,
            agent_name=self._agent_name,
          )
          yield ReasoningContentDeltaEvent(
            run_id=self._context.run_id,
            session_id=self._context.session_id,
            agent_id=self._agent_id,
            agent_name=self._agent_name,
            reasoning_content=self._native_reasoning_content,
          )
          yield ReasoningCompletedEvent(
            run_id=self._context.run_id,
            session_id=self._context.session_id,
            agent_id=self._agent_id,
            agent_name=self._agent_name,
          )

        # In streaming mode, content was already yielded via RunContentEvent deltas,
        # so omit it from ModelCallCompletedEvent to avoid duplication.
        completed_evt = self._make_model_call_completed("" if self._streaming else content, tool_calls, metrics)
        yield completed_evt

        if metrics is not None and not self._streaming:
          total_metrics = metrics if total_metrics is None else total_metrics + metrics

        # 5. If no tool calls -> done
        if not tool_calls:
          final_content = content
          final_parsed = parsed
          break

        # 6. Parallel tool dispatch (async generator — yields events then ToolBatchResult)
        batch: Optional[ToolBatchResult] = None
        async for item in self._execute_tools(tool_calls):
          if isinstance(item, ToolBatchResult):
            batch = item
          else:
            yield item  # type: ignore[misc]
        assert batch is not None

        # 7. Check stop_after_tool_call
        if any(r.should_stop for r in batch.results):
          final_content = content
          final_parsed = parsed
          break

        # 8. Deferred tools: if load_tools was called, refresh the tool set
        if self._deferred_tool_manager is not None:
          refreshed = self._deferred_tool_manager.get_active_tools()
          if len(refreshed) != len(self._tools):
            self._tools = refreshed
            self._tools_dicts = [{"type": "function", "function": t.to_dict()} for t in refreshed.values()]

        # 10. Append tool results to messages (already done in _execute_tools)
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

    # Capture native reasoning content if present
    if self._native_thinking and response.reasoning_content:
      self._native_reasoning_content = response.reasoning_content

    # Add assistant message to conversation history
    assistant_msg = Message(
      role="assistant",
      content=response.content,
      tool_calls=response.tool_calls or None,
    )
    if response.reasoning_content:
      assistant_msg.reasoning_content = response.reasoning_content
    if response.redacted_reasoning_content:
      assistant_msg.redacted_reasoning_content = response.redacted_reasoning_content
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
          compress_tool_results=self._compress_tool_results,
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
      compress_tool_results=self._compress_tool_results,
    )
    self._messages.append(Message(role="assistant", content=final_response.content))

    completed_evt = self._make_model_call_completed(final_response.content or "", [], final_response.response_usage)
    self._emit_fn(completed_evt)

    return final_response.content or "", final_response.response_usage, final_response.parsed

  # ------------------------------------------------------------------
  # Tool dispatch
  # ------------------------------------------------------------------

  async def _execute_tools(self, tool_calls: list[dict]) -> AsyncGenerator[Any, None]:
    """Execute tool calls — yields events in real-time, then a final ToolBatchResult.

    Parallel tools push events via an asyncio.Queue so they stream as they happen.
    Sequential tools yield events immediately after each tool completes.
    """
    parallel_calls: list[dict] = []
    sequential_calls: list[dict] = []

    for tc in tool_calls:
      fn_name = tc.get("function", {}).get("name", "")
      fn = self._tools.get(fn_name)
      if fn and fn.sequential:
        sequential_calls.append(tc)
      else:
        parallel_calls.append(tc)

    all_results: list[ToolResult] = []

    # Execute parallel tools — stream events via queue
    if parallel_calls:
      event_queue: asyncio.Queue[Any] = asyncio.Queue()
      _sentinel = object()

      async def _gather_then_signal() -> list:
        results = await asyncio.gather(
          *[self._execute_single_tool(tc, event_sink=event_queue.put_nowait) for tc in parallel_calls],
          return_exceptions=True,
        )
        event_queue.put_nowait(_sentinel)
        return list(results)

      gather_task = asyncio.create_task(_gather_then_signal())

      # Drain queue, yielding events as they arrive
      while True:
        item = await event_queue.get()
        if item is _sentinel:
          break
        yield item

      results = gather_task.result()
      for i, r in enumerate(results):
        if isinstance(r, BaseException):
          tc = parallel_calls[i]
          fn_name = tc.get("function", {}).get("name", "unknown")
          tr = ToolResult(
            tool_call_id=tc.get("id"),
            tool_name=fn_name,
            error=str(r),
          )
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

    # Execute sequential tools — yield events immediately after each
    for tc in sequential_calls:
      if self._cancellation_token:
        self._cancellation_token.raise_if_cancelled()
      result = await self._execute_single_tool(tc)
      for evt in result.events:
        yield evt
      all_results.append(result)

    yield ToolBatchResult(results=all_results, events=[])

  async def _execute_single_tool(
    self,
    tool_call: dict,
    event_sink: Optional[Callable[[BaseRunOutputEvent], None]] = None,
  ) -> ToolResult:
    """Execute one tool call with HITL checks, guardrails, events."""
    function_call = get_function_call_for_tool_call(tool_call, self._tools)
    fn_name = tool_call.get("function", {}).get("name", "unknown")
    fn = self._tools.get(fn_name)
    events: list[BaseRunOutputEvent] = []

    def emit(evt: BaseRunOutputEvent) -> None:
      if event_sink is not None:
        event_sink(evt)
      else:
        events.append(evt)

    # Build ToolExecution for tracking
    tool_execution = ToolExecution(
      tool_call_id=tool_call.get("id"),
      tool_name=fn_name,
      tool_args=function_call.arguments if function_call else None,
    )

    # Emit ToolCallStarted (snapshot to decouple from later result mutation)
    started_event = ToolCallStartedEvent(
      run_id=self._context.run_id,
      session_id=self._context.session_id,
      agent_id=self._agent_id,
      agent_name=self._agent_name,
      tool=_dc_replace(tool_execution),
    )
    emit(started_event)

    # ---- Permission check ----
    if self._permission_service is not None and fn is not None:
      from definable.agent.hitl.types import PermissionDecision, PermissionRequest

      perm_request = PermissionRequest(
        tool_name=fn_name,
        tool_args=function_call.arguments if function_call and function_call.arguments else {},
        tool_call_id=tool_call.get("id"),
      )
      perm_response = await self._permission_service.check(perm_request)

      if perm_response.decision == PermissionDecision.deny:
        denial_msg = perm_response.feedback or f"[DENIED] Tool '{fn_name}' was denied by the user."
        tool_execution.result = denial_msg
        tool_execution.tool_call_error = True
        self._all_tool_executions.append(tool_execution)

        completed_event = ToolCallCompletedEvent(
          run_id=self._context.run_id,
          session_id=self._context.session_id,
          agent_id=self._agent_id,
          agent_name=self._agent_name,
          tool=_dc_replace(tool_execution),
        )
        emit(completed_event)

        self._messages.append(
          Message(
            role="tool",
            content=denial_msg,
            tool_call_id=tool_call.get("id"),
            name=fn_name,
          )
        )
        return ToolResult(
          tool_call_id=tool_call.get("id"),
          tool_name=fn_name,
          error=denial_msg,
          tool_execution=tool_execution,
          events=events,
        )

    # ---- Tool guardrails ----
    if self._guardrails and hasattr(self._guardrails, "tool") and self._guardrails.tool:
      tool_args = tool_call.get("arguments", {}) if isinstance(tool_call.get("arguments"), dict) else {}
      gr_results = await self._guardrails.run_tool_checks(fn_name, tool_args, self._context)
      for gr in gr_results:
        if gr.action == "block":
          tool_execution.result = gr.message or f"Tool '{fn_name}' blocked by guardrail"
          tool_execution.tool_call_error = True
          self._all_tool_executions.append(tool_execution)
          return ToolResult(
            tool_call_id=tool_call.get("id"),
            tool_name=fn_name,
            result=tool_execution.result,
            error=tool_execution.result,
            tool_execution=tool_execution,
            events=events,
          )

    # ---- Execute (with ToolRetry support) ----
    if function_call:
      try:
        result_obj = await function_call.aexecute()
        if result_obj.status == "success":
          tool_execution.result = await self._resolve_tool_result(
            result_obj.result,
            fn_name=fn_name,
            tool_call_id=tool_call.get("id"),
            events=events,
            event_sink=event_sink,
          )
        else:
          tool_execution.result = str(result_obj.error)
        tool_execution.tool_call_error = result_obj.status == "failure"
      except Exception as exc:
        # Handle StopAgentRun — graceful loop termination
        from definable.exceptions import StopAgentRun as _StopAgentRun

        if isinstance(exc, _StopAgentRun):
          tool_execution.result = str(exc.user_message or exc)
          tool_execution.tool_call_error = False
          self._all_tool_executions.append(tool_execution)
          return ToolResult(
            tool_call_id=tool_call.get("id"),
            tool_name=fn_name,
            result=tool_execution.result,
            should_stop=True,
            tool_execution=tool_execution,
            events=events,
          )

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
    emit(completed_event)

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
  # Generator consumption
  # ------------------------------------------------------------------

  async def _resolve_tool_result(
    self,
    result: Any,
    *,
    fn_name: str,
    tool_call_id: Optional[str],
    events: list[BaseRunOutputEvent],
    event_sink: Optional[Callable[[BaseRunOutputEvent], None]] = None,
  ) -> str:
    """Consume a tool result, handling async/sync generators transparently.

    For generators: iterates all chunks, emits a ToolContentEvent per chunk,
    and returns the accumulated string.  For plain values: returns ``str(result)``.
    """
    from inspect import isasyncgen, isgenerator

    def _emit(evt: BaseRunOutputEvent) -> None:
      if event_sink is not None:
        event_sink(evt)
      else:
        events.append(evt)

    if isasyncgen(result):
      chunks: list[str] = []
      idx = 0
      last_content_event: Optional[ToolContentEvent] = None
      async for chunk in result:
        chunk_str = str(chunk)
        chunks.append(chunk_str)
        evt = ToolContentEvent(
          run_id=self._context.run_id,
          session_id=self._context.session_id,
          agent_id=self._agent_id,
          agent_name=self._agent_name,
          tool_name=fn_name,
          tool_call_id=tool_call_id,
          chunk=chunk_str,
          chunk_index=idx,
        )
        last_content_event = evt
        _emit(evt)
        idx += 1
      # Mark the last event as final (if any chunks were emitted)
      if last_content_event is not None:
        last_content_event.is_final = True
      return "\n".join(chunks) if chunks else ""

    if isgenerator(result):
      chunks = []
      idx = 0
      last_content_event = None
      for chunk in result:
        chunk_str = str(chunk)
        chunks.append(chunk_str)
        evt = ToolContentEvent(
          run_id=self._context.run_id,
          session_id=self._context.session_id,
          agent_id=self._agent_id,
          agent_name=self._agent_name,
          tool_name=fn_name,
          tool_call_id=tool_call_id,
          chunk=chunk_str,
          chunk_index=idx,
        )
        last_content_event = evt
        _emit(evt)
        idx += 1
      if last_content_event is not None:
        last_content_event.is_final = True
      return "\n".join(chunks) if chunks else ""

    return str(result)

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

  @property
  def native_reasoning_content(self) -> Optional[str]:
    """Native thinking content accumulated during the loop (if native thinking was active)."""
    return self._native_reasoning_content


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _merge_tool_call_deltas(existing: list[dict], new_deltas: list[Any]) -> list[dict]:
  """Merge streaming tool call deltas into accumulated tool calls.

  Handles two formats:
  - Streaming deltas (OpenAI-style): objects with .index attribute, name/arguments
    arrive across multiple chunks and must be concatenated.
  - Complete tool calls (Anthropic/Gemini/Mistral/Ollama): dicts without "index",
    each representing a fully-formed tool call that should be appended as-is.
  """
  for delta in new_deltas:
    # Determine index — distinguishes streaming deltas from complete tool calls.
    index: int | None = None
    if hasattr(delta, "index") and delta.index is not None:
      index = delta.index
    elif isinstance(delta, dict) and "index" in delta:
      index = delta["index"]

    # No index → complete tool call (e.g. Anthropic, Gemini, Mistral, Ollama).
    # Append directly instead of merging into an existing slot.
    if index is None:
      if isinstance(delta, dict):
        func = delta.get("function", {})
        existing.append({
          "id": delta.get("id", ""),
          "type": delta.get("type", "function"),
          "function": {
            "name": func.get("name", "") if isinstance(func, dict) else "",
            "arguments": func.get("arguments", "") if isinstance(func, dict) else "",
          },
        })
      else:
        func = getattr(delta, "function", None)
        existing.append({
          "id": getattr(delta, "id", "") or "",
          "type": getattr(delta, "type", "") or "function",
          "function": {
            "name": getattr(func, "name", "") or "" if func else "",
            "arguments": getattr(func, "arguments", "") or "" if func else "",
          },
        })
      continue

    # Streaming delta with index — merge into the existing slot.
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
