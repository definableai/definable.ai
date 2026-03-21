"""Agent v2 — overridable methods + EventBus.

Four overridable methods control flow:
  before_model_call()  →  build system prompt, inject memory/knowledge
  call_model()         →  call LLM with retry + streaming
  after_model_call()   →  validate output (override for guardrails)
  execute_tool()       →  run tool (override for HITL, sandboxing)

One event system for observation (never mutates flow):
  EventBus  →  tracing, UI streaming, logging, metrics
"""

import asyncio
import contextlib
import dataclasses
import json
from dataclasses import replace as _dc_replace
from typing import (
  TYPE_CHECKING,
  Any,
  AsyncIterator,
  Callable,
  Dict,
  Iterator,
  List,
  Optional,
  Protocol,
  Type,
  Union,
  runtime_checkable,
)
from uuid import uuid4

from definable.agent.config import AgentConfig
from definable.agent.event_bus import EventBus
from definable.agent.loop import CancelToken, Cancelled, ToolResult
from definable.agent.toolkit import Toolkit
from definable.agent.tracing.base import TraceWriter
from definable.media import Audio, File, Image, Video
from definable.model.message import Message
from definable.model.metrics import Metrics
from definable.model.response import ToolExecution
from definable.agent.events import (
  BaseRunOutputEvent,
  ModelCallCompletedEvent,
  ModelCallStartedEvent,
  ReasoningCompletedEvent,
  ReasoningContentDeltaEvent,
  ReasoningStartedEvent,
  RunCancelledEvent,
  RunCompletedEvent,
  RunContentEvent,
  RunContext,
  RunErrorEvent,
  RunInput,
  RunOutput,
  RunOutputEvent,
  RunPausedEvent,
  RunStartedEvent,
  RunStatus,
  ToolCallCompletedEvent,
  ToolCallStartedEvent,
  ToolContentEvent,
)
from definable.agent.run.requirement import RunRequirement
from definable.skill.base import Skill
from definable.tool.function import Function
from definable.utils.tools import get_function_call_for_tool_call
from pydantic import BaseModel

if TYPE_CHECKING:
  from definable.agent.compression import Compression, CompressionManager
  from definable.agent.interface.base import BaseInterface
  from definable.agent.interface.gateway import InterfaceGateway
  from definable.agent.tracing.base import Tracing
  from definable.knowledge import Knowledge
  from definable.memory.manager import Memory
  from definable.model.base import Model
  from definable.reader.base import BaseReader
  from definable.skill.registry import SkillRegistry

_REJECTED_MSG = (
  "[REJECTED] The user rejected this tool call. Do NOT retry this tool. Respond to the user explaining that the action was not performed."
)


@runtime_checkable
class AsyncLifecycleToolkit(Protocol):
  """Protocol for toolkits with async lifecycle (e.g. MCPToolkit)."""

  _initialized: bool

  async def initialize(self) -> None: ...
  async def shutdown(self) -> None: ...


class Agent:
  """Production-grade agent with overridable methods.

  Override ``before_model_call``, ``call_model``, ``after_model_call``,
  or ``execute_tool`` to customize behavior without subclassing the loop.
  """

  def __init__(
    self,
    *,
    # ── Identity ────────────────────────────────────────────
    name: Optional[str] = None,
    session_id: Optional[str] = None,
    instructions: Optional[str] = None,
    config: Optional[AgentConfig] = None,
    # ── Model ───────────────────────────────────────────────
    model: Union[str, "Model"],
    # ── Layers ──────────────────────────────────────────────
    memory: Union["Memory", bool, None] = False,
    knowledge: Union["Knowledge", str, bool, None] = False,
    compression: Union[bool, "Compression", None] = None,
    # ── Tools ───────────────────────────────────────────────
    tools: Optional[List[Function]] = None,
    toolkits: Optional[List[Toolkit]] = None,
    skills: Optional[List[Skill]] = None,
    skill_registry: Optional["SkillRegistry"] = None,
    # ── Observability ───────────────────────────────────────
    tracing: Union[bool, "Tracing", None] = False,
    debug: Union[bool, None] = False,
    # ── Media ───────────────────────────────────────────────
    audio_transcriber: Union[bool, Any, None] = None,
    # ── Interfaces ──────────────────────────────────────────
    interfaces: Union["BaseInterface", List["BaseInterface"], None] = None,
    gateway: Optional["InterfaceGateway"] = None,
    # ── Support ─────────────────────────────────────────────
    readers: Union[List["BaseReader"], bool, None] = None,
  ):
    # ── Model resolution ──────────────────────────────────
    if model is None:
      raise TypeError(
        "Agent requires a 'model' argument. Pass a Model instance "
        "(e.g., OpenAIChat(id='gpt-4o-mini')) or a string shorthand "
        "(e.g., 'openai/gpt-4o-mini')."
      )
    self.model: "Model"
    if isinstance(model, str):
      from definable.model.utils import resolve_model_string

      self.model = resolve_model_string(model)
    else:
      self.model = model

    self.tools = tools or []
    self.toolkits = toolkits or []
    self.skills = skills or []
    self.instructions: Optional[str] = "\n".join(str(i) for i in instructions) if isinstance(instructions, list) else instructions
    self.readers = self._init_readers(readers)
    self.guardrails = None  # Preserved for interface compat

    # ── Config ────────────────────────────────────────────
    self.config = config or AgentConfig()
    if name is not None:
      self.config = dataclasses.replace(self.config, agent_name=name)

    # ── Memory (tool-based) ──────────────────────────────
    self.memory = self._resolve_memory(memory)
    if self.memory is not None and hasattr(self.memory, "get_skill"):
      self.skills.append(self.memory.get_skill())

    # ── Knowledge ─────────────────────────────────────────
    self._knowledge: Optional["Knowledge"] = self._resolve_knowledge(knowledge)

    # ── Tracing ───────────────────────────────────────────
    self._tracing_config: Optional["Tracing"] = self._resolve_tracing(tracing, self.config)

    # ── Debug mode ────────────────────────────────────────
    if debug:
      from definable.agent.tracing.base import Tracing as _Tracing
      from definable.agent.tracing.debug import DebugExporter

      if self._tracing_config is None:
        self._tracing_config = _Tracing(exporters=[DebugExporter()])
      else:
        existing = self._tracing_config.exporters or []
        self._tracing_config = dataclasses.replace(self._tracing_config, exporters=[*existing, DebugExporter()])

    # ── Audio transcriber ─────────────────────────────────
    from definable.reader.audio import AudioTranscriber as _AudioTranscriber
    from definable.reader.audio import OpenAITranscriber as _OpenAITranscriber

    if audio_transcriber is True:
      self._audio_transcriber: Optional[_AudioTranscriber] = _OpenAITranscriber()
    elif isinstance(audio_transcriber, _AudioTranscriber):
      self._audio_transcriber = audio_transcriber
    else:
      self._audio_transcriber = None

    # ── Skill registry ────────────────────────────────────
    if skill_registry is not None:
      from definable.skill.registry import SkillRegistry

      if isinstance(skill_registry, SkillRegistry):
        self.skills.append(skill_registry.as_on_demand())

    # ── Initialize skills ─────────────────────────────────
    self._init_skills()

    # ── Internal state ────────────────────────────────────
    self._tools_dict: Dict[str, Function] = self._flatten_tools()
    self._trace_writer: Optional[TraceWriter] = self._init_tracing()
    self._compression_manager: Optional["CompressionManager"] = self._resolve_compression(compression)
    self._interfaces: List["BaseInterface"] = []
    self._gateway: Optional["InterfaceGateway"] = None
    self._triggers: List[Any] = []
    self._before_hooks: List[Callable] = []
    self._after_hooks: List[Callable] = []
    self._auth: Optional[Any] = None
    self._started = False
    self._event_bus: EventBus = EventBus()
    self._agent_owned_toolkits: list[Any] = []
    self._toolkit_init_lock: asyncio.Lock = asyncio.Lock()
    self._session_id_explicit = session_id is not None
    self.session_id = session_id or str(uuid4())

    # ── Interfaces ────────────────────────────────────────
    if interfaces is not None:
      iface_list = interfaces if isinstance(interfaces, list) else [interfaces]
      for iface in iface_list:
        iface.bind(self)
        self._interfaces.append(iface)

    if gateway is not None:
      gateway._bind_agent(self)
      for iface in self._interfaces:
        if iface not in gateway._interfaces:
          gateway.add(iface)
      self._gateway = gateway

    # ── Tracing observer ──────────────────────────────────
    if self._trace_writer:
      from definable.agent.helpers.tracing import TracingObserver

      self._tracing_observer = TracingObserver(self._event_bus, self._trace_writer)

  # ═══════════════════════════════════════════════════════════
  # 4 OVERRIDABLE METHODS
  # ═══════════════════════════════════════════════════════════

  async def before_model_call(
    self,
    messages: List[Message],
    tools: Dict[str, Function],
    turn: int,
  ) -> List[Message]:
    """Prepare messages before the model call.

    Default: build system prompt on turn 0 (instructions + skills +
    knowledge + memory), refresh memory on subsequent turns, compress
    if needed.

    Override to customize prompt assembly, inject custom context, or
    apply input validation.
    """
    if turn == 0:
      from definable.agent.helpers.instructions import build_system_prompt
      from definable.agent.helpers.knowledge import retrieve_context
      from definable.agent.helpers.memory import load_working_memory

      system = build_system_prompt(self.instructions, self.skills)

      if self._knowledge:
        ctx = await retrieve_context(self._knowledge, messages)
        if ctx:
          system = f"{system}\n\n{ctx}" if system else ctx

      if self.memory:
        wm = await load_working_memory(self.memory, self._current_user_id)
        if wm:
          system = f"{system}\n\n{wm}" if system else wm

      if self.readers and self._current_readers_context:
        # Inject file content into last user message
        for i in range(len(messages) - 1, -1, -1):
          if messages[i].role == "user":
            original = messages[i].content or ""
            messages[i] = Message(
              role="user",
              content=f"{self._current_readers_context}\n\n{original}",
              images=messages[i].images,
              videos=messages[i].videos,
              audio=messages[i].audio,
            )
            break

      if system:
        messages.insert(0, Message(role="system", content=system))
      self._base_system_prompt = system
    else:
      # Refresh working memory on subsequent turns
      if self.memory:
        from definable.agent.helpers.memory import load_working_memory

        wm = await load_working_memory(self.memory, self._current_user_id)
        if wm and messages and messages[0].role == "system":
          messages[0] = Message(
            role="system",
            content=f"{self._base_system_prompt}\n\n{wm}" if self._base_system_prompt else wm,
          )

      # Compress if needed
      if self._compression_manager:
        from definable.agent.helpers.compression import compress_if_needed

        tools_dicts = self._build_tools_dicts(tools)
        await compress_if_needed(self._compression_manager, messages, tools_dicts, self.model)

    return messages

  async def call_model(
    self,
    messages: List[Message],
    tools: Dict[str, Function],
    *,
    streaming: bool = False,
    output_schema: Optional[type] = None,
  ) -> tuple:
    """Call the LLM. Returns (content, tool_calls, metrics, parsed).

    Default: delegates to model.ainvoke/ainvoke_stream with retry.
    Override to add custom model routing, caching, or fallback logic.
    """
    tools_dicts = self._build_tools_dicts(tools) if tools else None
    max_retries = self.config.max_retries if self.config.retry_transient_errors else 0
    backoff_base = self.config.retry_backoff_base

    for attempt in range(max_retries + 1):
      try:
        assistant_msg = Message(role="assistant")

        if streaming:
          # Streaming path — yields RunContentEvent via _streaming_events list
          content = ""
          tool_calls: list = []
          metrics = None
          parsed = None
          reasoning = ""

          async for chunk in self.model.ainvoke_stream(
            messages=messages,
            assistant_message=assistant_msg,
            tools=tools_dicts,
            response_format=output_schema,
          ):
            if hasattr(chunk, "reasoning_content") and chunk.reasoning_content:
              reasoning += chunk.reasoning_content
              self._streaming_events.append(("reasoning_delta", chunk.reasoning_content))
            if hasattr(chunk, "content") and chunk.content:
              content += chunk.content
              self._streaming_events.append(("content", chunk.content))
            if hasattr(chunk, "tool_calls") and chunk.tool_calls:
              tool_calls = _merge_tool_call_deltas(tool_calls, chunk.tool_calls)
            if hasattr(chunk, "response_usage") and chunk.response_usage is not None:
              metrics = chunk.response_usage if metrics is None else metrics + chunk.response_usage
            if hasattr(chunk, "parsed") and chunk.parsed is not None:
              parsed = chunk.parsed

          # Parse structured output if not parsed from chunks
          if parsed is None and output_schema and content:
            parsed = self._try_parse_output(content, output_schema)

          # Store assistant message
          assistant_final = Message(
            role="assistant",
            content=content or None,
            tool_calls=tool_calls or None,
          )
          if reasoning:
            assistant_final.reasoning_content = reasoning
          if metrics:
            assistant_final.metrics = metrics
          messages.append(assistant_final)

          return content, tool_calls, metrics, parsed

        else:
          # Non-streaming path
          response = await self.model.ainvoke(
            messages=messages,
            assistant_message=assistant_msg,
            tools=tools_dicts,
            response_format=output_schema,
          )

          # Build assistant message
          assistant_final = Message(
            role="assistant",
            content=response.content,
            tool_calls=response.tool_calls or None,
          )
          if hasattr(response, "reasoning_content") and response.reasoning_content:
            assistant_final.reasoning_content = response.reasoning_content
          if hasattr(response, "response_usage") and response.response_usage is not None:
            assistant_final.metrics = response.response_usage
          messages.append(assistant_final)

          parsed = getattr(response, "parsed", None)
          # MagicMock attributes are truthy but not real parsed values
          if parsed is not None and not isinstance(parsed, BaseModel):
            parsed = None
          if parsed is None and output_schema and response.content:
            parsed = self._try_parse_output(response.content, output_schema)

          return (
            response.content or "",
            response.tool_calls or [],
            response.response_usage if hasattr(response, "response_usage") else None,
            parsed,
          )

      except (ConnectionError, TimeoutError, OSError):
        if not self.config.retry_transient_errors or attempt >= max_retries:
          raise
        delay = min(backoff_base * (2**attempt), 60.0)
        await asyncio.sleep(delay)

    raise RuntimeError("Exhausted retries")  # pragma: no cover

  async def after_model_call(
    self,
    content: str,
    tool_calls: list,
    metrics: Optional[Metrics],
    parsed: Any,
    turn: int,
  ) -> tuple:
    """Validate/transform model output. Returns (content, tool_calls, metrics, parsed).

    Default: pass through. Override to add output guardrails,
    content filtering, or response transformation.
    """
    return content, tool_calls, metrics, parsed

  async def execute_tool(
    self,
    tool_call: dict,
    tools: Dict[str, Function],
  ) -> ToolResult:
    """Execute a single tool call. Returns ToolResult.

    Default: look up the function, call it, handle HITL flags.
    Override to add sandboxing, custom error handling, or tool-level
    authorization.
    """
    fn_name = tool_call.get("function", {}).get("name", "unknown")
    fn = tools.get(fn_name)
    function_call = get_function_call_for_tool_call(tool_call, tools)

    # Build ToolExecution for tracking
    tool_execution = ToolExecution(
      tool_call_id=tool_call.get("id"),
      tool_name=fn_name,
      tool_args=function_call.arguments if function_call else None,
    )

    # Emit ToolCallStarted
    started_event = ToolCallStartedEvent(
      run_id=self._current_run_id,
      session_id=self._current_session_id,
      agent_id=self.agent_id,
      agent_name=self.agent_name,
      tool=_dc_replace(tool_execution),
    )
    await self._event_bus.emit(started_event)

    # ── HITL: requires_confirmation ──
    if fn and fn.requires_confirmation:
      tool_execution.requires_confirmation = True
      requirement = RunRequirement(tool_execution)
      return ToolResult(
        tool_call_id=tool_call.get("id"),
        tool_name=fn_name,
        is_paused=True,
        should_stop=bool(fn.stop_after_tool_call),
        requirement=requirement,
        tool_execution=tool_execution,
      )

    # ── HITL: requires_user_input ──
    if fn and fn.requires_user_input:
      tool_execution.requires_user_input = True
      tool_execution.user_input_schema = fn.user_input_schema
      requirement = RunRequirement(tool_execution)
      return ToolResult(
        tool_call_id=tool_call.get("id"),
        tool_name=fn_name,
        is_paused=True,
        should_stop=bool(fn.stop_after_tool_call),
        requirement=requirement,
        tool_execution=tool_execution,
      )

    # ── HITL: external_execution ──
    if fn and fn.external_execution:
      tool_execution.external_execution_required = True
      requirement = RunRequirement(tool_execution)
      return ToolResult(
        tool_call_id=tool_call.get("id"),
        tool_name=fn_name,
        is_paused=True,
        should_stop=bool(fn.stop_after_tool_call),
        requirement=requirement,
        tool_execution=tool_execution,
      )

    # ── Execute ──
    if function_call:
      try:
        result_obj = await function_call.aexecute()
        if result_obj.status == "success":
          tool_execution.result = await self._resolve_tool_result(result_obj.result, fn_name, tool_call.get("id"))
        else:
          tool_execution.result = str(result_obj.error)
        tool_execution.tool_call_error = result_obj.status == "failure"
      except Exception as exc:
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
          )
        raise
    else:
      tool_execution.result = f"Tool '{fn_name}' not found"
      tool_execution.tool_call_error = True

    self._all_tool_executions.append(tool_execution)

    # Emit ToolCallCompleted
    completed_event = ToolCallCompletedEvent(
      run_id=self._current_run_id,
      session_id=self._current_session_id,
      agent_id=self.agent_id,
      agent_name=self.agent_name,
      tool=tool_execution,
      content=tool_execution.result,
    )
    await self._event_bus.emit(completed_event)

    return ToolResult(
      tool_call_id=tool_call.get("id"),
      tool_name=fn_name,
      result=tool_execution.result if not tool_execution.tool_call_error else None,
      error=tool_execution.result if tool_execution.tool_call_error else None,
      should_stop=bool(fn and fn.stop_after_tool_call),
      tool_execution=tool_execution,
    )

  # ═══════════════════════════════════════════════════════════
  # THE CORE LOOP
  # ═══════════════════════════════════════════════════════════

  async def _run_loop(
    self,
    messages: List[Message],
    tools: Dict[str, Function],
    *,
    streaming: bool = False,
    cancel: Optional[CancelToken] = None,
    output_schema: Optional[type] = None,
  ) -> AsyncIterator[RunOutputEvent]:
    """The agentic loop. Yields events as they occur.

    Calls the 4 overridable methods in sequence:
    before_model_call → call_model → after_model_call → execute_tool
    """
    max_tool_rounds = self.config.max_tool_rounds
    total_metrics: Optional[Metrics] = None
    final_content: Optional[str] = None
    final_parsed: Any = None
    self._all_tool_executions: list[Any] = []
    self._streaming_events: list[Any] = []

    try:
      for turn in range(max_tool_rounds):
        if cancel:
          cancel.check()

        # 1. Prepare messages
        messages = await self.before_model_call(messages, tools, turn)

        # 2. Call model
        started_evt = ModelCallStartedEvent(
          run_id=self._current_run_id,
          session_id=self._current_session_id,
          agent_id=self.agent_id,
          agent_name=self.agent_name,
          turn=turn + 1,
          messages=list(messages),
          tool_definitions=self._build_tools_dicts(tools) if tools else None,
          response_format=output_schema,
          model_id=self.model.id,
          model_provider=getattr(self.model, "provider", "") or "",
        )
        await self._event_bus.emit(started_evt)

        self._streaming_events = []
        content, tool_calls, metrics, parsed = await self.call_model(messages, tools, streaming=streaming, output_schema=output_schema)

        # Yield streaming events collected during call_model
        reasoning_started = False
        reasoning_completed = False
        for evt_type, evt_data in self._streaming_events:
          if evt_type == "reasoning_delta":
            if not reasoning_started:
              reasoning_started = True
              re = ReasoningStartedEvent(
                run_id=self._current_run_id,
                session_id=self._current_session_id,
                agent_id=self.agent_id,
                agent_name=self.agent_name,
              )
              await self._event_bus.emit(re)
              yield re
            rde = ReasoningContentDeltaEvent(
              run_id=self._current_run_id,
              session_id=self._current_session_id,
              agent_id=self.agent_id,
              agent_name=self.agent_name,
              reasoning_content=evt_data,
            )
            await self._event_bus.emit(rde)
            yield rde
          elif evt_type == "content":
            if reasoning_started and not reasoning_completed:
              reasoning_completed = True
              rce = ReasoningCompletedEvent(
                run_id=self._current_run_id,
                session_id=self._current_session_id,
                agent_id=self.agent_id,
                agent_name=self.agent_name,
              )
              await self._event_bus.emit(rce)
              yield rce
            ce = RunContentEvent(
              run_id=self._current_run_id,
              session_id=self._current_session_id,
              agent_id=self.agent_id,
              agent_name=self.agent_name,
              content=evt_data,
            )
            await self._event_bus.emit(ce)
            yield ce

        if reasoning_started and not reasoning_completed:
          rce = ReasoningCompletedEvent(
            run_id=self._current_run_id,
            session_id=self._current_session_id,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
          )
          await self._event_bus.emit(rce)
          yield rce

        # 3. Validate output
        content, tool_calls, metrics, parsed = await self.after_model_call(content, tool_calls, metrics, parsed, turn)

        completed_evt = ModelCallCompletedEvent(
          run_id=self._current_run_id,
          session_id=self._current_session_id,
          agent_id=self.agent_id,
          agent_name=self.agent_name,
          turn=turn + 1,
          content="" if streaming else (content or None),
          tool_calls=tool_calls or None,
          metrics=metrics,
          model_id=self.model.id,
        )
        await self._event_bus.emit(completed_evt)

        if metrics is not None:
          total_metrics = metrics if total_metrics is None else total_metrics + metrics

        # 4. No tools → done
        if not tool_calls:
          final_content = content
          final_parsed = parsed
          break

        # 5. Execute tools (parallel + sequential)
        parallel_calls: list[dict] = []
        sequential_calls: list[dict] = []
        for tc in tool_calls:
          fn_name = tc.get("function", {}).get("name", "")
          fn = tools.get(fn_name)
          if fn and fn.sequential:
            sequential_calls.append(tc)
          else:
            parallel_calls.append(tc)

        all_results: list[ToolResult] = []

        # Parallel tools
        if parallel_calls:
          results = await asyncio.gather(
            *[self.execute_tool(tc, tools) for tc in parallel_calls],
            return_exceptions=True,
          )
          for i, r in enumerate(results):
            if isinstance(r, BaseException):
              tc = parallel_calls[i]
              fn_name = tc.get("function", {}).get("name", "unknown")
              messages.append(
                Message(
                  role="tool",
                  content=f"Error: {r}",
                  tool_call_id=tc.get("id"),
                  name=fn_name,
                )
              )
              all_results.append(
                ToolResult(
                  tool_call_id=tc.get("id"),
                  tool_name=fn_name,
                  error=str(r),
                )
              )
            else:
              all_results.append(r)

        # Sequential tools
        for tc in sequential_calls:
          if cancel:
            cancel.check()
          result = await self.execute_tool(tc, tools)
          all_results.append(result)

        # Add successful tool results to messages
        for r in all_results:
          if r.tool_execution and not r.is_paused:
            messages.append(
              Message(
                role="tool",
                content=r.tool_execution.result or "",
                tool_call_id=r.tool_call_id,
                name=r.tool_name,
              )
            )

        # Check stop_after_tool_call
        if any(r.should_stop for r in all_results):
          final_content = content
          final_parsed = parsed
          break

        # Check HITL pause
        paused = [r for r in all_results if r.is_paused]
        if paused:
          requirements = [r.requirement for r in paused if r.requirement is not None]
          paused_tools = [r.tool_execution for r in paused if r.tool_execution is not None]
          pe = RunPausedEvent(
            run_id=self._current_run_id,
            session_id=self._current_session_id,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            requirements=requirements,
            tools=paused_tools,
          )
          await self._event_bus.emit(pe)
          yield pe
          return

      # Yield RunCompleted
      completed = RunCompletedEvent(
        run_id=self._current_run_id,
        session_id=self._current_session_id,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
        content=final_content,
        parsed=final_parsed,
        metrics=total_metrics,
      )
      await self._event_bus.emit(completed)
      yield completed

    except Cancelled:
      raise
    except Exception as e:
      err = RunErrorEvent(
        run_id=self._current_run_id,
        session_id=self._current_session_id,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
        error_type=type(e).__name__,
        content=str(e),
      )
      await self._event_bus.emit(err)
      yield err
      raise

  # ═══════════════════════════════════════════════════════════
  # PUBLIC RUN METHODS
  # ═══════════════════════════════════════════════════════════

  def run(
    self,
    instruction: Union[str, Message, List[Message]],
    *,
    messages: Optional[List[Message]] = None,
    session_id: Optional[str] = None,
    run_id: Optional[str] = None,
    user_id: Optional[str] = None,
    images: Optional[List[Image]] = None,
    videos: Optional[List[Video]] = None,
    audio: Optional[List[Audio]] = None,
    files: Optional[List[File]] = None,
    output_schema: Optional[Type[BaseModel]] = None,
  ) -> RunOutput:
    """Synchronous run with multi-turn conversation support."""
    try:
      loop = asyncio.get_running_loop()
    except RuntimeError:
      loop = None

    if loop and loop.is_running():
      import concurrent.futures

      with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(
          asyncio.run,
          self.arun(
            instruction,
            messages=messages,
            session_id=session_id,
            run_id=run_id,
            user_id=user_id,
            images=images,
            videos=videos,
            audio=audio,
            files=files,
            output_schema=output_schema,
          ),
        )
        return future.result()
    else:
      new_loop = asyncio.new_event_loop()
      asyncio.set_event_loop(new_loop)
      try:
        return new_loop.run_until_complete(
          self.arun(
            instruction,
            messages=messages,
            session_id=session_id,
            run_id=run_id,
            user_id=user_id,
            images=images,
            videos=videos,
            audio=audio,
            files=files,
            output_schema=output_schema,
          )
        )
      finally:
        try:
          pending = asyncio.all_tasks(new_loop)
          for task in pending:
            task.cancel()
          if pending:
            new_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
          new_loop.run_until_complete(new_loop.shutdown_asyncgens())
          if hasattr(new_loop, "shutdown_default_executor"):
            new_loop.run_until_complete(new_loop.shutdown_default_executor())
        except Exception:
          pass
        finally:
          new_loop.close()

  async def arun(
    self,
    instruction: Union[str, Message, List[Message]],
    *,
    messages: Optional[List[Message]] = None,
    session_id: Optional[str] = None,
    run_id: Optional[str] = None,
    user_id: Optional[str] = None,
    images: Optional[List[Image]] = None,
    videos: Optional[List[Video]] = None,
    audio: Optional[List[Audio]] = None,
    files: Optional[List[File]] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    cancellation_token: Optional[CancelToken] = None,
  ) -> RunOutput:
    """Async run — the primary entry point."""
    # Validate output_schema
    if output_schema is not None:
      if not isinstance(output_schema, type) or not issubclass(output_schema, BaseModel):
        raise TypeError(f"output_schema must be a Pydantic BaseModel subclass, got {output_schema!r}.")

    # Ensure toolkits are initialized
    await self._ensure_toolkits_initialized()

    # Prepare run state
    _run_id = run_id or str(uuid4())
    _session_id = session_id or self.session_id
    new_messages = self._normalize_instruction(instruction, images, videos, audio, files)
    all_messages = (messages or []) + new_messages

    # Transcribe audio
    await self._transcribe_audio(new_messages)

    # Extract file content
    self._current_readers_context = await self._readers_extract(new_messages)

    # Set current run context for overridable methods
    self._current_run_id = _run_id
    self._current_session_id = _session_id
    self._current_user_id = user_id

    # Prepare context
    context = RunContext(
      run_id=_run_id,
      session_id=_session_id,
      user_id=user_id,
      dependencies=self.config.dependencies,
      session_state=dict(self.config.session_state or {}),
      output_schema=output_schema,
    )

    # Prepare tools
    tools = self._prepare_tools_for_run(context)

    # Build run input
    run_input = RunInput(
      input_content=instruction,
      images=images,
      videos=videos,
      audios=audio,
      files=files,
    )

    # Fire before hooks
    await self._fire_before_hooks(context)

    # Emit RunStarted
    started = RunStartedEvent(
      run_id=_run_id,
      session_id=_session_id,
      agent_id=self.agent_id,
      agent_name=self.agent_name,
      model=self.model.id,
      model_provider=getattr(self.model, "provider", None) or "",
      run_input=run_input,
    )
    await self._event_bus.emit(started)

    # Run the loop — use a mutable list wrapper so before_model_call
    # mutations are visible here for output messages
    self._loop_messages = list(all_messages)

    try:
      final_content: Optional[str] = None
      final_parsed: Any = None
      final_metrics: Optional[Metrics] = None
      paused_event: Optional[RunPausedEvent] = None

      async for event in self._run_loop(
        self._loop_messages,
        tools,
        streaming=False,
        cancel=cancellation_token,
        output_schema=output_schema,
      ):
        if isinstance(event, RunCompletedEvent):
          final_content = event.content
          final_parsed = event.parsed
          final_metrics = event.metrics
        elif isinstance(event, RunPausedEvent):
          paused_event = event

      # Build output messages (excluding system)
      output_messages = [m for m in self._loop_messages if m.role != "system"]

      if paused_event:
        result = RunOutput(
          run_id=_run_id,
          session_id=_session_id,
          agent_id=self.agent_id,
          agent_name=self.agent_name,
          input=run_input,
          tools=self._all_tool_executions or None,
          messages=output_messages,
          model=self.model.id,
          model_provider=getattr(self.model, "provider", None),
          status=RunStatus.paused,
          session_state=context.session_state,
          requirements=paused_event.requirements,
        )
      else:
        result = RunOutput(
          run_id=_run_id,
          session_id=_session_id,
          agent_id=self.agent_id,
          agent_name=self.agent_name,
          input=run_input,
          content=final_content,
          parsed=final_parsed,
          tools=self._all_tool_executions or None,
          metrics=final_metrics,
          messages=output_messages,
          model=self.model.id,
          model_provider=getattr(self.model, "provider", None),
          status=RunStatus.completed,
          session_state=context.session_state,
        )

    except Cancelled:
      cancelled_event = RunCancelledEvent(
        run_id=_run_id,
        session_id=_session_id,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
        reason="Cancelled via CancelToken",
      )
      await self._event_bus.emit(cancelled_event)
      result = RunOutput(
        run_id=_run_id,
        session_id=_session_id,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
        status=RunStatus.cancelled,
        model=self.model.id,
        model_provider=getattr(self.model, "provider", None),
      )

    # Fire after hooks
    await self._fire_after_hooks(result)
    return result

  def run_stream(
    self,
    instruction: Union[str, Message, List[Message]],
    *,
    messages: Optional[List[Message]] = None,
    session_id: Optional[str] = None,
    run_id: Optional[str] = None,
    user_id: Optional[str] = None,
    images: Optional[List[Image]] = None,
    output_schema: Optional[Type[BaseModel]] = None,
  ) -> Iterator[RunOutputEvent]:
    """Sync streaming run — yields events as they occur."""
    import queue
    import threading
    import time as _time

    event_queue: queue.Queue[Union[RunOutputEvent, Exception, None]] = queue.Queue()
    stop_event = threading.Event()
    loop_ready = threading.Event()
    loop_holder: Dict[str, asyncio.AbstractEventLoop] = {}

    timeout_seconds = self.config.stream_timeout_seconds or 300.0
    deadline = _time.monotonic() + timeout_seconds if timeout_seconds > 0 else None

    def run_async_stream() -> None:
      loop = asyncio.new_event_loop()
      asyncio.set_event_loop(loop)
      loop_holder["loop"] = loop
      loop_ready.set()

      async def stream_to_queue() -> None:
        try:
          async for event in self.arun_stream(
            instruction,
            messages=messages,
            session_id=session_id,
            run_id=run_id,
            user_id=user_id,
            images=images,
            output_schema=output_schema,
          ):
            if stop_event.is_set():
              break
            event_queue.put(event)
        except Exception as e:
          event_queue.put(e)
        finally:
          event_queue.put(None)

      try:
        loop.run_until_complete(stream_to_queue())
      finally:
        with contextlib.suppress(Exception):
          pending = asyncio.all_tasks(loop)
          for task in pending:
            task.cancel()
          if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
          loop.run_until_complete(loop.shutdown_asyncgens())
          if hasattr(loop, "shutdown_default_executor"):
            loop.run_until_complete(loop.shutdown_default_executor())
        with contextlib.suppress(Exception):
          loop.close()

    thread = threading.Thread(target=run_async_stream, daemon=True)
    thread.start()

    try:
      loop_ready.wait(timeout=1.0)
      while True:
        remaining = (deadline - _time.monotonic()) if deadline else None
        try:
          item = event_queue.get(timeout=remaining)
        except queue.Empty:
          stop_event.set()
          raise TimeoutError(f"Stream timed out after {timeout_seconds:.0f} seconds.")
        if item is None:
          break
        if isinstance(item, Exception):
          raise item
        yield item
    finally:
      stop_event.set()
      thread.join(timeout=5.0)

  async def arun_stream(
    self,
    instruction: Union[str, Message, List[Message]],
    *,
    messages: Optional[List[Message]] = None,
    session_id: Optional[str] = None,
    run_id: Optional[str] = None,
    user_id: Optional[str] = None,
    images: Optional[List[Image]] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    cancellation_token: Optional[CancelToken] = None,
  ) -> AsyncIterator[RunOutputEvent]:
    """Async streaming run — yields events as they occur."""
    await self._ensure_toolkits_initialized()

    _run_id = run_id or str(uuid4())
    _session_id = session_id or self.session_id
    new_messages = self._normalize_instruction(instruction, images)
    all_messages = (messages or []) + new_messages

    await self._transcribe_audio(new_messages)
    self._current_readers_context = await self._readers_extract(new_messages)
    self._current_run_id = _run_id
    self._current_session_id = _session_id
    self._current_user_id = user_id

    context = RunContext(
      run_id=_run_id,
      session_id=_session_id,
      user_id=user_id,
      dependencies=self.config.dependencies,
      session_state=dict(self.config.session_state or {}),
      output_schema=output_schema,
    )
    tools = self._prepare_tools_for_run(context)

    try:
      async for event in self._run_loop(
        all_messages,
        tools,
        streaming=True,
        cancel=cancellation_token,
        output_schema=output_schema,
      ):
        yield event  # type: ignore[misc]
    except Cancelled:
      cancelled_event = RunCancelledEvent(
        run_id=_run_id,
        session_id=_session_id,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
        reason="Cancelled via CancelToken",
      )
      await self._event_bus.emit(cancelled_event)
      yield cancelled_event
    except Exception as e:
      error_event = RunErrorEvent(
        run_id=_run_id,
        session_id=_session_id,
        agent_id=self.agent_id,
        error_type=type(e).__name__,
        content=str(e),
      )
      await self._event_bus.emit(error_event)
      yield error_event
      raise

  async def continue_run(
    self,
    *,
    run_output: RunOutput,
    cancellation_token: Optional[CancelToken] = None,
  ) -> RunOutput:
    """Resume a paused run after HITL requirements are resolved."""
    if not run_output.is_paused:
      raise ValueError("RunOutput is not paused — nothing to continue")
    unresolved = run_output.active_requirements
    if unresolved:
      raise ValueError(f"{len(unresolved)} requirement(s) still unresolved.")

    msgs = run_output.messages or []
    for req in run_output.requirements or []:
      te = req.tool_execution
      if te is None:
        continue
      if req.confirmation is False:
        msgs.append(
          Message(
            role="tool",
            content=_REJECTED_MSG,
            tool_call_id=te.tool_call_id,
            name=te.tool_name,
          )
        )
      elif req.confirmation is True:
        fn = self._tools_dict.get(te.tool_name)  # type: ignore[arg-type]
        if fn:
          function_call = get_function_call_for_tool_call(
            {"id": te.tool_call_id, "type": "function", "function": {"name": te.tool_name, "arguments": json.dumps(te.tool_args or {})}},
            self._tools_dict,
          )
          if function_call:
            result_obj = await function_call.aexecute()
            msgs.append(
              Message(
                role="tool",
                content=str(result_obj.result) if result_obj.status == "success" else str(result_obj.error),
                tool_call_id=te.tool_call_id,
                name=te.tool_name,
              )
            )
          else:
            msgs.append(Message(role="tool", content=f"Tool '{te.tool_name}' not found", tool_call_id=te.tool_call_id, name=te.tool_name))
        else:
          msgs.append(Message(role="tool", content=f"Tool '{te.tool_name}' not found", tool_call_id=te.tool_call_id, name=te.tool_name))
      elif req.external_execution_result is not None:
        msgs.append(
          Message(
            role="tool",
            content=req.external_execution_result,
            tool_call_id=te.tool_call_id,
            name=te.tool_name,
          )
        )

    return await self.arun(
      instruction=msgs[-1] if msgs and msgs[-1].role == "user" else "Continue.",
      messages=msgs,
      session_id=run_output.session_id,
      cancellation_token=cancellation_token,
    )

  async def continue_run_stream(
    self,
    *,
    run_output: RunOutput,
    cancellation_token: Optional[CancelToken] = None,
  ) -> AsyncIterator[RunOutputEvent]:
    """Streaming variant of continue_run."""
    if not run_output.is_paused:
      raise ValueError("RunOutput is not paused — nothing to continue")
    unresolved = run_output.active_requirements
    if unresolved:
      raise ValueError(f"{len(unresolved)} requirement(s) still unresolved.")

    msgs = run_output.messages or []
    for req in run_output.requirements or []:
      te = req.tool_execution
      if te is None:
        continue
      if req.confirmation is False:
        msgs.append(Message(role="tool", content=_REJECTED_MSG, tool_call_id=te.tool_call_id, name=te.tool_name))
      elif req.confirmation is True:
        fn = self._tools_dict.get(te.tool_name)  # type: ignore[arg-type]
        if fn:
          function_call = get_function_call_for_tool_call(
            {"id": te.tool_call_id, "type": "function", "function": {"name": te.tool_name, "arguments": json.dumps(te.tool_args or {})}},
            self._tools_dict,
          )
          if function_call:
            result_obj = await function_call.aexecute()
            msgs.append(
              Message(
                role="tool",
                content=str(result_obj.result) if result_obj.status == "success" else str(result_obj.error),
                tool_call_id=te.tool_call_id,
                name=te.tool_name,
              )
            )
      elif req.external_execution_result is not None:
        msgs.append(Message(role="tool", content=req.external_execution_result, tool_call_id=te.tool_call_id, name=te.tool_name))

    async for event in self.arun_stream(
      instruction=msgs[-1] if msgs and msgs[-1].role == "user" else "Continue.",
      messages=msgs,
      session_id=run_output.session_id,
      cancellation_token=cancellation_token,
    ):
      yield event

  # ═══════════════════════════════════════════════════════════
  # PROPERTIES
  # ═══════════════════════════════════════════════════════════

  @property
  def agent_id(self) -> str:
    return self.config.agent_id or str(id(self))

  @property
  def agent_name(self) -> str:
    return self.config.agent_name or self.__class__.__name__

  @property
  def name(self) -> str:
    return self.agent_name

  @property
  def tool_names(self) -> List[str]:
    return list(self._tools_dict.keys())

  @property
  def events(self) -> EventBus:
    return self._event_bus

  @property
  def auth(self) -> Optional[Any]:
    return self._auth

  @auth.setter
  def auth(self, provider: Optional[Any]) -> None:
    self._auth = provider

  @property
  def interfaces(self) -> List["BaseInterface"]:
    """Registered interfaces (read-only copy)."""
    return list(self._interfaces)

  @property
  def gateway(self) -> Optional["InterfaceGateway"]:
    """Gateway instance if configured."""
    return self._gateway

  def add_interface(self, interface: "BaseInterface") -> "Agent":
    """Legacy: attach an interface. Prefer ``interfaces=`` constructor param."""
    import warnings

    warnings.warn(
      "agent.add_interface() is deprecated. Pass interfaces= to Agent() constructor.",
      DeprecationWarning,
      stacklevel=2,
    )
    interface.bind(self)
    self._interfaces.append(interface)
    return self

  def create_gateway(self, **kwargs: Any) -> "InterfaceGateway":
    """Legacy: create and attach a gateway. Prefer ``gateway=`` constructor param."""
    import warnings

    warnings.warn(
      "agent.create_gateway() is deprecated. Pass gateway= to Agent() constructor.",
      DeprecationWarning,
      stacklevel=2,
    )
    from definable.agent.interface.gateway import InterfaceGateway as _InterfaceGateway

    gw = _InterfaceGateway(**kwargs)
    gw._bind_agent(self)
    for iface in self._interfaces:
      if iface not in gw._interfaces:
        gw.add(iface)
    self._gateway = gw
    return gw

  # ═══════════════════════════════════════════════════════════
  # HOOKS
  # ═══════════════════════════════════════════════════════════

  def before_request(self, fn: Optional[Callable] = None) -> Callable:
    """Register a hook that fires before every arun() call."""
    if fn is not None:
      self._before_hooks.append(fn)
      return fn

    def decorator(func: Callable) -> Callable:
      self._before_hooks.append(func)
      return func

    return decorator

  def after_response(self, fn: Optional[Callable] = None) -> Callable:
    """Register a hook that fires after every arun() call."""
    if fn is not None:
      self._after_hooks.append(fn)
      return fn

    def decorator(func: Callable) -> Callable:
      self._after_hooks.append(func)
      return func

    return decorator

  def use(self, middleware: Any) -> "Agent":
    """Add middleware (kept for backward compat — wraps arun)."""
    # Middleware is no longer a first-class concept; use before_request/after_response
    # or override methods instead. This stub prevents AttributeError.
    return self

  # ═══════════════════════════════════════════════════════════
  # LIFECYCLE
  # ═══════════════════════════════════════════════════════════

  def __enter__(self) -> "Agent":
    self._start()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    self._shutdown()

  async def __aenter__(self) -> "Agent":
    self._start()
    await self._ensure_toolkits_initialized()
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    await self._ashutdown()

  def _start(self) -> None:
    if self._started:
      return
    self._started = True

  def _shutdown(self) -> None:
    for skill in self.skills:
      with contextlib.suppress(Exception):
        skill.teardown()
    if self._trace_writer:
      self._trace_writer.shutdown()
    self._sync_close_async_resources()
    self._started = False

  def _sync_close_async_resources(self) -> None:
    async def _cleanup() -> None:
      for toolkit in self._agent_owned_toolkits:
        with contextlib.suppress(Exception):
          await toolkit.shutdown()
      self._agent_owned_toolkits.clear()
      if self.memory:
        with contextlib.suppress(Exception):
          await self.memory.close()

    try:
      asyncio.get_running_loop()
      from definable.utils.log import log_warning

      log_warning(
        "Agent._shutdown() cannot close async resources from inside a running event loop. Use 'async with Agent(...)' or await agent._ashutdown()."
      )
    except RuntimeError:
      with contextlib.suppress(Exception):
        asyncio.run(_cleanup())

  async def _ashutdown(self) -> None:
    for toolkit in self._agent_owned_toolkits:
      with contextlib.suppress(Exception):
        await toolkit.shutdown()
    self._agent_owned_toolkits.clear()
    if self.memory:
      with contextlib.suppress(Exception):
        await self.memory.close()
    for skill in self.skills:
      with contextlib.suppress(Exception):
        skill.teardown()
    if self._trace_writer:
      self._trace_writer.shutdown()
    self._started = False

  async def _ensure_toolkits_initialized(self) -> None:
    async with self._toolkit_init_lock:
      needs_refresh = False
      for toolkit in self.toolkits:
        if isinstance(toolkit, AsyncLifecycleToolkit) and not toolkit._initialized:
          try:
            await toolkit.initialize()
            self._agent_owned_toolkits.append(toolkit)
            needs_refresh = True
          except Exception as e:
            from definable.utils.log import log_warning

            log_warning(f"Toolkit {toolkit!r} init failed (non-fatal): {e}")
      if needs_refresh:
        self._tools_dict = self._flatten_tools()

  # ═══════════════════════════════════════════════════════════
  # TRIGGERS
  # ═══════════════════════════════════════════════════════════

  @property
  def triggers(self) -> List[Any]:
    return list(self._triggers)

  def on(self, trigger: Any) -> Callable:
    """Register a trigger handler."""

    def decorator(fn: Callable) -> Callable:
      trigger.handler = fn
      trigger.agent = self
      self._triggers.append(trigger)
      return fn

    return decorator

  def emit(self, event_name: str, data: Optional[dict] = None) -> None:
    """Fire all EventTriggers matching event_name."""
    from definable.agent.trigger.base import TriggerEvent
    from definable.agent.trigger.event import EventTrigger
    from definable.agent.trigger.executor import TriggerExecutor

    matching = [t for t in self._triggers if isinstance(t, EventTrigger) and t.event_name == event_name]
    if not matching:
      return

    event = TriggerEvent(
      body=data if isinstance(data, dict) else {"data": data} if data is not None else None,
      source=f"event({event_name})",
    )
    executor = TriggerExecutor(self)

    try:
      loop = asyncio.get_running_loop()
      for trigger in matching:
        loop.create_task(executor.execute(trigger, event))
    except RuntimeError:
      pass

  # ═══════════════════════════════════════════════════════════
  # SERVE
  # ═══════════════════════════════════════════════════════════

  async def aserve(
    self,
    *interfaces: "BaseInterface",
    gateway: Optional["InterfaceGateway"] = None,
    name: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    enable_server: Optional[bool] = None,
    dev: bool = False,
  ) -> None:
    """Start the full agent runtime."""
    import warnings

    if interfaces:
      warnings.warn("Passing interfaces to aserve() is deprecated.", DeprecationWarning, stacklevel=2)
    if gateway is not None:
      warnings.warn("Passing gateway to aserve() is deprecated.", DeprecationWarning, stacklevel=2)

    from definable.agent.runtime.runner import AgentRuntime

    resolved_gateway = gateway or self._gateway
    all_interfaces = list(self._interfaces)
    for iface in interfaces:
      if iface.agent is None:
        iface.bind(self)
      if iface not in all_interfaces:
        all_interfaces.append(iface)

    if resolved_gateway is None and len(all_interfaces) >= 2:
      from definable.agent.interface.gateway import InterfaceGateway as _InterfaceGateway

      resolved_gateway = _InterfaceGateway()
      resolved_gateway._bind_agent(self)
      for iface in all_interfaces:
        resolved_gateway.add(iface)

    runtime = AgentRuntime(
      agent=self,
      interfaces=all_interfaces or None,
      host=host,
      port=port,
      enable_server=enable_server,
      name=name,
      dev=dev,
      gateway=resolved_gateway,
    )
    await runtime.start()

  def serve(
    self,
    *interfaces: "BaseInterface",
    gateway: Optional["InterfaceGateway"] = None,
    name: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    enable_server: Optional[bool] = None,
    dev: bool = False,
  ) -> None:
    """Sync entry point: start the full agent runtime."""
    import warnings

    if interfaces:
      warnings.warn("Passing interfaces to serve() is deprecated.", DeprecationWarning, stacklevel=2)
    if gateway is not None:
      warnings.warn("Passing gateway to serve() is deprecated.", DeprecationWarning, stacklevel=2)

    if dev:
      from definable.agent.runtime._dev import is_dev_child, run_dev_mode

      if not is_dev_child():
        run_dev_mode()
        return

    asyncio.run(
      self.aserve(
        *interfaces,
        gateway=gateway,
        name=name,
        host=host,
        port=port,
        enable_server=enable_server,
        dev=dev,
      )
    )

  # ═══════════════════════════════════════════════════════════
  # INTERNAL HELPERS
  # ═══════════════════════════════════════════════════════════

  def _emit_trace(self, event: BaseRunOutputEvent) -> None:
    """Emit event to trace writer (fire-and-forget)."""
    if self._trace_writer:
      with contextlib.suppress(Exception):
        self._trace_writer.write(event)

  async def _fire_before_hooks(self, context: RunContext) -> None:
    import inspect

    for hook in self._before_hooks:
      try:
        result = hook(context)
        if inspect.isawaitable(result):
          await result
      except Exception as e:
        from definable.utils.log import log_error

        log_error(f"before_request hook {hook.__name__} failed: {e}")

  async def _fire_after_hooks(self, output: RunOutput) -> None:
    import inspect

    for hook in self._after_hooks:
      try:
        result = hook(output)
        if inspect.isawaitable(result):
          await result
      except Exception as e:
        from definable.utils.log import log_error

        log_error(f"after_response hook {hook.__name__} failed: {e}")

  def _normalize_instruction(
    self,
    instruction: Union[str, Message, List[Message]],
    images: Optional[List[Image]] = None,
    videos: Optional[List[Video]] = None,
    audio: Optional[List[Audio]] = None,
    files: Optional[List[File]] = None,
  ) -> List[Message]:
    if isinstance(instruction, str):
      return [Message(role="user", content=instruction, images=images, videos=videos, audio=audio, files=files)]
    elif isinstance(instruction, Message):
      return [instruction]
    elif isinstance(instruction, list):
      return instruction
    raise TypeError(f"Unexpected instruction type: {type(instruction)}")

  async def _transcribe_audio(self, messages: List[Message]) -> None:
    if self._audio_transcriber is None:
      return
    for msg in messages:
      if not msg.audio:
        continue
      transcripts: List[str] = []
      for audio_item in msg.audio:
        if audio_item.transcript:
          transcripts.append(audio_item.transcript)
          continue
        audio_bytes = audio_item.get_content_bytes()
        if audio_bytes is None:
          continue
        mime = audio_item.mime_type or "audio/ogg"
        try:
          text = await self._audio_transcriber.atranscribe(audio_bytes, mime)
        except Exception:
          continue
        audio_item.transcript = text
        transcripts.append(text)
      if transcripts:
        transcript_text = "\n".join(transcripts)
        if msg.content:
          msg.content = f"{msg.content}\n\n{transcript_text}"
        else:
          msg.content = transcript_text
        msg.audio = None

  async def _readers_extract(self, new_messages: List[Message]) -> Optional[str]:
    """Extract text from files in new_messages."""
    if not self.readers:
      return None
    file_items: List[File] = []
    for msg in new_messages:
      if msg.files:
        file_items.extend(msg.files)
    if not file_items:
      return None
    try:
      results = await self.readers.aread_all(file_items)
    except Exception:
      return None
    blocks: List[str] = []
    for result in results:
      if result.error or not result.content:
        continue
      mime_attr = f' type="{result.mime_type}"' if result.mime_type else ""
      blocks.append(f'<file name="{result.filename}"{mime_attr}>\n{result.content}\n</file>')
    if blocks:
      return "<file_contents>\n" + "\n".join(blocks) + "\n</file_contents>"
    return None

  def _flatten_tools(self) -> Dict[str, Function]:
    """Flatten tools from skills, toolkits, and direct tools."""
    result: Dict[str, Function] = {}
    for skill in self.skills:
      try:
        skill_tools = skill.tools
      except Exception:
        skill_tools = []
      for fn in skill_tools:
        if skill.dependencies:
          existing_deps = getattr(fn, "_dependencies", None) or {}
          fn._dependencies = {**existing_deps, **skill.dependencies}
        result[fn.name] = fn
    for toolkit in self.toolkits:
      for fn in toolkit.tools:
        if toolkit.dependencies:
          existing_deps = getattr(fn, "_dependencies", None) or {}
          fn._dependencies = {**existing_deps, **toolkit.dependencies}
        result[fn.name] = fn
    for fn in self.tools:
      result[fn.name] = fn
    return result

  def _init_skills(self) -> None:
    seen_names: Dict[str, Skill] = {}
    for skill in self.skills:
      if skill.name in seen_names:
        from definable.utils.log import log_warning

        log_warning(f"Duplicate skill name '{skill.name}'")
      seen_names[skill.name] = skill
      try:
        skill.setup()
        skill._initialized = True
      except Exception as e:
        from definable.utils.log import log_error

        log_error(f"Skill '{skill.name}' setup() failed: {e}")

  def _init_tracing(self) -> Optional[TraceWriter]:
    if self._tracing_config and self._tracing_config.exporters:
      return TraceWriter(self._tracing_config)
    return None

  def _prepare_tools_for_run(self, context: RunContext) -> Dict[str, Function]:
    tools: Dict[str, Function] = {}
    for name, fn in self._tools_dict.items():
      tool_copy = fn.model_copy()
      tool_copy._run_context = context
      existing_deps = fn._dependencies or {}
      config_deps = self.config.dependencies or {}
      tool_copy._dependencies = {**existing_deps, **config_deps}
      tool_copy._session_state = context.session_state
      tools[name] = tool_copy

    # Inject memory tools when v2 Memory is configured
    if self.memory and hasattr(self.memory, "get_tools"):
      user_id = context.user_id
      session_id = context.session_id
      for tool_fn in self.memory.get_tools(user_id, session_id):
        tc = tool_fn.model_copy()
        tc._run_context = context
        tc._session_state = context.session_state
        tools[tc.name] = tc

    return tools

  @staticmethod
  def _build_tools_dicts(tools: Dict[str, Function]) -> Optional[List[Dict]]:
    if not tools:
      return None
    return [{"type": "function", "function": fn.to_dict()} for fn in tools.values()]

  @staticmethod
  def _try_parse_output(content: str, output_schema: type) -> Any:
    try:
      parsed_data = json.loads(content)
      if isinstance(output_schema, type) and issubclass(output_schema, BaseModel):
        return output_schema.model_validate(parsed_data)
    except Exception:
      pass
    return None

  async def _resolve_tool_result(self, result: Any, fn_name: str, tool_call_id: Optional[str]) -> str:
    """Consume a tool result, handling async/sync generators."""
    from inspect import isasyncgen, isgenerator

    if isasyncgen(result):
      chunks: list[str] = []
      idx = 0
      async for chunk in result:
        chunk_str = str(chunk)
        chunks.append(chunk_str)
        evt = ToolContentEvent(
          run_id=self._current_run_id,
          session_id=self._current_session_id,
          agent_id=self.agent_id,
          agent_name=self.agent_name,
          tool_name=fn_name,
          tool_call_id=tool_call_id,
          chunk=chunk_str,
          chunk_index=idx,
        )
        await self._event_bus.emit(evt)
        idx += 1
      return "".join(chunks)
    elif isgenerator(result):
      chunks = []
      idx = 0
      for chunk in result:
        chunk_str = str(chunk)
        chunks.append(chunk_str)
        idx += 1
      return "".join(chunks)
    return str(result)

  def _resolve_memory(self, memory: Any) -> Any:
    if memory is False or memory is None:
      return None
    if memory is True:
      from definable.memory.v2 import Memory as MemoryV2
      from definable.memory.v2 import SQLiteStore

      return MemoryV2(store=SQLiteStore(".definable/memory.db"))
    return memory

  def _resolve_knowledge(self, knowledge: Any) -> Optional["Knowledge"]:
    if knowledge is False or knowledge is None:
      return None
    if knowledge is True:
      raise ValueError("knowledge=True is not supported. Pass a path string or Knowledge instance.")
    if isinstance(knowledge, str):
      from definable.knowledge.base import Knowledge as _Knowledge

      return _Knowledge.from_path(knowledge)
    return knowledge

  @staticmethod
  def _resolve_tracing(tracing_param: Any, config: Optional[AgentConfig]) -> Optional["Tracing"]:
    from definable.agent.tracing.base import Tracing as _Tracing

    if tracing_param is False:
      return config.tracing if config else None
    if tracing_param is True:
      return _Tracing()
    if isinstance(tracing_param, _Tracing):
      return tracing_param
    return config.tracing if config else None

  def _resolve_compression(self, compression: Any) -> Optional["CompressionManager"]:
    from definable.agent.compression import Compression as _Compression

    if compression is True:
      return self._build_compression_manager(_Compression())
    if isinstance(compression, _Compression):
      return self._build_compression_manager(compression)
    return None

  def _build_compression_manager(self, compression: "Compression") -> "CompressionManager":
    from definable.agent.compression import CompressionManager

    compression_model = self.model
    if compression.model is not None:
      if isinstance(compression.model, str):
        from definable.model.utils import resolve_model_string

        compression_model = resolve_model_string(compression.model)
      else:
        compression_model = compression.model

    return CompressionManager(
      model=compression_model,
      compress_tool_results=True,
      compress_tool_results_limit=compression.tool_results_limit,
      compress_token_limit=compression.token_limit,
      compress_tool_call_instructions=compression.instructions,
      compress_single_result_size=compression.single_result_size,
    )

  @staticmethod
  def _init_readers(readers: Any) -> Optional["BaseReader"]:
    if readers is None or readers is False:
      return None
    if readers is True:
      from definable.reader import BaseReader

      return BaseReader()
    from definable.reader.parsers.base_parser import BaseParser

    if isinstance(readers, BaseParser):
      from definable.reader import BaseReader
      from definable.reader.registry import ParserRegistry

      registry = ParserRegistry(include_defaults=False)
      registry.register(readers)
      return BaseReader(registry=registry)
    return readers

  def export_manifest(self) -> Dict[str, Any]:
    """Export agent configuration as a manifest."""
    return {
      "name": self.agent_name,
      "model": self.model.id if self.model else None,
      "tools": [fn.to_dict() for fn in self._tools_dict.values()],
      "instructions": self.instructions,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _merge_tool_call_deltas(accumulated: list[dict], new_deltas: list[dict]) -> list[dict]:
  """Merge streaming tool call deltas into accumulated list."""
  for delta in new_deltas:
    idx = delta.get("index", len(accumulated))
    while len(accumulated) <= idx:
      accumulated.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
    entry = accumulated[idx]
    if delta.get("id"):
      entry["id"] = delta["id"]
    fn_delta = delta.get("function", {})
    if fn_delta.get("name"):
      entry["function"]["name"] += fn_delta["name"]
    if fn_delta.get("arguments"):
      entry["function"]["arguments"] += fn_delta["arguments"]
  return accumulated
