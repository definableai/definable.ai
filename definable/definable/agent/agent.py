"""Agent class - production-grade wrapper around model execution."""

import asyncio
import contextlib
import dataclasses
from typing import (
  TYPE_CHECKING,
  Any,
  AsyncIterator,
  Awaitable,
  Callable,
  Dict,
  Iterator,
  List,
  Literal,
  Optional,
  Protocol,
  Type,
  Union,
  runtime_checkable,
)
from uuid import uuid4

from definable.agent.cancellation import AgentCancelled, CancellationToken
from definable.agent.config import AgentConfig
from definable.agent.event_bus import EventBus
from definable.agent.loop import AgentLoop
from definable.agent.middleware import Middleware
from definable.agent.toolkit import Toolkit
from definable.agent.tracing.base import TraceWriter
from definable.media import Audio, File, Image, Video
from definable.model.message import Message
from definable.model.metrics import Metrics
from definable.model.response import ToolExecution
from definable.agent.events import (
  BaseRunOutputEvent,
  DeepResearchCompletedEvent,
  DeepResearchStartedEvent,
  FileReadCompletedEvent,
  FileReadStartedEvent,
  KnowledgeRetrievalCompletedEvent,
  KnowledgeRetrievalStartedEvent,
  MemoryRecallCompletedEvent,
  MemoryRecallStartedEvent,
  MemoryUpdateCompletedEvent,
  MemoryUpdateStartedEvent,
  ReasoningCompletedEvent,
  ReasoningStartedEvent,
  ReasoningStepEvent,
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
)
from definable.skill.base import Skill
from definable.tool.function import Function
from definable.utils.tools import get_function_call_for_tool_call
from pydantic import BaseModel

if TYPE_CHECKING:
  from pathlib import Path

  from definable.agent.compression import CompressionManager
  from definable.agent.interface.base import BaseInterface
  from definable.agent.reasoning.thinking import Thinking
  from definable.agent.tracing.base import Tracing
  from definable.knowledge import Knowledge
  from definable.memory.manager import Memory
  from definable.model.base import Model
  from definable.agent.reasoning.step import ReasoningStep, ThinkingOutput
  from definable.agent.replay import Replay, ReplayComparison
  from definable.agent.research.config import DeepResearchConfig
  from definable.agent.research.engine import DeepResearch


@runtime_checkable
class AsyncLifecycleToolkit(Protocol):
  """Protocol for toolkits with async lifecycle (e.g. MCPToolkit).

  Toolkits satisfying this protocol can be auto-managed by Agent:
  - Agent.__aenter__ / arun() calls initialize() on uninitialized toolkits
  - Agent.__aexit__ / _ashutdown() calls shutdown() on agent-owned toolkits
  """

  _initialized: bool

  async def initialize(self) -> None: ...
  async def shutdown(self) -> None: ...

  @property
  def tools(self) -> list: ...


class Agent:
  """
  Production-grade agent wrapper around model execution.

  Agent provides a clean interface for running LLM-based agents with:
  - Direct access to primary components (model, tools, toolkits, instructions)
  - Multi-turn conversation support
  - Middleware for cross-cutting concerns
  - Extensible tracing system
  - Context manager for resource cleanup

  Example:
      from definable.agent import Agent, AgentConfig
      from definable.model import OpenAIChat

      agent = Agent(
          model=OpenAIChat(id="gpt-4"),
          tools=[search_tool, calculate_tool],
          instructions="You are a helpful assistant.",
      )

      # Simple run
      output = agent.run("What is 2+2?")
      print(output.content)

      # Multi-turn conversation
      output2 = agent.run(
          "And what about 3+3?",
          messages=output.messages,
          session_id=output.session_id,
      )

      # With context manager for cleanup
      with agent:
          output = agent.run("Hello!")
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
    knowledge: Union["Knowledge", bool, None] = False,
    thinking: Union[bool, "Thinking", None] = None,
    deep_research: Union[bool, "DeepResearchConfig", "DeepResearch", None] = None,
    # ── Tools ───────────────────────────────────────────────
    tools: Optional[List[Function]] = None,
    toolkits: Optional[List[Toolkit]] = None,
    skills: Optional[List[Skill]] = None,
    skill_registry: Optional[Any] = None,
    # ── Observability ───────────────────────────────────────
    tracing: Union[bool, "Tracing", None] = False,
    # ── Support ─────────────────────────────────────────────
    readers: Optional[Any] = None,
    guardrails: Optional[Any] = None,
  ):
    """
    Initialize the agent.

    Args:
        model: Model instance to use for generation (required).
        tools: List of tools (Function objects) available to the agent.
        toolkits: List of toolkits providing additional tools.
        skills: List of skills providing tools + domain expertise.
            Each skill contributes tools (merged into the tool set) and
            instructions (merged into the system prompt). Skills are
            the highest-level abstraction — use them to give your agent
            domain expertise alongside capabilities.
        skill_registry: Optional SkillRegistry for markdown-based skills.
            Auto-selects eager mode (all skills injected) when the
            registry has 15 or fewer skills, or lazy mode (catalog +
            read_skill tool) for larger registries. Override by calling
            ``registry.as_eager()`` or ``registry.as_lazy()`` directly
            and passing the result to ``skills=``.
        instructions: System instructions for the agent.
        memory: Optional MemoryManager instance for persistent memory.
        readers: File reader configuration. Accepts:
            - None: no file reading (default)
            - True: auto-create FileReaderRegistry with all available readers
            - FileReaderRegistry: custom registry with user-selected readers
            - FileReader: single reader, wrapped in a registry
        thinking: Enable agent-level thinking/reasoning before the main execution.
            Accepts True (default config), Thinking instance (custom), or None (disabled).
        name: Optional human-readable name for the agent. Overrides config.agent_name.
        session_id: Optional session ID for multi-turn memory. Generated once
            at init if not provided. All runs reuse it by default; callers
            can still override per-call.
        config: Optional advanced configuration settings.
    """
    # Direct attributes — resolve string model shorthand
    self.model: "Model"
    if isinstance(model, str):
      from definable.model.openai.chat import OpenAIChat

      self.model = OpenAIChat(id=model)
    else:
      self.model = model
    self.tools = tools or []
    self.toolkits = toolkits or []
    self.skills = skills or []
    self.instructions = instructions
    self.readers = self._init_readers(readers)
    self.guardrails = guardrails

    # Optional config for advanced settings
    self.config = config or AgentConfig()
    if name is not None:
      self.config = dataclasses.replace(self.config, agent_name=name)

    # Resolve memory: Memory | bool → Memory | None
    self.memory = self._resolve_memory(memory)

    # Resolve knowledge: Knowledge | bool → Knowledge | None
    self._knowledge: Optional[Any] = self._resolve_knowledge(knowledge)

    # Resolve tracing: direct param takes precedence over config.tracing fallback
    self._tracing_config: Optional[Any] = self._resolve_tracing(tracing, self.config)

    # Thinking layer — accepts Thinking or bool
    from definable.agent.reasoning.thinking import Thinking as _Thinking

    if thinking is True:
      self._thinking: Optional[Any] = _Thinking()
    elif isinstance(thinking, _Thinking):
      self._thinking = thinking
    else:
      self._thinking = None

    # Deep research layer
    from definable.agent.research.config import DeepResearchConfig as _DRConfig
    from definable.agent.research.engine import DeepResearch as _DREngine

    if isinstance(deep_research, _DREngine):
      self._deep_research_config: Optional[_DRConfig] = None
      self._prebuilt_researcher: Optional[_DREngine] = deep_research
    elif deep_research is True:
      self._deep_research_config = _DRConfig()
      self._prebuilt_researcher = None
    elif isinstance(deep_research, _DRConfig):
      self._deep_research_config = deep_research
      self._prebuilt_researcher = None
    else:
      self._deep_research_config = None
      self._prebuilt_researcher = None

    # Convert skill_registry to skills (eager/lazy based on size)
    if skill_registry is not None:
      from definable.skill.registry import SkillRegistry

      if isinstance(skill_registry, SkillRegistry):
        if len(skill_registry) <= 15:
          self.skills.extend(skill_registry.as_eager())
        else:
          self.skills.append(skill_registry.as_lazy())

    # Initialize skills (call setup, validate)
    self._init_skills()

    # Internal state
    self._tools_dict: Dict[str, Function] = self._flatten_tools()
    self._trace_writer: Optional[TraceWriter] = self._init_tracing()
    self._compression_manager: Optional["CompressionManager"] = self._init_compression()
    self._middleware: List[Middleware] = []
    self._interfaces: List["BaseInterface"] = []
    self._triggers: List[Any] = []
    self._before_hooks: List[Callable] = []
    self._after_hooks: List[Callable] = []
    self._auth: Optional[Any] = None
    self._started = False
    self._pending_memory_tasks: list[asyncio.Task] = []
    self._event_bus: EventBus = EventBus()
    self._agent_owned_toolkits: list[Any] = []
    self._toolkit_init_lock: asyncio.Lock = asyncio.Lock()
    self.session_id = session_id or str(uuid4())

    # Deep research engine (prebuilt instance or lazy init from config)
    self._researcher: Optional["DeepResearch"] = self._prebuilt_researcher or (
      self._init_deep_research(self._deep_research_config) if self._deep_research_config else None
    )

  # --- Properties ---

  @property
  def agent_id(self) -> str:
    """Get the agent's unique identifier."""
    return self.config.agent_id or str(id(self))

  @property
  def agent_name(self) -> str:
    """Get the agent's name."""
    return self.config.agent_name or self.__class__.__name__

  @property
  def tool_names(self) -> List[str]:
    """Get list of available tool names."""
    return list(self._tools_dict.keys())

  @property
  def events(self) -> EventBus:
    """User-registerable event bus for callbacks on run events.

    Example::

        @agent.events.on(ToolCallStartedEvent)
        def on_tool(event):
            print(f"Tool: {event.tool.tool_name}")
    """
    return self._event_bus

  # --- Lifecycle Management ---

  def __enter__(self) -> "Agent":
    """Context manager entry."""
    self._start()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    """Context manager exit."""
    self._shutdown()

  async def __aenter__(self) -> "Agent":
    """Async context manager entry."""
    self._start()
    await self._ensure_toolkits_initialized()
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    """Async context manager exit."""
    await self._ashutdown()

  def _start(self) -> None:
    """Initialize resources."""
    if self._started:
      return
    self._started = True
    # Future: initialize connections, warm up caches, etc.

  def _shutdown(self) -> None:
    """Cleanup resources."""
    # Teardown skills
    for skill in self.skills:
      with contextlib.suppress(Exception):
        skill.teardown()
    if self._trace_writer:
      self._trace_writer.shutdown()
    self._started = False

  async def _ashutdown(self) -> None:
    """Async cleanup."""
    await self._drain_memory_tasks()
    # Shutdown toolkits we initialized (not user-managed ones)
    for toolkit in self._agent_owned_toolkits:
      with contextlib.suppress(Exception):
        await toolkit.shutdown()
    self._agent_owned_toolkits.clear()
    if self.memory:
      with contextlib.suppress(Exception):
        await self.memory.close()
    self._shutdown()

  async def _ensure_toolkits_initialized(self) -> None:
    """Initialize any AsyncLifecycleToolkit instances that aren't yet initialized.

    Skips already-initialized toolkits (user-managed), tracks which toolkits
    we initialized (for shutdown), and refreshes _tools_dict after init.
    """
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

  # --- Middleware Support ---

  def use(self, middleware: Middleware) -> "Agent":
    """
    Add middleware to the execution chain.

    Middleware is executed in the order added (outside-in),
    with post-processing in reverse order (inside-out).

    Args:
        middleware: Middleware instance to add.

    Returns:
        Self for method chaining.

    Example:
        agent.use(LoggingMiddleware(logger)).use(RetryMiddleware())
    """
    self._middleware.append(middleware)
    return self

  # --- Agent-Level Hooks ---

  def before_request(self, fn: Optional[Callable] = None) -> Callable:
    """Register a hook that fires before every ``arun()`` call.

    Supports both ``@agent.before_request`` (no parens) and
    ``@agent.before_request()`` (with parens).  The hook receives a
    :class:`RunContext` and is always non-fatal (errors are logged).

    Example::

      @agent.before_request
      async def log_request(context):
          print(f"Run {context.run_id} starting")
    """
    if fn is not None:
      # Used as @agent.before_request (no parens)
      self._before_hooks.append(fn)
      return fn

    # Used as @agent.before_request() (with parens)
    def decorator(func: Callable) -> Callable:
      self._before_hooks.append(func)
      return func

    return decorator

  def after_response(self, fn: Optional[Callable] = None) -> Callable:
    """Register a hook that fires after every ``arun()`` call.

    Supports both ``@agent.after_response`` (no parens) and
    ``@agent.after_response()`` (with parens).  The hook receives a
    :class:`RunOutput` and is always non-fatal (errors are logged).

    Example::

      @agent.after_response
      async def log_response(output):
          print(f"Run {output.run_id} completed: {output.content[:50]}")
    """
    if fn is not None:
      self._after_hooks.append(fn)
      return fn

    def decorator(func: Callable) -> Callable:
      self._after_hooks.append(func)
      return func

    return decorator

  async def _fire_before_hooks(self, context: RunContext) -> None:
    """Call all before_request hooks (non-fatal)."""
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
    """Call all after_response hooks (non-fatal)."""
    import inspect

    for hook in self._after_hooks:
      try:
        result = hook(output)
        if inspect.isawaitable(result):
          await result
      except Exception as e:
        from definable.utils.log import log_error

        log_error(f"after_response hook {hook.__name__} failed: {e}")

  # --- Auth ---

  @property
  def auth(self) -> Optional[Any]:
    """Get the auth provider."""
    return self._auth

  @auth.setter
  def auth(self, provider: Any) -> None:
    """Set the auth provider."""
    self._auth = provider

  # --- Run Methods ---

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
    """
    Synchronous run with multi-turn conversation support.

    Args:
        instruction: New user message (string, Message, or list).
        messages: Optional conversation history for multi-turn.
        session_id: Session identifier (auto-generated if not provided).
        run_id: Run identifier (auto-generated if not provided).
        user_id: User identifier for memory scoping and multi-user support.
        images: Images to include with the instruction.
        videos: Videos to include with the instruction.
        audio: Audio to include with the instruction.
        files: Files to include with the instruction.
        output_schema: Optional Pydantic model for structured output.

    Returns:
        RunOutput with response, metrics, tool executions, and messages.
    """
    try:
      loop = asyncio.get_running_loop()
    except RuntimeError:
      loop = None

    if loop and loop.is_running():
      # We're in an async context, create a new thread
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
      # Create a new event loop to avoid "Event loop is closed" errors
      # when making multiple sequential sync calls with async HTTP clients
      loop = asyncio.new_event_loop()
      asyncio.set_event_loop(loop)
      try:
        return loop.run_until_complete(
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
        # Robust cleanup sequence for async HTTP clients (httpx, etc.)
        try:
          # 1. Cancel pending tasks
          pending = asyncio.all_tasks(loop)
          for task in pending:
            task.cancel()
          # Allow cancelled tasks to complete
          if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
          # 2. Shutdown async generators (critical for httpx cleanup)
          loop.run_until_complete(loop.shutdown_asyncgens())
          # 3. Shutdown default executor (Python 3.9+)
          if hasattr(loop, "shutdown_default_executor"):
            loop.run_until_complete(loop.shutdown_default_executor())
        except Exception:
          pass
        finally:
          loop.close()

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
    cancellation_token: Optional[CancellationToken] = None,
  ) -> RunOutput:
    """
    Async run with middleware chain execution.

    Args:
        instruction: New user message (string, Message, or list).
        messages: Optional conversation history for multi-turn.
        session_id: Session identifier (auto-generated if not provided).
        run_id: Run identifier (auto-generated if not provided).
        user_id: User identifier for memory scoping and multi-user support.
        images: Images to include with the instruction.
        videos: Videos to include with the instruction.
        audio: Audio to include with the instruction.
        files: Files to include with the instruction.
        output_schema: Optional Pydantic model for structured output.

    Returns:
        RunOutput with response, metrics, tool executions, and messages.
    """
    run_id = run_id or str(uuid4())
    session_id = session_id or self.session_id

    # Auto-initialize async toolkits (e.g. MCPToolkit) if not already done
    await self._ensure_toolkits_initialized()

    # Normalize instruction to messages
    new_messages = self._normalize_instruction(instruction, images, videos, audio, files)
    all_messages = (messages or []) + new_messages

    # Build context with messages in metadata for middleware access
    context = RunContext(
      run_id=run_id,
      session_id=session_id,
      user_id=user_id,
      dependencies=self.config.dependencies,
      session_state=dict(self.config.session_state or {}),
      output_schema=output_schema,
      metadata={"_messages": all_messages},
    )

    # Fire before_request hooks
    await self._fire_before_hooks(context)

    # Pre-execution pipeline: readers → knowledge → research → memory recall
    await self._run_pre_execution_pipeline(context, new_messages, all_messages)

    # Input guardrails (after memory recall, before execution)
    if self.guardrails and self.guardrails.input:
      input_block = await self._run_input_guardrails(context, new_messages)
      if input_block is not None:
        await self._fire_after_hooks(input_block)
        return input_block

    run_input = RunInput(
      input_content=instruction,
      images=images,
      videos=videos,
      audios=audio,
      files=files,
    )

    # Build the execution chain with middleware
    async def core_handler(ctx: RunContext) -> RunOutput:
      return await self._execute_run(ctx, all_messages, run_input, cancellation_token=cancellation_token)

    # Wrap with middleware (innermost to outermost)
    handler = core_handler
    for middleware in reversed(self._middleware):
      prev_handler = handler

      async def wrapped_handler(ctx: RunContext, mw=middleware, h=prev_handler) -> RunOutput:
        return await mw(ctx, h)

      handler = wrapped_handler

    # Execute
    result = await handler(context)

    # Output guardrails (after execution, before memory store)
    if self.guardrails and self.guardrails.output and result.content:
      output_block = await self._run_output_guardrails(context, result)
      if output_block is not None:
        result = output_block

    # Memory store (after execution, fire-and-forget)
    # Include both user message(s) and assistant response for full conversational context
    if self._should_store_memory():
      store_messages = (new_messages + [Message(role="assistant", content=result.content)]) if result.content else new_messages
      self._memory_store(store_messages, context)

    # Fire after_response hooks
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
    """
    Streaming run that yields events as they occur in real-time.

    Args:
        instruction: New user message.
        messages: Optional conversation history.
        session_id: Session identifier.
        run_id: Run identifier.
        user_id: User identifier for memory scoping and multi-user support.
        images: Images to include.
        output_schema: Optional structured output schema.

    Yields:
        RunOutputEvent instances as the run progresses.
    """
    import queue
    import sys
    import threading
    import time

    event_queue: queue.Queue[Union[RunOutputEvent, Exception, None]] = queue.Queue()
    stop_event = threading.Event()
    loop_ready = threading.Event()
    loop_holder: Dict[str, asyncio.AbstractEventLoop] = {}
    queue_errors: List[BaseException] = []
    queue_errors_lock = threading.Lock()

    timeout_seconds = self.config.stream_timeout_seconds
    if timeout_seconds is None:
      timeout_seconds = 300.0  # type: ignore[unreachable]
    deadline = time.monotonic() + timeout_seconds if timeout_seconds and timeout_seconds > 0 else None

    def record_queue_error(err: BaseException) -> None:
      with queue_errors_lock:
        queue_errors.append(err)

    def safe_put(item: Union[RunOutputEvent, Exception, None]) -> None:
      try:
        event_queue.put(item)
      except Exception as exc:
        record_queue_error(exc)
        stop_event.set()

    def request_loop_cancel() -> None:
      loop = loop_holder.get("loop")
      if loop and loop.is_running():
        try:

          def _cancel_tasks() -> None:
            for task in asyncio.all_tasks(loop):
              task.cancel()

          loop.call_soon_threadsafe(_cancel_tasks)
        except Exception as exc:
          record_queue_error(exc)

    def run_async_stream() -> None:
      """Run async stream in background thread, push events to queue."""
      # Create a new event loop for this thread
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
            safe_put(event)
        except Exception as e:
          safe_put(e)
        finally:
          safe_put(None)  # Sentinel to signal completion

      try:
        loop.run_until_complete(stream_to_queue())
      finally:
        # Robust cleanup sequence for async HTTP clients (httpx, etc.)
        with contextlib.suppress(Exception):
          # 1. Cancel pending tasks
          pending = asyncio.all_tasks(loop)
          for task in pending:
            task.cancel()
          if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
          # 2. Shutdown async generators (critical for httpx cleanup)
          loop.run_until_complete(loop.shutdown_asyncgens())
          # 3. Shutdown default executor (Python 3.9+)
          if hasattr(loop, "shutdown_default_executor"):
            loop.run_until_complete(loop.shutdown_default_executor())
        with contextlib.suppress(Exception):
          loop.close()

    # Start background thread
    thread = threading.Thread(target=run_async_stream, daemon=True)
    thread.start()

    # Yield events as they arrive
    try:
      loop_ready.wait(timeout=1.0)
      while True:
        with queue_errors_lock:
          if queue_errors:
            raise queue_errors[0]
        if deadline is None:
          try:
            item = event_queue.get()
          except Exception as exc:
            stop_event.set()
            request_loop_cancel()
            raise exc
        else:
          remaining = deadline - time.monotonic()
          if remaining <= 0:
            stop_event.set()
            request_loop_cancel()
            raise TimeoutError(f"Stream timed out after {timeout_seconds:.0f} seconds.")
          try:
            item = event_queue.get(timeout=remaining)
          except queue.Empty:
            with queue_errors_lock:
              if queue_errors:
                raise queue_errors[0]
            stop_event.set()
            request_loop_cancel()
            raise TimeoutError(f"Stream timed out after {timeout_seconds:.0f} seconds.")
          except Exception as exc:
            stop_event.set()
            request_loop_cancel()
            raise exc
        if item is None:  # Sentinel - stream complete
          break
        if isinstance(item, Exception):
          raise item
        yield item
    finally:
      stop_event.set()
      request_loop_cancel()
      thread.join(timeout=5.0)
      if thread.is_alive():
        request_loop_cancel()
        thread.join(timeout=5.0)
      if thread.is_alive() and sys.exc_info()[0] is None:
        raise TimeoutError("Background stream thread did not terminate.")

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
    cancellation_token: Optional[CancellationToken] = None,
  ) -> AsyncIterator[RunOutputEvent]:
    """
    Async streaming run that yields events with full agent loop support.

    Uses the same AgentLoop as arun() but in streaming mode,
    yielding RunContentEvent deltas as tokens arrive.

    Args:
        instruction: New user message.
        messages: Optional conversation history.
        session_id: Session identifier.
        run_id: Run identifier.
        user_id: User identifier for memory scoping and multi-user support.
        images: Images to include.
        output_schema: Optional structured output schema.
        cancellation_token: Optional token for cooperative cancellation.

    Yields:
        RunOutputEvent instances as the run progresses.
    """
    run_id = run_id or str(uuid4())
    session_id = session_id or self.session_id

    # Auto-initialize async toolkits (e.g. MCPToolkit) if not already done
    await self._ensure_toolkits_initialized()

    # Normalize instruction to messages
    new_messages = self._normalize_instruction(instruction, images)
    all_messages = (messages or []) + new_messages

    # Build context with messages in metadata for middleware access
    context = RunContext(
      run_id=run_id,
      session_id=session_id,
      user_id=user_id,
      dependencies=self.config.dependencies,
      session_state=dict(self.config.session_state or {}),
      output_schema=output_schema,
      metadata={"_messages": all_messages},
    )

    # Pre-execution pipeline: readers → knowledge → research → memory recall — yield events
    for evt in await self._run_pre_execution_pipeline(context, new_messages, all_messages):
      yield evt

    # Input guardrails (streaming)
    if self.guardrails and self.guardrails.input:
      input_block = await self._run_input_guardrails(context, new_messages)
      if input_block is not None:
        blocked_event = RunCompletedEvent(
          run_id=context.run_id,
          session_id=context.session_id,
          agent_id=self.agent_id,
          agent_name=self.agent_name,
          content=input_block.content,
          metadata={"status": "blocked"},
        )
        self._emit(blocked_event)
        yield blocked_event
        return

    # Prepare tools
    tools = self._prepare_tools_for_run(context)

    run_input = RunInput(
      input_content=instruction,
      images=images,
    )

    # Emit RunStarted
    started_event = RunStartedEvent(
      run_id=context.run_id,
      session_id=context.session_id,
      agent_id=self.agent_id,
      agent_name=self.agent_name,
      model=self.model.id,
      model_provider=self.model.provider,  # type: ignore[arg-type]
      run_input=run_input,
    )
    self._emit(started_event)
    await self._event_bus.emit(started_event)
    yield started_event

    try:
      # Build invoke messages (system prompt, thinking, knowledge, memory, readers)
      # Note: Thinking streaming (yielding ReasoningContentDeltaEvent) is still handled
      # by the agent for now — the loop only handles the main model+tool loop.
      invoke_messages, _reasoning_steps, _reasoning_agent_messages = await self._build_invoke_messages(context, all_messages, tools)

      # For streaming thinking events, we'd need to re-implement the thinking phase
      # with streaming here. For now, the thinking is done in _build_invoke_messages
      # (non-streaming). TODO: Extract streaming thinking into a separate method.

      # Create the streaming loop
      loop = AgentLoop(
        model=self.model,
        tools=tools,
        messages=invoke_messages,
        context=context,
        config=self.config,
        streaming=True,
        cancellation_token=cancellation_token,
        compression_manager=self._compression_manager,
        guardrails=self.guardrails,
        emit_fn=self._emit,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
      )

      # Yield events from the streaming loop
      final_content: Optional[str] = None
      async for event in loop.run_streaming():
        await self._event_bus.emit(event)
        # Don't double-emit to trace for content deltas (RunCompleted has full content)
        if not isinstance(event, RunContentEvent):
          self._emit(event)
        if isinstance(event, RunCompletedEvent):
          final_content = event.content
        yield event

      # Output guardrails (streaming — modify final_content before completed event)
      if self.guardrails and self.guardrails.output and final_content:
        _temp_output = RunOutput(
          run_id=context.run_id,
          session_id=context.session_id,
          content=final_content,
        )
        output_block = await self._run_output_guardrails(context, _temp_output)
        if output_block is not None:
          final_content = output_block.content or final_content

      # Memory store (after streaming, fire-and-forget)
      if self._should_store_memory():
        store_messages = (new_messages + [Message(role="assistant", content=final_content)]) if final_content else new_messages
        for evt in self._memory_store(store_messages, context):
          yield evt

    except AgentCancelled:
      from definable.agent.events import RunCancelledEvent

      cancelled_event = RunCancelledEvent(
        run_id=context.run_id,
        session_id=context.session_id,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
        reason="Cancelled via CancellationToken",
      )
      self._emit(cancelled_event)
      yield cancelled_event

    except Exception as e:
      error_event = RunErrorEvent(
        run_id=context.run_id,
        session_id=context.session_id,
        agent_id=self.agent_id,
        error_type=type(e).__name__,
        content=str(e),
      )
      self._emit(error_event)
      yield error_event
      raise

  async def continue_run(
    self,
    *,
    run_output: RunOutput,
    cancellation_token: Optional[CancellationToken] = None,
  ) -> RunOutput:
    """Resume a paused run after HITL requirements are resolved.

    After ``arun()`` returns a paused ``RunOutput`` (with
    ``run_output.is_paused == True``), the caller resolves each
    requirement (e.g. ``req.confirm()`` or ``req.reject()``) and then
    calls ``continue_run()`` to resume execution.

    Args:
        run_output: The paused RunOutput from a previous ``arun()`` call.
        cancellation_token: Optional token for cooperative cancellation.

    Returns:
        A new RunOutput with the resumed execution results.

    Raises:
        ValueError: If the run is not paused or has unresolved requirements.
    """
    if not run_output.is_paused:
      raise ValueError("RunOutput is not paused — nothing to continue")
    unresolved = run_output.active_requirements
    if unresolved:
      raise ValueError(f"{len(unresolved)} requirement(s) still unresolved. Resolve them before calling continue_run().")

    # Build messages from the paused run
    messages = run_output.messages or []

    # For each resolved requirement, add the tool result message
    for req in run_output.requirements or []:
      te = req.tool_execution
      if te is None:
        continue
      if req.confirmation is False:
        # Rejected — add rejection message
        messages.append(
          Message(
            role="tool",
            content="[REJECTED] The user rejected this tool call.",
            tool_call_id=te.tool_call_id,
            name=te.tool_name,
          )
        )
      elif req.confirmation is True:
        # Confirmed — execute the tool now
        fn = self._tools_dict.get(te.tool_name)  # type: ignore[arg-type]
        if fn:
          function_call = get_function_call_for_tool_call(
            {
              "id": te.tool_call_id,
              "function": {"name": te.tool_name, "arguments": str(te.tool_args or "{}")},
            },
            self._tools_dict,
          )
          if function_call:
            result_obj = await function_call.aexecute()
            messages.append(
              Message(
                role="tool",
                content=str(result_obj.result) if result_obj.status == "success" else str(result_obj.error),
                tool_call_id=te.tool_call_id,
                name=te.tool_name,
              )
            )
          else:
            messages.append(Message(role="tool", content=f"Tool '{te.tool_name}' not found", tool_call_id=te.tool_call_id, name=te.tool_name))
        else:
          messages.append(Message(role="tool", content=f"Tool '{te.tool_name}' not found", tool_call_id=te.tool_call_id, name=te.tool_name))
      elif req.external_execution_result is not None:
        messages.append(
          Message(
            role="tool",
            content=req.external_execution_result,
            tool_call_id=te.tool_call_id,
            name=te.tool_name,
          )
        )

    # Re-enter the agent run with the updated messages
    return await self.arun(
      instruction=messages[-1] if messages and messages[-1].role == "user" else "Continue.",
      messages=messages,
      session_id=run_output.session_id,
      run_id=None,  # New run_id for the continuation
      cancellation_token=cancellation_token,
    )

  async def continue_run_stream(
    self,
    *,
    run_output: RunOutput,
    cancellation_token: Optional[CancellationToken] = None,
  ) -> AsyncIterator[RunOutputEvent]:
    """Streaming variant of :meth:`continue_run`.

    Same semantics as ``continue_run`` but yields events as they occur.

    Args:
        run_output: The paused RunOutput from a previous call.
        cancellation_token: Optional token for cooperative cancellation.

    Yields:
        RunOutputEvent instances as the resumed run progresses.

    Raises:
        ValueError: If the run is not paused or has unresolved requirements.
    """
    if not run_output.is_paused:
      raise ValueError("RunOutput is not paused — nothing to continue")
    unresolved = run_output.active_requirements
    if unresolved:
      raise ValueError(f"{len(unresolved)} requirement(s) still unresolved. Resolve them before calling continue_run_stream().")

    # Build messages from the paused run
    messages = run_output.messages or []

    # For each resolved requirement, add the tool result message
    for req in run_output.requirements or []:
      te = req.tool_execution
      if te is None:
        continue
      if req.confirmation is False:
        messages.append(
          Message(
            role="tool",
            content="[REJECTED] The user rejected this tool call.",
            tool_call_id=te.tool_call_id,
            name=te.tool_name,
          )
        )
      elif req.confirmation is True:
        fn = self._tools_dict.get(te.tool_name)  # type: ignore[arg-type]
        if fn:
          function_call = get_function_call_for_tool_call(
            {
              "id": te.tool_call_id,
              "function": {"name": te.tool_name, "arguments": str(te.tool_args or "{}")},
            },
            self._tools_dict,
          )
          if function_call:
            result_obj = await function_call.aexecute()
            messages.append(
              Message(
                role="tool",
                content=str(result_obj.result) if result_obj.status == "success" else str(result_obj.error),
                tool_call_id=te.tool_call_id,
                name=te.tool_name,
              )
            )
          else:
            messages.append(Message(role="tool", content=f"Tool '{te.tool_name}' not found", tool_call_id=te.tool_call_id, name=te.tool_name))
        else:
          messages.append(Message(role="tool", content=f"Tool '{te.tool_name}' not found", tool_call_id=te.tool_call_id, name=te.tool_name))
      elif req.external_execution_result is not None:
        messages.append(
          Message(
            role="tool",
            content=req.external_execution_result,
            tool_call_id=te.tool_call_id,
            name=te.tool_name,
          )
        )

    # Re-enter the streaming agent run with the updated messages
    async for event in self.arun_stream(
      instruction=messages[-1] if messages and messages[-1].role == "user" else "Continue.",
      messages=messages,
      session_id=run_output.session_id,
      run_id=None,
      cancellation_token=cancellation_token,
    ):
      yield event

  # --- Knowledge & Memory Helpers ---

  async def _knowledge_retrieve(self, context: RunContext) -> List[RunOutputEvent]:
    """Retrieve knowledge documents, emit events, inject into context."""
    kc = self._knowledge
    if not (kc and kc.enabled):
      return []

    messages = context.metadata.get("_messages") if context.metadata else None
    if not messages:
      return []

    from definable.agent.middleware import KnowledgeMiddleware

    km = KnowledgeMiddleware(kc)
    query = km._extract_query(messages)
    if not query:
      return []

    import time

    events: List[RunOutputEvent] = []
    started = KnowledgeRetrievalStartedEvent(
      run_id=context.run_id,
      session_id=context.session_id,
      agent_id=self.agent_id,
      agent_name=self.agent_name,
      query=query,
    )
    self._emit(started)
    events.append(started)

    start_time = time.perf_counter()
    try:
      documents = await kc.asearch(
        query=query,
        top_k=kc.top_k,
        rerank=kc.rerank,
      )
    except Exception:
      elapsed = (time.perf_counter() - start_time) * 1000
      completed = KnowledgeRetrievalCompletedEvent(
        run_id=context.run_id,
        session_id=context.session_id,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
        query=query,
        documents_found=0,
        documents_used=0,
        duration_ms=elapsed,
      )
      self._emit(completed)
      events.append(completed)
      return events

    documents_found = len(documents)

    # Filter by min_score
    if kc.min_score is not None:
      documents = [d for d in documents if d.reranking_score is not None and d.reranking_score >= kc.min_score]

    if documents:
      context_text = km._format_context(documents)
      context.knowledge_context = context_text
      context.knowledge_documents = documents
      context.active_layers.add("knowledge")
      if context.metadata is None:
        context.metadata = {}
      context.metadata["_knowledge_position"] = kc.context_position

    elapsed = (time.perf_counter() - start_time) * 1000
    completed = KnowledgeRetrievalCompletedEvent(
      run_id=context.run_id,
      session_id=context.session_id,
      agent_id=self.agent_id,
      agent_name=self.agent_name,
      query=query,
      documents_found=documents_found,
      documents_used=len(documents),
      duration_ms=elapsed,
    )
    self._emit(completed)
    events.append(completed)
    return events

  def _init_deep_research(self, config: "DeepResearchConfig") -> Optional["DeepResearch"]:
    """Initialize the deep research engine if configured.

    Non-fatal: returns None with a warning if search capability cannot be found.
    """
    from definable.utils.log import log_debug, log_warning

    if not config or not config.enabled:
      return None

    try:
      from definable.agent.research.engine import DeepResearch
      from definable.agent.research.search import create_search_provider

      # Try explicit search_fn or provider first
      if config.search_fn is not None or config.search_provider != "duckduckgo":
        provider = create_search_provider(
          provider=config.search_provider,
          config=config.search_provider_config,
          search_fn=config.search_fn,
        )
      else:
        # Try auto-discovering from WebSearch skill
        provider = self._discover_search_provider()  # type: ignore[assignment]
        if provider is None:
          # Fall back to DuckDuckGo
          provider = create_search_provider("duckduckgo")  # type: ignore[unreachable]

      compression_model = config.compression_model or self.model
      log_debug("Deep research engine initialized")
      return DeepResearch(
        model=self.model,
        search_provider=provider,
        compression_model=compression_model,
        config=config,
      )
    except Exception as e:
      log_warning(f"Failed to initialize deep research: {e}")
      return None

  def _discover_search_provider(self) -> Optional[Any]:
    """Try to auto-discover a search provider from WebSearch skill."""
    from definable.utils.log import log_debug

    for skill in self.skills:
      # Check for WebSearch skill with its _search_fn
      skill_cls_name = type(skill).__name__
      if skill_cls_name == "WebSearch" and hasattr(skill, "_search_fn"):
        from definable.agent.research.search import CallableSearchProvider
        from definable.agent.research.search.base import SearchResult

        raw_fn = skill._search_fn

        async def _wrapped(query: str, max_results: int = 10) -> list:
          import asyncio

          text = await asyncio.to_thread(raw_fn, query, max_results)
          # WebSearch._search_fn returns formatted string, not SearchResult list.
          # Parse it back into SearchResult objects.
          results = []
          for block in text.split("\n\n---\n\n"):
            lines = block.strip().split("\n", 2)
            if len(lines) >= 2:
              title = lines[0].strip("*").strip()
              url = lines[1].strip()
              snippet = lines[2] if len(lines) > 2 else ""
              results.append(SearchResult(title=title, url=url, snippet=snippet))
          return results

        log_debug("Auto-discovered search provider from WebSearch skill")
        return CallableSearchProvider(_wrapped)  # type: ignore[return-value]

    return None

  async def _deep_research(self, context: RunContext) -> List[RunOutputEvent]:
    """Execute deep research pipeline, emit events, inject context."""
    if not self._researcher:
      return []

    config = self._deep_research_config or (self._researcher._config if self._researcher else None)
    if not config or not config.enabled:
      return []

    # Extract query from last user message
    messages = context.metadata.get("_messages") if context.metadata else None
    if not messages:
      return []

    query = None
    for msg in reversed(messages):
      if hasattr(msg, "role") and msg.role == "user" and msg.content:
        query = msg.content if isinstance(msg.content, str) else str(msg.content)
        break
    if not query:
      return []

    # Auto trigger: ask model if research is needed
    if config.trigger == "auto":
      try:
        needs = await self._researcher.needs_research(query)
        if not needs:
          return []
      except Exception:
        pass  # Default to running research on failure

    import time

    events: List[RunOutputEvent] = []
    started = DeepResearchStartedEvent(
      run_id=context.run_id,
      session_id=context.session_id,
      agent_id=self.agent_id,
      agent_name=self.agent_name,
      query=query,
      depth=config.depth,
    )
    self._emit(started)
    events.append(started)

    start_time = time.perf_counter()
    try:
      result = await self._researcher.arun(query)
      context.research_context = result.context
      context.research_result = result
    except Exception as e:
      from definable.utils.log import log_warning

      log_warning(f"Deep research failed: {e}")
      elapsed = (time.perf_counter() - start_time) * 1000
      completed = DeepResearchCompletedEvent(
        run_id=context.run_id,
        session_id=context.session_id,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
        query=query,
        duration_ms=elapsed,
      )
      self._emit(completed)
      events.append(completed)
      return events

    elapsed = (time.perf_counter() - start_time) * 1000
    completed = DeepResearchCompletedEvent(
      run_id=context.run_id,
      session_id=context.session_id,
      agent_id=self.agent_id,
      agent_name=self.agent_name,
      query=query,
      sources_used=result.metrics.total_sources_read,
      facts_extracted=result.metrics.unique_facts,
      contradictions_found=result.metrics.contradictions_found,
      waves_executed=result.metrics.waves_executed,
      duration_ms=elapsed,
      compression_ratio=result.metrics.compression_ratio_avg,
    )
    self._emit(completed)
    events.append(completed)
    return events

  async def _drain_memory_tasks(self) -> None:
    """Await all pending memory background tasks (with timeout)."""
    if not self._pending_memory_tasks:
      return
    done, _ = await asyncio.wait(self._pending_memory_tasks, timeout=30.0)
    for task in done:
      with contextlib.suppress(Exception):
        task.result()
    self._pending_memory_tasks = [t for t in self._pending_memory_tasks if not t.done()]

  async def _memory_recall(self, context: RunContext, new_messages: List[Message]) -> List[RunOutputEvent]:
    """Recall relevant memories, emit events, inject into context."""
    assert self.memory is not None
    await self._drain_memory_tasks()
    import time

    user_id = context.user_id or "default"

    events: List[RunOutputEvent] = []

    # Extract the last user message as the query (for event metadata)
    query = None
    for msg in reversed(new_messages):
      if msg.role == "user" and msg.content:
        query = msg.content if isinstance(msg.content, str) else str(msg.content)
        break

    started = MemoryRecallStartedEvent(
      run_id=context.run_id,
      session_id=context.session_id,
      agent_id=self.agent_id,
      agent_name=self.agent_name,
      query=query or "",
    )
    self._emit(started)
    events.append(started)

    start_time = time.perf_counter()

    # Ensure store is initialized
    await self.memory._ensure_initialized()

    # Get all memories for this user
    memories = await self.memory.aget_user_memories(user_id=user_id)

    elapsed = (time.perf_counter() - start_time) * 1000

    if memories:
      formatted = self.memory.format_memories_for_prompt(memories)
      context.memory_context = f"<memories_from_previous_interactions>\n{formatted}\n</memories_from_previous_interactions>"
      context.active_layers.add("memory")

    completed = MemoryRecallCompletedEvent(
      run_id=context.run_id,
      session_id=context.session_id,
      agent_id=self.agent_id,
      agent_name=self.agent_name,
      query=query or "",
      tokens_used=len(context.memory_context or "") // 4,
      chunks_included=len(memories),
      chunks_available=len(memories),
      duration_ms=elapsed,
    )
    self._emit(completed)
    events.append(completed)
    return events

  def _memory_store(self, new_messages: List[Message], context: RunContext) -> List[RunOutputEvent]:
    """Fire-and-forget LLM-based memory extraction from new messages, emit events."""
    assert self.memory is not None
    import time

    if not self.memory.update_on_run:
      return []

    events: List[RunOutputEvent] = []
    message_count = len(new_messages)

    started = MemoryUpdateStartedEvent(
      run_id=context.run_id,
      session_id=context.session_id,
      agent_id=self.agent_id,
      agent_name=self.agent_name,
      message_count=message_count,
    )
    self._emit(started)
    events.append(started)

    try:
      loop = asyncio.get_running_loop()
      memory = self.memory

      # Ensure the memory manager has a model (use agent's model if not set)
      if memory.model is None:
        memory.model = self.model

      async def _store_and_emit() -> None:
        from definable.utils.log import log_warning

        start_time = time.perf_counter()
        try:
          await memory._ensure_initialized()
          # Extract user messages for memory processing
          # Pass all messages (user + assistant) so memory LLM has full conversational context
          if new_messages:
            await memory.acreate_user_memories(
              messages=new_messages,
              user_id=context.user_id or "default",
              agent_id=self.agent_id,
            )
        except Exception as e:
          log_warning(f"Memory store failed: {type(e).__name__}: {e}")
        finally:
          elapsed = (time.perf_counter() - start_time) * 1000
          completed = MemoryUpdateCompletedEvent(
            run_id=context.run_id,
            session_id=context.session_id,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            message_count=message_count,
            duration_ms=elapsed,
          )
          self._emit(completed)

      task = loop.create_task(_store_and_emit())
      self._pending_memory_tasks.append(task)
      task.add_done_callback(lambda t: self._pending_memory_tasks.remove(t) if t in self._pending_memory_tasks else None)
    except RuntimeError:
      pass  # No running loop — skip memory storage

    return events

  # --- Readers Helpers ---

  @staticmethod
  def _init_readers(readers: Optional[Any]) -> Optional[Any]:
    """Resolve the readers= parameter into a BaseReader or None.

    Accepts:
      - None/False → None
      - True → BaseReader() with all defaults
      - BaseReader instance → use as-is
      - BaseParser instance → wrap in BaseReader with custom ParserRegistry
      - ProviderReader instance (e.g., MistralReader) → use as-is
      - Legacy FileReaderRegistry → use as-is (it's now BaseReader)
    """
    if readers is None or readers is False:
      return None
    if readers is True:
      from definable.reader import BaseReader

      return BaseReader()
    # Check if it's a BaseParser (single parser → wrap in BaseReader)
    from definable.reader.parsers.base_parser import BaseParser

    if isinstance(readers, BaseParser):
      from definable.reader import BaseReader
      from definable.reader.registry import ParserRegistry

      registry = ParserRegistry(include_defaults=False)
      registry.register(readers)
      return BaseReader(registry=registry)
    # Assume it's already a BaseReader / ProviderReader — use as-is
    return readers

  async def _readers_extract(self, context: RunContext, new_messages: List[Message]) -> List[RunOutputEvent]:
    """Extract text from files in new_messages, inject into context."""
    if not self.readers:
      return []

    # Collect files from all new messages
    files: List[File] = []
    for msg in new_messages:
      if msg.files:
        files.extend(msg.files)
    if not files:
      return []

    import time

    events: List[RunOutputEvent] = []
    started = FileReadStartedEvent(
      run_id=context.run_id,
      session_id=context.session_id,
      agent_id=self.agent_id,
      agent_name=self.agent_name,
      file_count=len(files),
    )
    self._emit(started)
    events.append(started)

    start_time = time.perf_counter()
    try:
      results = await self.readers.aread_all(files)
    except Exception:
      from definable.utils.log import log_warning

      log_warning("File reading failed", exc_info=True)
      elapsed = (time.perf_counter() - start_time) * 1000
      completed = FileReadCompletedEvent(
        run_id=context.run_id,
        session_id=context.session_id,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
        file_count=len(files),
        files_read=0,
        files_failed=len(files),
        duration_ms=elapsed,
      )
      self._emit(completed)
      events.append(completed)
      return events

    # Format successful results into context block
    file_blocks: List[str] = []
    files_read = 0
    files_failed = 0
    for result in results:
      if result.error:
        files_failed += 1
      elif result.content:
        files_read += 1
        mime_attr = f' type="{result.mime_type}"' if result.mime_type else ""
        file_blocks.append(f'<file name="{result.filename}"{mime_attr}>\n{result.content}\n</file>')

    if file_blocks:
      context.readers_context = "<file_contents>\n" + "\n".join(file_blocks) + "\n</file_contents>"

    elapsed = (time.perf_counter() - start_time) * 1000
    completed = FileReadCompletedEvent(
      run_id=context.run_id,
      session_id=context.session_id,
      agent_id=self.agent_id,
      agent_name=self.agent_name,
      file_count=len(files),
      files_read=files_read,
      files_failed=files_failed,
      duration_ms=elapsed,
    )
    self._emit(completed)
    events.append(completed)
    return events

  # --- Guardrail Helpers ---

  def _extract_input_text(self, new_messages: List[Message]) -> str:
    """Extract text content from the new user messages for guardrail checking."""
    parts: List[str] = []
    for msg in new_messages:
      if msg.role == "user" and msg.content:
        parts.append(msg.content if isinstance(msg.content, str) else str(msg.content))
    return "\n".join(parts)

  async def _run_input_guardrails(self, context: RunContext, new_messages: List[Message]) -> Optional[RunOutput]:
    """Run input guardrails. Returns RunOutput if blocked, None if allowed."""
    assert self.guardrails is not None

    from definable.agent.guardrail.events import GuardrailBlockedEvent, GuardrailCheckedEvent

    text = self._extract_input_text(new_messages)
    if not text:
      return None

    results = await self.guardrails.run_input_checks(text, context)

    for result in results:
      gname = (result.metadata or {}).get("guardrail_name", "unknown")
      duration = (result.metadata or {}).get("duration_ms")

      self._emit(
        GuardrailCheckedEvent(
          run_id=context.run_id,
          session_id=context.session_id,
          agent_id=self.agent_id,
          agent_name=self.agent_name,
          guardrail_name=gname,
          guardrail_type="input",
          action=result.action,
          message=result.message,
          duration_ms=duration,
        )
      )

      if result.action == "block":
        self._emit(
          GuardrailBlockedEvent(
            run_id=context.run_id,
            session_id=context.session_id,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            guardrail_name=gname,
            guardrail_type="input",
            reason=result.message or "Blocked by input guardrail",
          )
        )

        reason = result.message or "Blocked by input guardrail"
        if self.guardrails.on_block == "raise":
          from definable.exceptions import CheckTrigger, InputCheckError

          raise InputCheckError(reason, check_trigger=CheckTrigger.GUARDRAIL_BLOCKED)

        return RunOutput(
          run_id=context.run_id,
          session_id=context.session_id,
          agent_id=self.agent_id,
          agent_name=self.agent_name,
          content=reason,
          status=RunStatus.blocked,
        )

      if result.action == "modify" and result.modified_text is not None:
        # Replace the last user message content
        for i in range(len(new_messages) - 1, -1, -1):
          if new_messages[i].role == "user":
            new_messages[i] = Message(
              role="user",
              content=result.modified_text,
              images=new_messages[i].images,
              videos=new_messages[i].videos,
              audio=new_messages[i].audio,
              files=new_messages[i].files,
            )
            break
        # Also update all_messages in context metadata
        all_messages = context.metadata.get("_messages") if context.metadata else None
        if all_messages:
          for i in range(len(all_messages) - 1, -1, -1):
            if all_messages[i].role == "user":
              all_messages[i] = Message(
                role="user",
                content=result.modified_text,
                images=all_messages[i].images,
                videos=all_messages[i].videos,
                audio=all_messages[i].audio,
                files=all_messages[i].files,
              )
              break

    return None

  async def _run_output_guardrails(self, context: RunContext, result: RunOutput) -> Optional[RunOutput]:
    """Run output guardrails. Returns modified RunOutput if blocked/modified, None if allowed."""
    assert self.guardrails is not None

    from definable.agent.guardrail.events import GuardrailBlockedEvent, GuardrailCheckedEvent

    text = result.content if isinstance(result.content, str) else str(result.content or "")
    if not text:
      return None

    results = await self.guardrails.run_output_checks(text, context)

    for gr in results:
      gname = (gr.metadata or {}).get("guardrail_name", "unknown")
      duration = (gr.metadata or {}).get("duration_ms")

      self._emit(
        GuardrailCheckedEvent(
          run_id=context.run_id,
          session_id=context.session_id,
          agent_id=self.agent_id,
          agent_name=self.agent_name,
          guardrail_name=gname,
          guardrail_type="output",
          action=gr.action,
          message=gr.message,
          duration_ms=duration,
        )
      )

      if gr.action == "block":
        self._emit(
          GuardrailBlockedEvent(
            run_id=context.run_id,
            session_id=context.session_id,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            guardrail_name=gname,
            guardrail_type="output",
            reason=gr.message or "Blocked by output guardrail",
          )
        )

        reason = gr.message or "Blocked by output guardrail"
        if self.guardrails.on_block == "raise":
          from definable.exceptions import CheckTrigger, OutputCheckError

          raise OutputCheckError(reason, check_trigger=CheckTrigger.GUARDRAIL_BLOCKED)

        return RunOutput(
          run_id=context.run_id,
          session_id=context.session_id,
          agent_id=self.agent_id,
          agent_name=self.agent_name,
          content=reason,
          status=RunStatus.blocked,
          messages=result.messages,
          metrics=result.metrics,
        )

      if gr.action == "modify" and gr.modified_text is not None:
        result.content = gr.modified_text
        if result.metadata is None:
          result.metadata = {}
        result.metadata["guardrail_modified"] = True

    return None

  async def _run_tool_guardrails(self, context: RunContext, tool_execution: ToolExecution) -> Optional[str]:
    """Run tool guardrails. Returns block reason string if blocked, None if allowed."""
    assert self.guardrails is not None

    from definable.agent.guardrail.events import GuardrailBlockedEvent, GuardrailCheckedEvent

    tool_name = tool_execution.tool_name or ""
    tool_args = tool_execution.tool_args or {}

    results = await self.guardrails.run_tool_checks(tool_name, tool_args, context)

    for gr in results:
      gname = (gr.metadata or {}).get("guardrail_name", "unknown")
      duration = (gr.metadata or {}).get("duration_ms")

      self._emit(
        GuardrailCheckedEvent(
          run_id=context.run_id,
          session_id=context.session_id,
          agent_id=self.agent_id,
          agent_name=self.agent_name,
          guardrail_name=gname,
          guardrail_type="tool",
          action=gr.action,
          message=gr.message,
          duration_ms=duration,
        )
      )

      if gr.action == "block":
        reason = gr.message or f"Tool '{tool_name}' blocked by guardrail"
        self._emit(
          GuardrailBlockedEvent(
            run_id=context.run_id,
            session_id=context.session_id,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            guardrail_name=gname,
            guardrail_type="tool",
            reason=reason,
          )
        )
        return reason

    return None

  # --- Thinking Layer ---

  _DEFAULT_THINKING_PROMPT = "Analyze the user's request. Determine what they need, your approach, and which tools (if any) to use."

  def _build_thinking_prompt(
    self,
    context: RunContext,
    tools: Dict[str, Function],
  ) -> str:
    """Build a context-aware thinking prompt with tool catalog, agent role, and context flags."""
    parts = [self._DEFAULT_THINKING_PROMPT]

    # Agent role (first 500 chars)
    if self.instructions:
      truncated = self.instructions[:500]
      if len(self.instructions) > 500:
        truncated += "..."
      parts.append(f"\nYour role: {truncated}")

    # Tool catalog (name + one-line description)
    if tools:
      tool_lines = []
      for name, fn in tools.items():
        desc = (fn.description or "").split("\n")[0][:100]
        tool_lines.append(f"- {name}: {desc}" if desc else f"- {name}")
      parts.append("\nAvailable tools:\n" + "\n".join(tool_lines))

    # Context availability flags (NOT the full content)
    flags = []
    if context.knowledge_context:
      flags.append("knowledge base context is available")
    if context.memory_context:
      flags.append("conversation memory is available")
    if flags:
      parts.append(f"\nContext: {'; '.join(flags)}.")

    return "\n".join(parts)

  @staticmethod
  def _format_thinking_injection(output: "ThinkingOutput") -> str:
    """Format ThinkingOutput into a compact system prompt injection."""
    parts = [f"<analysis>{output.approach}"]
    if output.tool_plan:
      parts.append(f" Tools: {', '.join(output.tool_plan)}.")
    parts.append("</analysis>")
    return "".join(parts)

  def _extract_last_user_query(self, messages: List[Message]) -> Optional[str]:
    """Extract the content of the last user message (for trigger pre-checks)."""
    for msg in reversed(messages):
      if hasattr(msg, "role") and msg.role == "user" and msg.content:
        return msg.content if isinstance(msg.content, str) else str(msg.content)
    return None

  @staticmethod
  def _build_routing_prompt(layer_name: str, query: str, context_str: str) -> str:
    """Build a precise, layer-specific YES/NO routing prompt.

    Generic prompts cause routing models to over-fire (always YES). Explicit
    criteria with positive/negative signals produce accurate routing decisions.
    """
    ctx_block = f"\nRecent conversation:\n{context_str}\n" if context_str else ""
    q = query[:300]

    if layer_name == "knowledge base":
      return (
        f"You are a routing system. Answer ONLY with YES or NO.\n\n"
        f"QUESTION: Does this query need the knowledge base?\n\n"
        f"KNOWLEDGE BASE contains factual documents: company policies, product info, procedures, uploaded content.\n\n"
        f"Answer YES when the query asks about:\n"
        f"- Company rules, benefits, policies (PTO, salary, leave, procedures)\n"
        f"- Product details, features, or documentation\n"
        f"- Factual questions about the organization or domain\n"
        f"- 'How does X work?', 'What is the policy for Y?', 'Tell me about Z'\n\n"
        f"Answer NO when the query is:\n"
        f"- Simple math or calculations ('add 1 and 2', 'what is 5*7')\n"
        f"- General conversation, greetings, or chit-chat\n"
        f"- Coding tasks, logic puzzles, or general reasoning\n"
        f"- Questions answerable from common world knowledge (no documents needed)\n"
        f"- Personal-only questions about the user (memory handles those){ctx_block}\n"
        f"Query: '{q}'\n\n"
        f"Answer YES or NO only:"
      )

    if layer_name == "memory":
      return (
        f"You are a routing system. Answer ONLY with YES or NO.\n\n"
        f"QUESTION: Does this query need personal memory recall?\n\n"
        f"MEMORY stores personal information about this user from past conversations.\n\n"
        f"Answer YES when the query involves:\n"
        f"- User's name, role, preferences, or personal details\n"
        f"- References to past interactions ('what did I tell you', 'remember when', 'last time')\n"
        f"- Possessive questions ('my name', 'my preference', 'my project')\n"
        f"- Follow-ups that require knowing who the user is or what they said before\n\n"
        f"Answer NO when the query is:\n"
        f"- Simple math or calculations ('add 1 and 2')\n"
        f"- General factual questions not specific to this user\n"
        f"- Topics fully answerable without user-specific context\n"
        f"- The very first message with no personal reference{ctx_block}\n"
        f"Query: '{q}'\n\n"
        f"Answer YES or NO only:"
      )

    if layer_name == "analysis/thinking":
      return (
        f"You are a routing system. Answer ONLY with YES or NO.\n\n"
        f"QUESTION: Does this query need extended step-by-step reasoning?\n\n"
        f"THINKING enables slow, careful analysis before the assistant responds.\n\n"
        f"Answer YES when the query involves:\n"
        f"- Multi-step math, logic proofs, or complex reasoning\n"
        f"- Code architecture, algorithm design, or debugging\n"
        f"- Strategic planning, trade-off analysis, or ambiguous decisions\n"
        f"- Tasks where rushing to an answer risks being wrong\n\n"
        f"Answer NO when the query is:\n"
        f"- Simple arithmetic ('add 1 and 2', 'what is 5+3')\n"
        f"- Direct factual lookups ('what is the PTO policy')\n"
        f"- Simple instructions ('send an email to X')\n"
        f"- Casual conversation or greetings{ctx_block}\n"
        f"Query: '{q}'\n\n"
        f"Answer YES or NO only:"
      )

    # Fallback for custom layer names
    ctx_section = f"Recent conversation:\n{context_str}\n" if context_str else ""
    return (
      f"You are a routing system. Answer ONLY with YES or NO.\n\n"
      f"QUESTION: Does this query require accessing the {layer_name}?\n\n"
      f"{ctx_section}"
      f"Query: '{q}'\n\n"
      f"Answer YES or NO only:"
    )

  async def _should_invoke_layer(
    self,
    layer_name: str,
    query: str,
    decision_prompt: Optional[str] = None,
    routing_model: Optional[Any] = None,
    messages: Optional[List[Message]] = None,
  ) -> bool:
    """Lightweight YES/NO pre-check: does this query need the given layer?

    Uses routing_model (if provided) or falls back to the agent's model.
    Includes recent conversation context so the gate has enough signal.
    Returns True on failure to default to running the layer (fail-open).
    """
    model = routing_model or self.model

    context_str = ""
    if messages:
      recent = messages[-3:]
      context_str = "\n".join(
        f"{m.role}: {(m.content[:200] if isinstance(m.content, str) else str(m.content)[:200])}"
        for m in recent
        if m.role in ("user", "assistant") and m.content
      )

    if decision_prompt:
      prompt = decision_prompt
    else:
      prompt = self._build_routing_prompt(layer_name, query, context_str)

    try:
      response = await model.ainvoke(
        messages=[Message(role="user", content=prompt)],
        assistant_message=Message(role="assistant", content=""),
      )
      answer = (response.content or "").strip().upper()
      return answer in ("YES", "Y")
    except Exception as e:
      from definable.utils.log import log_warning

      log_warning(f"Layer routing check failed for '{layer_name}', defaulting to run: {e}")
      return True  # fail-open: run the layer if routing fails

  def _build_layer_guide(self, context: Optional[RunContext] = None) -> str:
    """Build a capabilities menu section for the system prompt.

    Only included when at least one layer has a custom description
    or a non-default trigger (i.e., the model guides activation).

    When ``context`` is provided, reflects the actual fetch state:
    layers that were retrieved are marked as such; layers that were
    configured with trigger="auto" but not fetched this turn are noted
    as available-but-not-retrieved.
    """
    from definable.agent.config import (
      DEFAULT_KNOWLEDGE_DESCRIPTION,
      DEFAULT_MEMORY_DESCRIPTION,
      DEFAULT_RESEARCH_DESCRIPTION,
      DEFAULT_THINKING_DESCRIPTION,
    )

    active = context.active_layers if context is not None else set()
    items: List[str] = []

    # Memory layer
    if self.memory:
      needs_guide = bool(self.memory.description) or self.memory.trigger != "always"
      if needs_guide:
        desc = self.memory.description or DEFAULT_MEMORY_DESCRIPTION
        if "memory" in active:
          items.append(f"- **Memory** [retrieved this turn]: {desc}")
        elif self.memory.trigger == "auto":
          items.append(f"- **Memory** [available, not retrieved this turn]: {desc}")
        else:
          items.append(f"- **Memory**: {desc}")

    # Knowledge layer
    if self._knowledge:
      needs_guide = bool(self._knowledge.description) or self._knowledge.trigger != "always"
      if needs_guide:
        desc = self._knowledge.description or DEFAULT_KNOWLEDGE_DESCRIPTION
        if "knowledge" in active:
          items.append(f"- **Knowledge Base** [retrieved this turn]: {desc}")
        elif self._knowledge.trigger == "auto":
          items.append(f"- **Knowledge Base** [available, not retrieved this turn]: {desc}")
        else:
          items.append(f"- **Knowledge Base**: {desc}")

    # Thinking layer
    if self._thinking and self._thinking.enabled:
      needs_guide = bool(self._thinking.description) or self._thinking.trigger != "always"
      if needs_guide:
        desc = self._thinking.description or DEFAULT_THINKING_DESCRIPTION
        items.append(f"- **Analysis**: {desc}")

    # Deep research layer
    if self._researcher and self._deep_research_config:
      needs_guide = bool(self._deep_research_config.description) or self._deep_research_config.trigger != "always"
      if needs_guide:
        desc = self._deep_research_config.description or DEFAULT_RESEARCH_DESCRIPTION
        items.append(f"- **Research**: {desc}")

    if not items:
      return ""

    lines = [
      "## Capabilities Available",
      "",
      "The following capabilities are available and will activate when relevant:",
      "",
    ] + items
    return "\n".join(lines)

  async def _evaluate_layer_trigger(
    self,
    trigger: Literal["always", "auto", "never"],
    callback: Callable[[], Awaitable[List[RunOutputEvent]]],
    *,
    layer_name: str = "",
    query_messages: Optional[List[Message]] = None,
    all_messages: Optional[List[Message]] = None,
    decision_prompt: Optional[str] = None,
    routing_model: Optional[Any] = None,
  ) -> List[RunOutputEvent]:
    """Evaluate a layer trigger and conditionally run the callback.

    Returns callback's events if the layer runs, [] if skipped.
    Fails open on 'auto' gate errors (returns callback result).
    """
    if trigger == "always":
      return await callback()
    if trigger == "auto":
      if query_messages is None:
        return []
      query = self._extract_last_user_query(query_messages)
      if query and await self._should_invoke_layer(
        layer_name,
        query,
        decision_prompt,
        routing_model,
        all_messages,
      ):
        return await callback()
    # "never" falls through
    return []

  def _should_store_memory(self) -> bool:
    """Return True if memory store should run this turn.

    Store fires on "always" and "auto" (we always persist what was said,
    regardless of whether recall ran). Only "never" disables storage.
    """
    if not self.memory:
      return False
    return self.memory.trigger != "never"

  async def _thinking_should_run(self, messages: List[Message]) -> bool:
    """Return True if the thinking layer should execute this turn."""
    if not (self._thinking and self._thinking.enabled):
      return False
    trigger = self._thinking.trigger
    if trigger == "always":
      return True
    if trigger == "auto":
      query = self._extract_last_user_query(messages)
      if query:
        return await self._should_invoke_layer("analysis/thinking", query)
    return False  # "never"

  async def _run_pre_execution_pipeline(
    self,
    context: RunContext,
    new_messages: List[Message],
    all_messages: List[Message],
  ) -> List[RunOutputEvent]:
    """Pre-execution pipeline: readers → knowledge → research → memory recall.

    Populates context fields (knowledge_context, research_context, memory_context,
    readers_context, active_layers) consumed by _execute_run() and arun_stream().
    """
    events: List[RunOutputEvent] = []

    # File reading (before knowledge — extracted content may inform the query)
    events.extend(await self._readers_extract(context, new_messages))

    # Knowledge retrieval
    if self._knowledge and self._knowledge.enabled:
      events.extend(
        await self._evaluate_layer_trigger(
          trigger=self._knowledge.trigger,
          callback=lambda: self._knowledge_retrieve(context),
          layer_name="knowledge base",
          query_messages=all_messages,
          all_messages=all_messages,
          decision_prompt=self._knowledge.decision_prompt,
          routing_model=self._knowledge.routing_model,
        )
      )

    # Deep research (after knowledge, before memory)
    events.extend(await self._deep_research(context))

    # Memory recall
    if self.memory:
      events.extend(
        await self._evaluate_layer_trigger(
          trigger=self.memory.trigger,
          callback=lambda: self._memory_recall(context, new_messages),
          layer_name="memory",
          query_messages=new_messages,
          all_messages=all_messages,
          decision_prompt=self.memory.decision_prompt,
          routing_model=self.memory.routing_model,
        )
      )

    return events

  async def _execute_thinking(
    self,
    context: RunContext,
    invoke_messages: List[Message],
    tools: Dict[str, Function],
  ) -> "tuple[Optional[ThinkingOutput], list[ReasoningStep], list[Message]]":
    """Execute thinking phase: analyze query and produce reasoning steps.

    Returns:
      Tuple of (thinking_output, reasoning_steps, reasoning_messages)
    """
    import json

    from definable.agent.reasoning.step import ThinkingOutput, thinking_output_to_reasoning_steps
    from definable.utils.log import log_warning

    assert self._thinking is not None

    thinking_model = self._thinking.model or self.model

    # Use custom instructions if set, otherwise build context-aware prompt
    if self._thinking.instructions:
      thinking_prompt = self._thinking.instructions
    else:
      thinking_prompt = self._build_thinking_prompt(context, tools)

    # Emit ReasoningStarted
    self._emit(
      ReasoningStartedEvent(
        run_id=context.run_id,
        session_id=context.session_id,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
      )
    )

    # Build thinking messages: system prompt + user/assistant messages (no tools)
    thinking_messages: list[Message] = [Message(role="system", content=thinking_prompt)]
    for msg in invoke_messages:
      if msg.role in ("user", "assistant"):
        thinking_messages.append(msg)

    # Call model — use structured output only if provider supports it
    assistant_msg = Message(role="assistant")
    if thinking_model.supports_native_structured_outputs:
      response = await thinking_model.ainvoke(
        messages=thinking_messages,
        assistant_message=assistant_msg,
        response_format=ThinkingOutput,
      )
    else:
      # Append JSON schema instructions for models that don't support response_format
      thinking_messages[0] = Message(
        role="system",
        content=thinking_messages[0].content  # type: ignore[operator]
        + '\n\nRespond with ONLY valid JSON: {"analysis": "string", "approach": "string", "tool_plan": ["string"] | null}',
      )
      response = await thinking_model.ainvoke(
        messages=thinking_messages,
        assistant_message=assistant_msg,
      )

    # Parse ThinkingOutput from structured response
    from definable.agent.reasoning.step import ReasoningStep

    thinking_output: Optional[ThinkingOutput] = None
    reasoning_steps: list[ReasoningStep] = []
    if response.content:
      try:
        parsed = json.loads(response.content) if isinstance(response.content, str) else response.content
        if isinstance(parsed, dict):
          thinking_output = ThinkingOutput(**parsed)
          reasoning_steps = thinking_output_to_reasoning_steps(thinking_output)
      except Exception as e:
        log_warning(f"Failed to parse thinking response: {e}")
        # Graceful fallback
        thinking_output = ThinkingOutput(analysis="Could not parse", approach="Respond directly")  # type: ignore[call-arg]
        reasoning_steps = thinking_output_to_reasoning_steps(thinking_output)

    # Emit ReasoningStep for each step
    for step in reasoning_steps:
      self._emit(
        ReasoningStepEvent(
          run_id=context.run_id,
          session_id=context.session_id,
          agent_id=self.agent_id,
          agent_name=self.agent_name,
          reasoning_content=step.reasoning or "",
        )
      )

    # Build reasoning messages list for RunOutput
    reasoning_agent_messages = thinking_messages + [
      Message(
        role="assistant",
        content=response.content,
        metrics=response.response_usage,  # type: ignore[arg-type]
      )
    ]

    # Emit ReasoningCompleted
    self._emit(
      ReasoningCompletedEvent(
        run_id=context.run_id,
        session_id=context.session_id,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
      )
    )

    return thinking_output, reasoning_steps, reasoning_agent_messages

  @staticmethod
  def _format_reasoning_context(steps: "list[ReasoningStep]") -> str:
    """Format reasoning steps into XML context for system prompt injection."""
    if not steps:
      return ""

    lines = ["<reasoning>"]
    for i, step in enumerate(steps, 1):
      lines.append(f'  <step number="{i}">')
      if step.title:
        lines.append(f"    <title>{step.title}</title>")
      if step.reasoning:
        lines.append(f"    <reasoning>{step.reasoning}</reasoning>")
      if step.action:
        lines.append(f"    <action>{step.action}</action>")
      if step.confidence is not None:
        lines.append(f"    <confidence>{step.confidence}</confidence>")
      lines.append("  </step>")
    lines.append("</reasoning>")
    return "\n".join(lines)

  # --- Internal Methods ---

  async def _build_invoke_messages(
    self,
    context: RunContext,
    messages: List[Message],
    tools: Dict[str, Function],
  ) -> tuple:
    """Build the invoke message list with system prompt, thinking, knowledge, memory, readers.

    This consolidates the duplicated message-building logic from _execute_run and arun_stream.

    Returns:
        (invoke_messages, reasoning_steps, reasoning_agent_messages)
    """
    invoke_messages = messages.copy()

    # Build system message: instructions → skills → layer guide → thinking → knowledge → memory
    system_content = self.instructions or ""

    # Append skill instructions
    skill_instructions = self._build_skill_instructions()
    if skill_instructions:
      system_content = f"{system_content}\n\n{skill_instructions}" if system_content else skill_instructions

    # Layer guide
    layer_guide = self._build_layer_guide(context)
    if layer_guide:
      system_content = f"{system_content}\n\n{layer_guide}" if system_content else layer_guide

    # Thinking phase (BEFORE knowledge/memory)
    reasoning_steps: Optional[list] = None
    reasoning_agent_messages: Optional[list] = None
    if await self._thinking_should_run(invoke_messages):
      thinking_output, reasoning_steps, reasoning_agent_messages = await self._execute_thinking(
        context,
        invoke_messages,
        tools,
      )
      if thinking_output:
        injection = self._format_thinking_injection(thinking_output)
        if injection:
          system_content = f"{system_content}\n\n{injection}" if system_content else injection

    # Append knowledge context
    if context.knowledge_context:
      position = (context.metadata or {}).get("_knowledge_position", "system")
      if position == "system":
        system_content = f"{system_content}\n\n{context.knowledge_context}" if system_content else context.knowledge_context

    # Append research context
    if context.research_context:
      system_content = f"{system_content}\n\n{context.research_context}" if system_content else context.research_context

    # Append memory context
    if context.memory_context:
      system_content = f"{system_content}\n\n{context.memory_context}" if system_content else context.memory_context

    if system_content:
      invoke_messages.insert(0, Message(role="system", content=system_content))

    # Inject extracted file content into the last user message
    if context.readers_context:
      for i in range(len(invoke_messages) - 1, -1, -1):
        if invoke_messages[i].role == "user":
          original_content = invoke_messages[i].content or ""
          invoke_messages[i] = Message(
            role="user",
            content=f"{context.readers_context}\n\n{original_content}",
            images=invoke_messages[i].images,
            videos=invoke_messages[i].videos,
            audio=invoke_messages[i].audio,
          )
          break

    return invoke_messages, reasoning_steps, reasoning_agent_messages

  async def _execute_run(
    self,
    context: RunContext,
    messages: List[Message],
    run_input: Optional[RunInput] = None,
    cancellation_token: Optional[CancellationToken] = None,
  ) -> RunOutput:
    """Core execution logic — delegates to AgentLoop."""
    # Prepare tools with injected context
    tools = self._prepare_tools_for_run(context)

    # Emit RunStarted
    started_event = RunStartedEvent(
      run_id=context.run_id,
      session_id=context.session_id,
      agent_id=self.agent_id,
      agent_name=self.agent_name,
      model=self.model.id,
      model_provider=self.model.provider,  # type: ignore[arg-type]
      run_input=run_input,
    )
    self._emit(started_event)
    await self._event_bus.emit(started_event)

    try:
      # Build invoke messages (system prompt, thinking, knowledge, memory, readers)
      invoke_messages, reasoning_steps, reasoning_agent_messages = await self._build_invoke_messages(context, messages, tools)

      # Create the loop
      loop = AgentLoop(
        model=self.model,
        tools=tools,
        messages=invoke_messages,
        context=context,
        config=self.config,
        streaming=False,
        cancellation_token=cancellation_token,
        compression_manager=self._compression_manager,
        guardrails=self.guardrails,
        emit_fn=self._emit,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
      )

      # Run the loop, collect events
      final_content: Optional[str] = None
      final_metrics: Optional[Metrics] = None

      async for event in loop.run():
        await self._event_bus.emit(event)
        self._emit(event)

        if isinstance(event, RunCompletedEvent):
          final_content = event.content
          final_metrics = event.metrics
        elif isinstance(event, RunPausedEvent):
          # Build paused RunOutput
          output_messages = [m for m in loop.messages if m.role != "system"]
          return RunOutput(
            run_id=context.run_id,
            session_id=context.session_id,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tools=loop.tool_executions or None,
            messages=output_messages,
            model=self.model.id,
            model_provider=self.model.provider,
            status=RunStatus.paused,
            session_state=context.session_state,
            requirements=event.requirements,
          )

      # Build output messages (excluding system message)
      output_messages = [m for m in loop.messages if m.role != "system"]

      return RunOutput(
        run_id=context.run_id,
        session_id=context.session_id,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
        content=final_content,
        tools=loop.tool_executions or None,
        metrics=final_metrics,
        messages=output_messages,
        model=self.model.id,
        model_provider=self.model.provider,
        status=RunStatus.completed,
        session_state=context.session_state,
        reasoning_steps=reasoning_steps or None,
        reasoning_messages=reasoning_agent_messages or None,
        reasoning_content=self._format_reasoning_context(reasoning_steps) if reasoning_steps else None,
      )

    except AgentCancelled:
      from definable.agent.events import RunCancelledEvent

      cancelled_event = RunCancelledEvent(
        run_id=context.run_id,
        session_id=context.session_id,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
        reason="Cancelled via CancellationToken",
      )
      self._emit(cancelled_event)
      await self._event_bus.emit(cancelled_event)
      return RunOutput(
        run_id=context.run_id,
        session_id=context.session_id,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
        status=RunStatus.cancelled,
        model=self.model.id,
        model_provider=self.model.provider,
      )

    except Exception as e:
      self._emit(
        RunErrorEvent(
          run_id=context.run_id,
          session_id=context.session_id,
          agent_id=self.agent_id,
          error_type=type(e).__name__,
          content=str(e),
        )
      )
      raise

  def _init_skills(self) -> None:
    """Initialize skills: call setup(), validate names."""
    seen_names: Dict[str, Skill] = {}
    for skill in self.skills:
      # Warn on duplicate skill names (last one wins for tools)
      if skill.name in seen_names:
        from definable.utils.log import log_warning

        log_warning(f"Duplicate skill name '{skill.name}' — tools from the later skill will override earlier ones.")
      seen_names[skill.name] = skill

      # Call setup() for one-time initialization (non-fatal)
      try:
        skill.setup()
        skill._initialized = True
      except Exception as e:
        from definable.utils.log import log_error

        log_error(f"Skill '{skill.name}' setup() failed: {e}")

  def _build_skill_instructions(self) -> str:
    """Collect instructions from all skills into a merged block.

    Returns:
      A single string with all skill instructions separated by
      blank lines, or empty string if no skills provide instructions.
    """
    parts: List[str] = []
    for skill in self.skills:
      try:
        text = skill.get_instructions()
      except Exception:
        text = ""
      if text and text.strip():
        parts.append(text.strip())
    return "\n\n".join(parts)

  def _flatten_tools(self) -> Dict[str, Function]:
    """
    Flatten tools from skills, toolkits, and direct tools into a single dict.

    Processing order (later entries override earlier ones):
      1. Skill tools (lowest priority)
      2. Toolkit tools
      3. Direct tools (highest priority — explicit always wins)
    """
    result: Dict[str, Function] = {}

    # 1. Process skill tools first (lowest priority)
    for skill in self.skills:
      try:
        skill_tools = skill.tools
      except Exception:
        skill_tools = []
      for fn in skill_tools:
        # Merge skill dependencies into the tool
        if skill.dependencies:
          existing_deps = getattr(fn, "_dependencies", None) or {}
          fn._dependencies = {**existing_deps, **skill.dependencies}
        result[fn.name] = fn

    # 2. Process toolkits (can override skill tools)
    for toolkit in self.toolkits:
      for fn in toolkit.tools:
        # Merge toolkit dependencies with tool's existing dependencies
        if toolkit.dependencies:
          existing_deps = getattr(fn, "_dependencies", None) or {}
          fn._dependencies = {**existing_deps, **toolkit.dependencies}
        result[fn.name] = fn

    # 3. Process direct tools (highest priority — can override anything)
    for fn in self.tools:
      result[fn.name] = fn

    return result

  def _init_tracing(self) -> Optional[TraceWriter]:
    """Initialize trace writer using resolved tracing config."""
    if self._tracing_config and self._tracing_config.exporters:
      return TraceWriter(self._tracing_config)
    return None

  # --- Layer Resolvers (called once during __init__) ---

  def _resolve_memory(self, memory: Any) -> Optional[Any]:
    """Resolve memory param to Memory | None.

    Accepts:
      - False/None → None
      - True → Memory(store=InMemoryStore())
      - Memory instance → pass through
    """
    if memory is False or memory is None:
      return None
    if memory is True:
      from definable.memory.manager import Memory
      from definable.memory.store.in_memory import InMemoryStore

      return Memory(store=InMemoryStore())

    # Memory instance — pass through
    return memory

  def _resolve_knowledge(self, knowledge: Any) -> Optional[Any]:
    """Resolve knowledge param to Knowledge | None.

    Accepts:
      - False/None → None
      - True → ValueError
      - Knowledge instance → pass through (has agent-integration fields)
    """
    if knowledge is False or knowledge is None:
      return None
    if knowledge is True:
      raise ValueError("knowledge=True is not supported. Pass a Knowledge instance (Agent(knowledge=Knowledge(vector_db=..., top_k=5))).")

    # Knowledge instance — pass through directly
    return knowledge

  @staticmethod
  def _resolve_tracing(tracing_param: Any, config: Optional[AgentConfig]) -> Optional[Any]:
    """Resolve tracing param to Tracing | None.

    Accepts Tracing, bool, or None.
    Direct param takes precedence; config.tracing is a fallback.
    """
    from definable.agent.tracing.base import Tracing as _Tracing

    if tracing_param is False:
      # Fall back to config.tracing if set
      return config.tracing if config else None
    if tracing_param is True:
      return _Tracing()
    if isinstance(tracing_param, _Tracing):
      return tracing_param
    return config.tracing if config else None

  def _init_compression(self) -> Optional["CompressionManager"]:
    """Initialize compression manager if compression is configured."""
    if self.config.compression and self.config.compression.enabled:
      from definable.agent.compression import CompressionManager

      # Use specified model or fall back to agent's model
      compression_model: Optional["Model"] = None
      config_model = self.config.compression.model
      if config_model is None:
        compression_model = self.model
      elif isinstance(config_model, str):
        # String model specs are not fully supported - use agent's model
        # (get_model() in CompressionManager only supports 'aimlapi' provider)
        compression_model = self.model
      else:
        # Model instance passed directly
        compression_model = config_model

      return CompressionManager(
        model=compression_model,
        compress_tool_results=True,
        compress_tool_results_limit=self.config.compression.tool_results_limit,
        compress_token_limit=self.config.compression.token_limit,
        compress_tool_call_instructions=self.config.compression.instructions,
      )
    return None

  def _prepare_tools_for_run(self, context: RunContext) -> Dict[str, Function]:
    """
    Create tool copies with injected context (thread-safe).

    Each run gets its own tool instances to avoid state leakage
    between concurrent runs.
    """
    tools: Dict[str, Function] = {}
    for name, fn in self._tools_dict.items():
      # model_copy creates a new instance
      tool_copy = fn.model_copy()
      tool_copy._run_context = context
      # Merge existing deps (from toolkit) with config deps
      existing_deps = fn._dependencies or {}
      config_deps = self.config.dependencies or {}
      tool_copy._dependencies = {**existing_deps, **config_deps}
      tool_copy._session_state = context.session_state
      tools[name] = tool_copy
    return tools

  def _normalize_instruction(
    self,
    instruction: Union[str, Message, List[Message]],
    images: Optional[List[Image]] = None,
    videos: Optional[List[Video]] = None,
    audio: Optional[List[Audio]] = None,
    files: Optional[List[File]] = None,
  ) -> List[Message]:
    """Normalize various input types to List[Message]."""
    if isinstance(instruction, str):
      return [
        Message(
          role="user",
          content=instruction,
          images=images,
          videos=videos,
          audio=audio,
          files=files,
        )
      ]
    elif isinstance(instruction, Message):
      return [instruction]
    elif isinstance(instruction, list):
      return instruction
    raise TypeError(f"Unexpected instruction type: {type(instruction)}")

  def _emit(self, event: BaseRunOutputEvent) -> None:
    """Emit event to trace writer (fire-and-forget)."""
    if self._trace_writer:
      with contextlib.suppress(Exception):
        # Tracing should never break the main flow
        self._trace_writer.write(event)

  # --- Triggers ---

  @property
  def triggers(self) -> List[Any]:
    """Registered triggers (read-only copy)."""
    return list(self._triggers)

  def on(self, trigger: Any) -> Callable:
    """Register a trigger handler.

    Can be used as a decorator::

      @agent.on(Webhook("/github"))
      async def handle_github(event):
          ...

    Args:
      trigger: A BaseTrigger instance (Webhook, Cron, EventTrigger).

    Returns:
      Decorator that registers the handler and returns the original function.
    """

    def decorator(fn: Callable) -> Callable:
      trigger.handler = fn
      trigger.agent = self
      self._triggers.append(trigger)
      return fn

    return decorator

  def emit(self, event_name: str, data: Optional[Any] = None) -> None:
    """Fire all EventTriggers matching *event_name* (fire-and-forget).

    Args:
      event_name: Name of the event to fire.
      data: Optional data dict to include in the TriggerEvent body.
    """
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
      pass  # No running loop — skip

  # --- Interfaces ---

  def add_interface(self, interface: "BaseInterface") -> "Agent":
    """Register an interface with this agent.

    Binds the interface to this agent and stores it for use with serve().

    Args:
      interface: BaseInterface instance to register.

    Returns:
      Self for method chaining.
    """
    interface.bind(self)
    self._interfaces.append(interface)
    return self

  async def aserve(
    self,
    *interfaces: "BaseInterface",
    name: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    enable_server: Optional[bool] = None,
    dev: bool = False,
  ) -> None:
    """Async entry point: start the full agent runtime.

    Starts registered interfaces, webhook/cron triggers, and an HTTP
    server in a single event loop.  Use :meth:`serve` for the sync
    version.

    Args:
      *interfaces: Additional interfaces to run (merged with registered ones).
      name: Optional prefix for log messages (defaults to agent_name).
      host: Host to bind the HTTP server to.
      port: Port for the HTTP server.
      enable_server: Force-enable/disable the HTTP server.  When *None*
        (default), the server starts if any Webhook triggers exist.
      dev: Enable development mode with Swagger docs and info-level logging.
    """
    from definable.agent.runtime.runner import AgentRuntime

    # Merge passed interfaces with registered ones
    all_interfaces = list(self._interfaces)
    for iface in interfaces:
      if iface.agent is None:
        iface.bind(self)
      if iface not in all_interfaces:
        all_interfaces.append(iface)

    runtime = AgentRuntime(
      agent=self,
      interfaces=all_interfaces or None,
      host=host,
      port=port,
      enable_server=enable_server,
      name=name,
      dev=dev,
    )
    await runtime.start()

  def serve(
    self,
    *interfaces: "BaseInterface",
    name: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    enable_server: Optional[bool] = None,
    dev: bool = False,
  ) -> None:
    """Sync entry point: start the full agent runtime.

    Blocking call that starts interfaces, triggers, and an HTTP server.
    Equivalent to ``asyncio.run(agent.aserve(...))``.

    When ``dev=True``, enables hot-reload mode: the parent process
    watches for ``.py`` file changes and automatically restarts the
    server.  Swagger docs are available at ``/docs``.

    Args:
      *interfaces: Additional interfaces to run (merged with registered ones).
      name: Optional prefix for log messages (defaults to agent_name).
      host: Host to bind the HTTP server to.
      port: Port for the HTTP server.
      enable_server: Force-enable/disable the HTTP server.  When *None*
        (default), the server starts if any Webhook triggers exist.
      dev: Enable development mode with hot reload and Swagger docs.
    """
    if dev:
      from definable.agent.runtime._dev import is_dev_child, run_dev_mode

      if not is_dev_child():
        run_dev_mode()
        return

    asyncio.run(
      self.aserve(
        *interfaces,
        name=name,
        host=host,
        port=port,
        enable_server=enable_server,
        dev=dev,
      )
    )

  # --- Replay & Compare ---

  def replay(
    self,
    *,
    run_output: Optional[RunOutput] = None,
    trace_file: Optional[Union[str, "Path"]] = None,
    run_id: Optional[str] = None,
    events: Optional[List[BaseRunOutputEvent]] = None,
    model: Optional["Model"] = None,
    instructions: Optional[str] = None,
    tools: Optional[List[Function]] = None,
  ) -> Union["Replay", RunOutput]:
    """Load a past run for inspection, or re-execute with overrides.

    Provide exactly one source: run_output, trace_file, run_id, or events.
    If override args (model, instructions, tools) are also given, the
    original input is re-executed live and a RunOutput is returned.

    Args:
      run_output: A RunOutput from a previous agent.run() call.
      trace_file: Path to a JSONL trace file.
      run_id: Run ID to find in the agent's configured trace directory.
      events: Pre-loaded list of trace events.
      model: Override model for re-execution.
      instructions: Override instructions for re-execution.
      tools: Override tools for re-execution.

    Returns:
      Replay for inspection, or RunOutput if re-executing.
    """
    try:
      loop = asyncio.get_running_loop()
    except RuntimeError:
      loop = None

    if loop and loop.is_running():
      import concurrent.futures

      with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(
          asyncio.run,
          self.areplay(
            run_output=run_output,
            trace_file=trace_file,
            run_id=run_id,
            events=events,
            model=model,
            instructions=instructions,
            tools=tools,
          ),
        )
        return future.result()
    else:
      new_loop = asyncio.new_event_loop()
      asyncio.set_event_loop(new_loop)
      try:
        return new_loop.run_until_complete(
          self.areplay(
            run_output=run_output,
            trace_file=trace_file,
            run_id=run_id,
            events=events,
            model=model,
            instructions=instructions,
            tools=tools,
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

  async def areplay(
    self,
    *,
    run_output: Optional[RunOutput] = None,
    trace_file: Optional[Union[str, "Path"]] = None,
    run_id: Optional[str] = None,
    events: Optional[List[BaseRunOutputEvent]] = None,
    model: Optional["Model"] = None,
    instructions: Optional[str] = None,
    tools: Optional[List[Function]] = None,
  ) -> Union["Replay", RunOutput]:
    """Async version of replay(). See replay() for documentation."""
    from pathlib import Path as _Path

    from definable.agent.replay import Replay

    # Build Replay from the provided source
    replay: Optional[Replay] = None

    if run_output is not None:
      replay = Replay.from_run_output(run_output)
    elif events is not None:
      replay = Replay.from_events(events, run_id=run_id)
    elif trace_file is not None:
      replay = Replay.from_trace_file(_Path(trace_file), run_id=run_id)
    elif run_id is not None:
      # Auto-discover trace file from configured trace dir
      replay = self._replay_from_trace_dir(run_id)
    else:
      raise ValueError("Provide one of: run_output, trace_file, run_id, or events")

    # If no overrides, return the Replay for inspection
    has_overrides = model is not None or instructions is not None or tools is not None
    if not has_overrides:
      return replay

    # Re-execute: extract original input and run with overrides
    original_input = replay.input
    if original_input is None:
      raise ValueError("Cannot re-execute: original run input not available in the replay source")

    # Create a new agent with overrides applied
    re_agent = Agent(
      model=model or self.model,
      tools=tools if tools is not None else self.tools,
      toolkits=self.toolkits,
      skills=self.skills,
      instructions=instructions if instructions is not None else self.instructions,
      config=self.config,
    )

    input_content = original_input.input_content

    return await re_agent.arun(
      input_content,
      images=list(original_input.images) if original_input.images else None,
      videos=list(original_input.videos) if original_input.videos else None,
      audio=list(original_input.audios) if original_input.audios else None,
      files=list(original_input.files) if original_input.files else None,
    )

  def _replay_from_trace_dir(self, run_id: str) -> "Replay":
    """Find a run_id in the agent's configured trace directory."""
    from definable.agent.replay import Replay

    if not (self.config.tracing and self.config.tracing.exporters):
      raise ValueError("No tracing configured on this agent; cannot auto-discover trace files. Provide trace_file= instead.")

    from definable.agent.tracing.jsonl import JSONLExporter

    for exporter in self.config.tracing.exporters:
      if isinstance(exporter, JSONLExporter):
        trace_dir = exporter.trace_dir
        # Scan JSONL files for the run_id
        for jsonl_path in sorted(trace_dir.glob("*.jsonl")):
          # Quick check: scan file text for run_id before full parse
          try:
            text = jsonl_path.read_text(encoding="utf-8")
          except OSError:
            continue
          if run_id not in text:
            continue
          # Full parse
          replay = Replay.from_trace_file(jsonl_path, run_id=run_id)
          if replay.run_id == run_id:
            return replay

    raise ValueError(f"Run ID {run_id!r} not found in any trace file")

  def compare(
    self,
    a: Union["Replay", RunOutput],
    b: Union["Replay", RunOutput],
  ) -> "ReplayComparison":
    """Compare two runs side-by-side.

    Args:
      a: First run (Replay or RunOutput).
      b: Second run (Replay or RunOutput).

    Returns:
      ReplayComparison with diffs for content, cost, tokens, and tool calls.
    """
    from definable.agent.replay.compare import compare_runs

    return compare_runs(a, b)

  def __repr__(self) -> str:
    parts = [f"model={self.model.id!r}", f"tools={len(self._tools_dict)}"]
    if self.skills:
      parts.append(f"skills={len(self.skills)}")
    parts.append(f"name={self.agent_name!r}")
    return f"Agent({', '.join(parts)})"
