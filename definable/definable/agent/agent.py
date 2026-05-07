"""Agent class - production-grade wrapper around model execution."""

import asyncio
import contextlib
import dataclasses
from typing import (
  TYPE_CHECKING,
  Any,
  AsyncGenerator,
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
  RunCompletedEvent,
  RunContext,
  RunErrorEvent,
  RunInput,
  RunOutput,
  RunOutputEvent,
  RunStartedEvent,
  RunStatus,
)
from definable.skill.base import Skill
from definable.tool.function import Function
from pydantic import BaseModel

if TYPE_CHECKING:
  from pathlib import Path

  from definable.agent.auth.base import AuthProvider
  from definable.agent.compression import Compression, CompressionManager
  from definable.agent.context import Context
  from definable.agent.context.deferred import DeferredToolManager
  from definable.agent.context.manager import ContextManager
  from definable.agent.observability.config import ObservabilityConfig
  from definable.agent.guardrail.base import Guardrails
  from definable.agent.interface.base import BaseInterface
  from definable.agent.interface.gateway import InterfaceGateway
  from definable.agent.pipeline.debug import DebugConfig
  from definable.agent.pipeline.pipeline import Pipeline
  from definable.agent.pipeline.state import LoopState
  from definable.agent.pipeline.sub_agent import SubAgentPolicy
  from definable.agent.reasoning.thinking import Thinking
  from definable.agent.tracing.base import Tracing
  from definable.knowledge import Knowledge
  from definable.memory.manager import Memory
  from definable.model.base import Model
  from definable.agent.reasoning.step import ReasoningStep, ThinkingOutput
  from definable.agent.replay import Replay, ReplayComparison
  from definable.agent.research.config import DeepResearchConfig
  from definable.agent.research.engine import DeepResearch
  from definable.agent.trigger.base import BaseTrigger
  from definable.reader.base import BaseReader
  from definable.skill.registry import SkillRegistry


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

  # _EFFORT_PROMPTS moved to definable.agent.layers

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
    thinking: Union[bool, "Thinking", None] = None,
    compression: Union[bool, "Compression", None] = None,
    context: Union[bool, "Context", None] = None,
    deep_research: Union[bool, "DeepResearchConfig", "DeepResearch", None] = None,
    # ── Tools ───────────────────────────────────────────────
    tools: Optional[List[Function]] = None,
    toolkits: Optional[List[Toolkit]] = None,
    skills: Optional[List[Skill]] = None,
    skill_registry: Optional["SkillRegistry"] = None,
    # ── Observability ───────────────────────────────────────
    tracing: Union[bool, "Tracing", None] = False,
    debug: Union[bool, "DebugConfig", None] = False,
    observability: Union[bool, "ObservabilityConfig", None] = False,
    # ── Advanced ───────────────────────────────────────
    sub_agents: Union[bool, "SubAgentPolicy", None] = None,
    # ── Media ────────────────────────────────────────────────
    audio_transcriber: Union[bool, Any, None] = None,
    # ── Security ──────────────────────────────────────────────
    security: Union[bool, Any, None] = None,
    # ── Usage Tracking ───────────────────────────────────────
    usage: Union[bool, Any, None] = None,
    # ── Interfaces ──────────────────────────────────────────
    interfaces: Union["BaseInterface", List["BaseInterface"], None] = None,
    gateway: Optional["InterfaceGateway"] = None,
    # ── Support ─────────────────────────────────────────────
    readers: Union[List["BaseReader"], bool, None] = None,
    guardrails: Optional["Guardrails"] = None,
    # ── HITL ──────────────────────────────────────────────
    permission_resolver: Optional[Any] = None,
    permission_defaults: Optional[Dict[str, Any]] = None,
    question_resolver: Optional[Any] = None,
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
            Uses on-demand mode: only a compact XML catalog is injected
            into the system prompt; the model activates individual skills
            via the ``activate_skill`` tool. Override by calling
            ``registry.as_eager()`` or ``registry.as_on_demand()`` directly
            and passing the result to ``skills=``.
        instructions: System instructions for the agent.
        memory: Optional Memory instance for session history.
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
        audio_transcriber: Optional audio transcription backend. When set,
            audio in incoming messages is automatically transcribed to text
            before reaching the model. Accepts True (default OpenAITranscriber
            using Whisper), an AudioTranscriber instance (custom backend), or
            None (disabled — audio passes through raw to the model).
        config: Optional advanced configuration settings.
    """
    # Direct attributes — resolve string model shorthand
    from definable.agent.resolution import resolve_model

    self.model: "Model" = resolve_model(model)
    self.tools = tools or []
    self.toolkits = toolkits or []
    self.skills = skills or []
    self.instructions: Optional[str] = "\n".join(str(i) for i in instructions) if isinstance(instructions, list) else instructions
    from definable.agent.resolution import init_readers

    self.readers = init_readers(readers)
    self.guardrails = guardrails

    # Optional config for advanced settings
    self.config = config or AgentConfig()
    if name is not None:
      self.config = dataclasses.replace(self.config, agent_name=name)

    # Resolve memory: Memory | bool → Memory | None
    from definable.agent.resolution import resolve_memory, resolve_memory_embedder

    self.memory = resolve_memory(memory)
    resolve_memory_embedder(self.memory, self.model)
    # v2 memory: auto-inject the memory-manager skill into the system prompt
    if self.memory and hasattr(self.memory, "get_skill"):
      self.skills = list(self.skills) + [self.memory.get_skill()]

    # Resolve knowledge: Knowledge | bool → Knowledge | None
    from definable.agent.resolution import (
      resolve_audio_transcriber,
      resolve_compression,
      resolve_context,
      resolve_debug,
      resolve_deep_research,
      resolve_deferred_tools,
      resolve_knowledge,
      resolve_observability,
      resolve_security,
      resolve_sub_agents,
      resolve_thinking,
      resolve_tracing,
      resolve_usage,
    )

    self._knowledge: Optional["Knowledge"] = resolve_knowledge(knowledge)

    # Resolve tracing → debug → observability (each may augment tracing exporters)
    self._tracing_config: Optional["Tracing"] = resolve_tracing(tracing, self.config)
    self._debug_config, self._tracing_config = resolve_debug(debug, self._tracing_config)
    self._observability_config, self._observability_exporter, self._tracing_config = resolve_observability(observability, self._tracing_config)

    # Thinking, deep research, sub-agents
    self._thinking: Optional["Thinking"] = resolve_thinking(thinking)
    self._deep_research_config, self._prebuilt_researcher = resolve_deep_research(deep_research)
    self._sub_agent_policy = resolve_sub_agents(sub_agents)

    # Audio, security, usage
    self._audio_transcriber = resolve_audio_transcriber(audio_transcriber)
    self._security, self.guardrails = resolve_security(security, self.guardrails)
    self._usage_tracker = resolve_usage(usage)

    # Convert skill_registry to on-demand skill (model picks skills based on query)
    if skill_registry is not None:
      from definable.skill.registry import SkillRegistry

      if isinstance(skill_registry, SkillRegistry):
        self.skills.append(skill_registry.as_on_demand())

    # Initialize skills (call setup, validate)
    from definable.agent.resolution import flatten_tools, init_skills, init_tracing

    init_skills(self.skills)

    # Internal state
    self._tools_dict: Dict[str, Function] = flatten_tools(self.skills, self.toolkits, self.tools)
    self._trace_writer: Optional[TraceWriter] = init_tracing(self._tracing_config)
    self._compression_manager: Optional["CompressionManager"] = resolve_compression(compression, self.model)
    self._context_manager: Optional["ContextManager"] = resolve_context(context, self.model)
    self._deferred_tool_manager: Optional["DeferredToolManager"] = resolve_deferred_tools(self._context_manager, self._tools_dict)
    self._middleware: List[Middleware] = []
    self._output_validators: List[Callable[..., Any]] = []
    self._interfaces: List["BaseInterface"] = []
    self._gateway: Optional["InterfaceGateway"] = None

    # HITL: Permission service + question resolver (must init before interface bind)
    self._permission_service: Optional[Any] = None
    self._question_resolver: Optional[Any] = None
    if permission_resolver is not None or permission_defaults is not None:
      from definable.agent.hitl.permissions import PermissionService
      from definable.agent.hitl.settings import Settings

      self._permission_service = PermissionService(
        resolver=permission_resolver,
        defaults=dict(permission_defaults or {}),
        settings=Settings.load(),
      )
    if question_resolver is not None:
      self._question_resolver = question_resolver

    # Resolve interfaces passed at construction (after HITL init so bind() can check)
    if interfaces is not None:
      iface_list = interfaces if isinstance(interfaces, list) else [interfaces]
      for iface in iface_list:
        iface.bind(self)
        self._interfaces.append(iface)

    # Resolve gateway passed at construction
    if gateway is not None:
      gateway._bind_agent(self)
      for iface in self._interfaces:
        if iface not in gateway._interfaces:
          gateway.add(iface)
      self._gateway = gateway
    self._triggers: List[Any] = []
    self._before_hooks: List[Callable] = []
    self._after_hooks: List[Callable] = []
    self._auth: Optional["AuthProvider"] = None
    self._started = False
    self._pending_memory_tasks: list[asyncio.Task] = []
    self._event_bus: EventBus = EventBus()
    self._agent_owned_toolkits: list[Any] = []
    self._toolkit_init_lock: asyncio.Lock = asyncio.Lock()
    self._session_id_explicit = session_id is not None
    self.session_id = session_id or str(uuid4())

    # Build pipeline (deprecated — harness.py is the primary path)
    # Kept for backward compat: agent.pipeline, agent.hook() still work
    self._pipeline = self._build_pipeline()

    # Deep research engine (prebuilt instance or lazy init from config)
    self._researcher: Optional["DeepResearch"] = self._prebuilt_researcher or (
      self._init_deep_research(self._deep_research_config) if self._deep_research_config else None
    )

  # --- Pipeline ---

  def _build_pipeline(self) -> "Pipeline":
    """Build the default pipeline from this agent's config.

    Called once during __init__. The pipeline is reused for all runs.
    Hooks and phase customization happen on the returned Pipeline.
    """
    from definable.agent.pipeline.pipeline import Pipeline
    from definable.agent.pipeline.phases.compose import ComposePhase
    from definable.agent.pipeline.phases.guard import GuardInputPhase, GuardOutputPhase
    from definable.agent.pipeline.phases.invoke import InvokeLoopPhase
    from definable.agent.pipeline.phases.prepare import PreparePhase
    from definable.agent.pipeline.phases.recall import RecallPhase
    from definable.agent.pipeline.phases.store import StorePhase
    from definable.agent.pipeline.phases.think import ThinkPhase

    pipeline = Pipeline(
      phases=[
        PreparePhase(self),
        RecallPhase(self),
        ThinkPhase(self),
        GuardInputPhase(self),
        ComposePhase(self),
        InvokeLoopPhase(self),  # streaming/cancellation set per-run
        GuardOutputPhase(self),
        StorePhase(self),
      ],
      debug=self._debug_config,
    )

    # Wire trace writer to event stream
    if self._trace_writer:
      import contextlib as _cl

      def _trace_handler(event: BaseRunOutputEvent) -> None:
        with _cl.suppress(Exception):
          self._trace_writer.write(event)  # type: ignore[union-attr]

      pipeline.event_stream.subscribe(_trace_handler)

    # Wire event bus to event stream
    async def _bus_handler(event: object) -> None:
      await self._event_bus.emit(event)

    pipeline.event_stream.subscribe(_bus_handler)

    return pipeline

  @property
  def pipeline(self) -> "Pipeline":
    """Access the agent's execution pipeline for customization.

    Example::

        agent.pipeline.add_phase(MyPhase(), after="recall")
        agent.pipeline.remove_phase("think")
    """
    return self._pipeline

  def hook(self, spec: str, callback: Optional[Callable] = None, *, priority: int = 0) -> Callable:
    """Register a hook on the pipeline.

    Delegates to ``self._pipeline.hook()``. See Pipeline.hook() for details.

    Example::

        @agent.hook("before:invoke_loop")
        async def my_hook(state):
            print(f"Messages: {len(state.invoke_messages)}")
            return state
    """
    return self._pipeline.hook(spec, callback, priority=priority)

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
  def name(self) -> str:
    """Alias for agent_name."""
    return self.agent_name

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

  @property
  def observability(self) -> Optional["ObservabilityConfig"]:
    """Observability config, if enabled."""
    return self._observability_config

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
    from definable.agent.lifecycle import start

    start(self)

  def _shutdown(self) -> None:
    from definable.agent.lifecycle import shutdown

    shutdown(self)

  async def _ashutdown(self) -> None:
    from definable.agent.lifecycle import ashutdown

    await ashutdown(self)

  async def _ensure_toolkits_initialized(self) -> None:
    from definable.agent.lifecycle import ensure_toolkits_initialized

    await ensure_toolkits_initialized(self)

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

  def output_validator(self, fn: Optional[Callable] = None) -> Callable:
    """Register a validator that runs after the agent produces output.

    The validator receives the output content and the RunContext. It can:
    - Return the output unchanged (pass validation)
    - Return a modified output (transform)
    - Raise ``RetryAgentRun`` to have the model retry with feedback

    Supports both ``@agent.output_validator`` and ``@agent.output_validator()``.

    Example::

      @agent.output_validator
      async def check_output(output, context):
          if not output or len(str(output)) < 10:
              from definable import RetryAgentRun
              raise RetryAgentRun("Output too short, provide more detail.")
          return output

    Validators run in registration order after the pipeline completes.
    Unlike ``after_response`` hooks, validators CAN raise exceptions to
    signal retry or stop.
    """
    if fn is not None:
      self._output_validators.append(fn)
      return fn

    def decorator(func: Callable) -> Callable:
      self._output_validators.append(func)
      return func

    return decorator

  async def _run_output_validators(self, output: Any, context: RunContext) -> Any:
    """Run all output validators. Returns final (possibly transformed) output."""
    import inspect

    for validator in self._output_validators:
      sig = inspect.signature(validator)
      params = list(sig.parameters.keys())
      # Support both (output,) and (output, context) signatures
      if len(params) >= 2:
        result = validator(output, context)
      else:
        result = validator(output)
      if inspect.isawaitable(result):
        result = await result
      if result is not None:
        output = result
    return output

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
  def auth(self) -> Optional["AuthProvider"]:
    """Get the auth provider."""
    return self._auth

  @auth.setter
  def auth(self, provider: Optional["AuthProvider"]) -> None:
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
      # when making multiple sequential sync calls with async HTTP clients.
      # Clear pending memory tasks from previous loops to avoid
      # "Event loop is closed" errors when waiting on orphaned tasks.
      self._pending_memory_tasks.clear()
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

    Delegates to the pipeline (PreparePhase → RecallPhase → ThinkPhase →
    GuardInputPhase → ComposePhase → InvokeLoopPhase → GuardOutputPhase →
    StorePhase). Middleware wraps the pipeline from outside.

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
        cancellation_token: Optional token for cooperative cancellation.

    Returns:
        RunOutput with response, metrics, tool executions, and messages.
    """
    # Validate output_schema early (#95/#98)
    if output_schema is not None:
      if not isinstance(output_schema, type) or not issubclass(output_schema, BaseModel):
        raise TypeError(
          f"output_schema must be a Pydantic BaseModel subclass, got {output_schema!r}. Example: output_schema=MyModel where MyModel(BaseModel)."
        )

    # Build initial LoopState from arguments
    state = self._build_initial_state(
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
      cancellation_token=cancellation_token,
    )

    # Transcribe audio in new messages (before pipeline — enriches text for all models)
    await self._transcribe_audio(state.new_messages)

    assert state.context is not None
    context = state.context

    # Set ambient RunContext so tools can access it via get_current_run_context()
    from definable.run.base import set_current_run_context

    set_current_run_context(context)
    try:
      # Fire agent-level before_request hooks (outside pipeline — receives RunContext)
      await self._fire_before_hooks(context)

      # Build middleware chain where core handler is the pipeline
      async def core_handler(ctx: RunContext) -> RunOutput:
        return await self._execute_via_pipeline(state)

      # Wrap with middleware (innermost to outermost)
      handler = core_handler
      for middleware in reversed(self._middleware):
        prev_handler = handler

        async def wrapped_handler(ctx: RunContext, mw=middleware, h=prev_handler) -> RunOutput:
          return await mw(ctx, h)

        handler = wrapped_handler

      # Execute pipeline through middleware chain
      result = await handler(context)

      # Run output validators (can raise RetryAgentRun or transform output)
      if self._output_validators and result.content is not None:
        result.content = await self._run_output_validators(result.content, context)

      # Record usage metrics
      if self._usage_tracker is not None and result.metrics is not None:
        await self._usage_tracker.arecord_run(result.metrics, self.model.id if self.model else None)

      # Fire agent-level after_response hooks (outside pipeline — receives RunOutput)
      await self._fire_after_hooks(result)

      return result
    finally:
      set_current_run_context(None)

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

    Delegates to the pipeline in streaming mode. Each phase yields
    (state, event) tuples; events are forwarded to the caller.

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
    # Build initial LoopState with streaming=True
    state = self._build_initial_state(
      instruction,
      messages=messages,
      session_id=session_id,
      run_id=run_id,
      user_id=user_id,
      images=images,
      output_schema=output_schema,
      cancellation_token=cancellation_token,
      streaming=True,
    )

    # Transcribe audio in new messages (before pipeline — enriches text for all models)
    await self._transcribe_audio(state.new_messages)

    # Set ambient RunContext for tools
    from definable.run.base import set_current_run_context

    assert state.context is not None
    set_current_run_context(state.context)
    try:
      from definable.agent.harness import execute_run

      async for updated_state, event in execute_run(
        self,
        state,
        cancellation_token=state.cancellation_token,
      ):
        state = updated_state
        if event is not None:
          self._emit(event)
          await self._event_bus.emit(event)
          yield event  # type: ignore[misc]

    except AgentCancelled:
      from definable.agent.events import RunCancelledEvent

      cancelled_event = RunCancelledEvent(
        run_id=state.run_id,
        session_id=state.session_id,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
        reason="Cancelled via CancellationToken",
      )
      self._emit(cancelled_event)
      yield cancelled_event

    except Exception as e:
      error_event = RunErrorEvent(
        run_id=state.run_id,
        session_id=state.session_id,
        agent_id=self.agent_id,
        error_type=type(e).__name__,
        content=str(e),
      )
      self._emit(error_event)
      yield error_event
      raise
    finally:
      set_current_run_context(None)

  # --- Knowledge & Memory Helpers ---

  async def _knowledge_retrieve(self, context: RunContext) -> List[RunOutputEvent]:
    from definable.agent.layers import knowledge_retrieve

    return await knowledge_retrieve(self, context)

  def _init_deep_research(self, config: "DeepResearchConfig") -> Optional["DeepResearch"]:
    from definable.agent.layers import init_deep_research

    return init_deep_research(self, config)

  def _discover_search_provider(self) -> object:
    from definable.agent.layers import discover_search_provider

    return discover_search_provider(self)

  async def _deep_research(self, context: RunContext) -> List[RunOutputEvent]:
    from definable.agent.layers import deep_research

    return await deep_research(self, context)

  async def _drain_memory_tasks(self) -> None:
    from definable.agent.lifecycle import drain_memory_tasks

    await drain_memory_tasks(self)

  async def _memory_recall(self, context: RunContext, new_messages: List[Message]) -> List[RunOutputEvent]:
    from definable.agent.layers import memory_recall

    return await memory_recall(self, context, new_messages)

  async def _memory_recall_semantic(self, session_id: str, user_id: str, query: str) -> str:
    from definable.agent.layers import memory_recall_semantic

    return await memory_recall_semantic(self, session_id, user_id, query)

  def _memory_store(self, new_messages: List[Message], context: RunContext) -> List[RunOutputEvent]:
    from definable.agent.layers import memory_store

    return memory_store(self, new_messages, context)

  # --- Readers Helpers ---

  @staticmethod
  def _init_readers(readers: "List[BaseReader] | BaseReader | bool | None") -> Optional["BaseReader"]:
    from definable.agent.resolution import init_readers

    return init_readers(readers)

  async def _readers_extract(self, context: RunContext, new_messages: List[Message]) -> List[RunOutputEvent]:
    from definable.agent.layers import readers_extract

    return await readers_extract(self, context, new_messages)

  # --- Guardrail Helpers ---

  def _extract_input_text(self, new_messages: List[Message]) -> str:
    from definable.agent.layers import extract_input_text

    return extract_input_text(new_messages)

  async def _run_input_guardrails(self, context: RunContext, new_messages: List[Message]) -> Optional[RunOutput]:
    from definable.agent.layers import run_input_guardrails

    return await run_input_guardrails(self, context, new_messages)

  async def _run_output_guardrails(self, context: RunContext, result: RunOutput) -> Optional[RunOutput]:
    from definable.agent.layers import run_output_guardrails

    return await run_output_guardrails(self, context, result)

  async def _run_tool_guardrails(self, context: RunContext, tool_execution: ToolExecution) -> Optional[str]:
    from definable.agent.layers import run_tool_guardrails

    return await run_tool_guardrails(self, context, tool_execution)

  # --- Thinking Layer ---

  # _THINKING_PROMPTS moved to definable.agent.layers

  def _extract_last_user_query(self, messages: List[Message]) -> Optional[str]:
    from definable.agent.layers import extract_last_user_query

    return extract_last_user_query(messages)

  @staticmethod
  def _build_routing_prompt(layer_name: str, query: str, context_str: str) -> str:
    from definable.agent.layers import build_routing_prompt

    return build_routing_prompt(layer_name, query, context_str)

  async def _should_invoke_layer(
    self,
    layer_name: str,
    query: str,
    decision_prompt: Optional[str] = None,
    routing_model: Optional["Model"] = None,
    messages: Optional[List[Message]] = None,
  ) -> bool:
    from definable.agent.layers import should_invoke_layer

    return await should_invoke_layer(self, layer_name, query, decision_prompt, routing_model, messages)

  def _build_layer_guide(self, context: Optional[RunContext] = None) -> str:
    from definable.agent.prompt import build_layer_guide

    return build_layer_guide(self, context)

  async def _evaluate_layer_trigger(
    self,
    trigger: Literal["always", "auto", "never"],
    callback: Callable[[], Awaitable[List[RunOutputEvent]]],
    *,
    layer_name: str = "",
    query_messages: Optional[List[Message]] = None,
    all_messages: Optional[List[Message]] = None,
    decision_prompt: Optional[str] = None,
    routing_model: Optional["Model"] = None,
  ) -> List[RunOutputEvent]:
    from definable.agent.layers import evaluate_layer_trigger

    return await evaluate_layer_trigger(
      self,
      trigger,
      callback,
      layer_name=layer_name,
      query_messages=query_messages,
      all_messages=all_messages,
      decision_prompt=decision_prompt,
      routing_model=routing_model,
    )

  def _should_store_memory(self) -> bool:
    from definable.agent.layers import should_store_memory

    return should_store_memory(self)

  async def _thinking_should_run(self, messages: List[Message]) -> bool:
    from definable.agent.layers import thinking_should_run

    return await thinking_should_run(self, messages)

  async def _run_pre_execution_pipeline(
    self,
    context: RunContext,
    new_messages: List[Message],
    all_messages: List[Message],
  ) -> List[RunOutputEvent]:
    from definable.agent.layers import run_pre_execution_pipeline

    return await run_pre_execution_pipeline(self, context, new_messages, all_messages)

  def _build_thinking_messages(
    self,
    context: RunContext,
    invoke_messages: List[Message],
    tools: Dict[str, Function],
  ) -> "tuple[list[Message], bool]":
    from definable.agent.layers import build_thinking_messages

    return build_thinking_messages(self, context, invoke_messages, tools)

  async def _execute_thinking(
    self,
    context: RunContext,
    invoke_messages: List[Message],
    tools: Dict[str, Function],
  ) -> "AsyncGenerator[Union[str, tuple[Optional[ThinkingOutput], Optional[str], list[ReasoningStep], list[Message]]], None]":
    from definable.agent.layers import execute_thinking

    async for item in execute_thinking(self, context, invoke_messages, tools):
      yield item

  def _enable_native_thinking(self) -> None:
    from definable.agent.layers import enable_native_thinking

    enable_native_thinking(self)

  @staticmethod
  def _format_reasoning_context(steps: "list[ReasoningStep]") -> str:
    from definable.agent.prompt import format_reasoning_context

    return format_reasoning_context(steps)

  # --- Internal Methods ---

  async def _build_invoke_messages(
    self,
    context: RunContext,
    messages: List[Message],
    tools: Dict[str, Function],
    *,
    thinking_output: "Optional[ThinkingOutput]" = None,
    thinking_text: Optional[str] = None,
    reasoning_steps: "Optional[list[ReasoningStep]]" = None,
    reasoning_messages: "Optional[list[Message]]" = None,
  ) -> tuple:
    from definable.agent.prompt import build_invoke_messages

    return await build_invoke_messages(
      self,
      context,
      messages,
      tools,
      thinking_output=thinking_output,
      thinking_text=thinking_text,
      reasoning_steps=reasoning_steps,
      reasoning_messages=reasoning_messages,
    )

  async def _execute_via_pipeline(self, state: "LoopState") -> RunOutput:
    """Execute the full run via harness and return RunOutput."""
    from definable.agent.harness import execute_run

    try:
      async for updated_state, event in execute_run(
        self,
        state,
        cancellation_token=state.cancellation_token,
      ):
        state = updated_state
        if event is not None:
          self._emit(event)
          await self._event_bus.emit(event)
      return self._state_to_run_output(state)

    except AgentCancelled:
      from definable.agent.events import RunCancelledEvent

      cancelled_event = RunCancelledEvent(
        run_id=state.run_id,
        session_id=state.session_id,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
        reason="Cancelled via CancellationToken",
      )
      self._emit(cancelled_event)
      await self._event_bus.emit(cancelled_event)
      return RunOutput(
        run_id=state.run_id,
        session_id=state.session_id,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
        status=RunStatus.cancelled,
        model=self.model.id,
        model_provider=self.model.provider,
      )

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

      # Detect if native thinking is active
      _native_thinking = bool(self._thinking and self._thinking.enabled and self._thinking.should_use_native(self.model))

      # Create the loop
      loop = AgentLoop(
        model=self.model,
        tools=tools,
        messages=invoke_messages,
        context=context,
        config=self.config,
        streaming=False,
        native_thinking=_native_thinking,
        cancellation_token=cancellation_token,
        compression_manager=self._compression_manager,
        guardrails=self.guardrails,
        emit_fn=self._emit,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
      )

      # Run the loop, collect events
      final_content: Optional[str] = None
      final_parsed: Any = None
      final_metrics: Optional[Metrics] = None

      async for event in loop.run():
        await self._event_bus.emit(event)
        self._emit(event)

        if isinstance(event, RunCompletedEvent):
          final_content = event.content
          final_parsed = event.parsed
          final_metrics = event.metrics

      # Build output messages (excluding system message)
      output_messages = [m for m in loop.messages if m.role != "system"]

      # Determine reasoning content: native thinking takes priority over Definable's layer
      final_reasoning_content = loop.native_reasoning_content or (self._format_reasoning_context(reasoning_steps) if reasoning_steps else None)

      return RunOutput(
        run_id=context.run_id,
        session_id=context.session_id,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
        input=run_input,
        content=final_content,
        parsed=final_parsed,
        tools=loop.tool_executions or None,
        metrics=final_metrics,
        messages=output_messages,
        model=self.model.id,
        model_provider=self.model.provider,
        status=RunStatus.completed,
        session_state=context.session_state,
        reasoning_steps=reasoning_steps or None,
        reasoning_messages=reasoning_agent_messages or None,
        reasoning_content=final_reasoning_content,
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
        input=run_input,
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
    from definable.agent.resolution import init_skills

    init_skills(self.skills)

  def _build_skill_instructions(self) -> str:
    from definable.agent.prompt import build_skill_instructions

    return build_skill_instructions(self)

  def _flatten_tools(self) -> Dict[str, Function]:
    from definable.agent.resolution import flatten_tools

    return flatten_tools(self.skills, self.toolkits, self.tools)

  def _init_tracing(self) -> Optional[TraceWriter]:
    from definable.agent.resolution import init_tracing

    return init_tracing(self._tracing_config)

  # --- Layer Resolvers (called once during __init__) ---

  def _resolve_memory(self, memory: "Memory | bool | None") -> Optional["Memory"]:
    from definable.agent.resolution import resolve_memory

    return resolve_memory(memory)

  def _resolve_memory_embedder(self) -> None:
    from definable.agent.resolution import resolve_memory_embedder

    resolve_memory_embedder(self.memory, self.model)

  def _create_embedder_for_model(self) -> "Any":
    from definable.agent.resolution import create_embedder_for_model

    return create_embedder_for_model(self.model)

  def _resolve_knowledge(self, knowledge: "Knowledge | str | bool | None") -> Optional["Knowledge"]:
    from definable.agent.resolution import resolve_knowledge

    return resolve_knowledge(knowledge)

  @staticmethod
  def _resolve_tracing(tracing_param: "Tracing | bool | None", config: Optional[AgentConfig]) -> Optional["Tracing"]:
    from definable.agent.resolution import resolve_tracing

    return resolve_tracing(tracing_param, config)

  def _resolve_context(self, context: Union[bool, "Context", None]) -> Optional["ContextManager"]:
    from definable.agent.resolution import resolve_context

    return resolve_context(context, self.model)

  def _resolve_deferred_tools(self) -> Optional["DeferredToolManager"]:
    from definable.agent.resolution import resolve_deferred_tools

    return resolve_deferred_tools(self._context_manager, self._tools_dict)

  def _resolve_compression(self, compression: Union[bool, "Compression", None]) -> Optional["CompressionManager"]:
    from definable.agent.resolution import resolve_compression

    return resolve_compression(compression, self.model)

  def _build_compression_manager(self, compression: "Compression") -> "CompressionManager":
    from definable.agent.resolution import build_compression_manager

    return build_compression_manager(compression, self.model)

  def _build_initial_state(
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
    output_schema: Optional[type] = None,
    cancellation_token: Optional[CancellationToken] = None,
    streaming: bool = False,
  ) -> "LoopState":
    """Build initial LoopState from arun() arguments."""
    from definable.agent.pipeline.state import LoopState, LoopStatus

    _run_id = run_id or str(uuid4())
    _session_id = session_id or self.session_id

    new_messages = self._normalize_instruction(instruction, images, videos, audio, files)
    all_messages = (messages or []) + new_messages

    context = RunContext(
      run_id=_run_id,
      session_id=_session_id,
      user_id=user_id,
      dependencies=self.config.dependencies,
      session_state=dict(self.config.session_state or {}),
      output_schema=output_schema,
      metadata={"_messages": all_messages},
    )

    run_input = RunInput(
      input_content=instruction,
      images=images,
      videos=videos,
      audios=audio,
      files=files,
    )

    return LoopState(
      run_id=_run_id,
      session_id=_session_id,
      user_id=user_id,
      agent_id=self.agent_id,
      agent_name=self.agent_name,
      raw_instruction=instruction,
      new_messages=new_messages,
      all_messages=all_messages,
      context=context,
      config=self.config,
      model=self.model,
      status=LoopStatus.pending,
      run_input=run_input,
      streaming=streaming,
      cancellation_token=cancellation_token,
    )

  def _state_to_run_output(self, state: "LoopState") -> RunOutput:
    """Convert final LoopState to RunOutput."""
    from definable.agent.pipeline.state import LoopStatus

    status_map = {
      LoopStatus.completed: RunStatus.completed,
      LoopStatus.paused: RunStatus.paused,
      LoopStatus.cancelled: RunStatus.cancelled,
      LoopStatus.blocked: RunStatus.blocked,
      LoopStatus.error: RunStatus.error,
    }

    return RunOutput(
      run_id=state.run_id,
      session_id=state.session_id,
      agent_id=state.agent_id,
      agent_name=state.agent_name,
      input=state.run_input,
      content=state.content,
      parsed=state.parsed,
      tools=state.tool_executions or None,
      metrics=state.metrics,
      messages=state.output_messages,
      model=self.model.id,
      model_provider=self.model.provider,
      status=status_map.get(state.status, RunStatus.completed),
      session_state=state.context.session_state if state.context else None,
      reasoning_steps=state.reasoning_steps or None,
      reasoning_messages=state.reasoning_messages or None,
      reasoning_content=state.native_reasoning_content
      or state.thinking_text
      or (self._format_reasoning_context(state.reasoning_steps) if state.reasoning_steps else None),
      phase_metrics=state.phase_metrics or None,
    )

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

    # Inject spawn_agent tool when sub-agent policy is configured
    if self._sub_agent_policy:
      from definable.agent.pipeline.sub_agent import _build_spawn_agent_function

      spawn_fn = _build_spawn_agent_function(self, self._sub_agent_policy)
      tools["spawn_agent"] = spawn_fn

    if self.memory and hasattr(self.memory, "get_tools"):
      user_id = context.user_id or "default"
      session_id = context.session_id or "default"
      for fn in self.memory.get_tools(user_id, session_id):
        tools[fn.name] = fn

    # Inject ask_user tool when question resolver is configured
    if self._question_resolver is not None:
      from definable.agent.hitl.question import build_ask_user_tool

      tools["ask_user"] = build_ask_user_tool(self._question_resolver)

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

  async def _transcribe_audio(self, messages: List[Message]) -> None:
    from definable.agent.lifecycle import transcribe_audio

    await transcribe_audio(self, messages)

  def _emit(self, event: BaseRunOutputEvent) -> None:
    """Emit event to trace writer (fire-and-forget)."""
    if self._trace_writer:
      with contextlib.suppress(Exception):
        # Tracing should never break the main flow
        self._trace_writer.write(event)

  # --- Triggers ---

  @property
  def triggers(self) -> List["BaseTrigger"]:
    """Registered triggers (read-only copy)."""
    return list(self._triggers)

  def on(self, trigger: "BaseTrigger") -> Callable:
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

  def emit(self, event_name: str, data: Optional[dict] = None) -> None:
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

  # --- Security ---

  @property
  def security(self) -> Optional[Any]:
    """Return the SecurityConfig, or None if not configured."""
    return self._security

  async def security_audit(self) -> Any:
    """Run a security audit on this agent's configuration.

    Returns a SecurityReport with findings and a score (0–100).
    """
    from definable.agent.security.audit import security_audit

    return await security_audit(self)

  # --- Usage Tracking ---

  @property
  def usage_tracker(self) -> Optional[Any]:
    """Return the UsageTracker, or None if not configured."""
    return self._usage_tracker

  # --- Scheduler ---

  @property
  def scheduler(self) -> Optional[Any]:
    """Return a Scheduler for this agent's triggers, or None if no schedulable triggers exist."""
    from definable.agent.trigger.interval import Interval
    from definable.agent.trigger.oneshot import OneShot

    try:
      from definable.agent.trigger.cron import Cron

      schedulable_types = (Cron, Interval, OneShot)
    except ImportError:
      schedulable_types = (Interval, OneShot)  # type: ignore[assignment]

    schedulable = [t for t in self._triggers if isinstance(t, schedulable_types)]
    if not schedulable:
      return None

    from definable.agent.scheduler.scheduler import Scheduler

    sched = Scheduler()
    for trigger in schedulable:
      sched.add(trigger)
    return sched

  # --- Interfaces ---

  @property
  def gateway(self) -> Optional["InterfaceGateway"]:
    """Return the InterfaceGateway, or None if not created."""
    return self._gateway

  @property
  def interfaces(self) -> List["BaseInterface"]:
    """Read-only list of attached interfaces."""
    return list(self._interfaces)

  async def aserve(
    self,
    *,
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
      name: Optional prefix for log messages (defaults to agent_name).
      host: Host to bind the HTTP server to.
      port: Port for the HTTP server.
      enable_server: Force-enable/disable the HTTP server.  When *None*
        (default), the server starts if any Webhook triggers exist.
      dev: Enable development mode with Swagger docs and info-level logging.
    """
    from definable.runtime.runner import AgentRuntime

    resolved_gateway = self._gateway

    all_interfaces = list(self._interfaces)

    # Auto-create gateway for 2+ interfaces (production-grade supervision)
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
    *,
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
      name: Optional prefix for log messages (defaults to agent_name).
      host: Host to bind the HTTP server to.
      port: Port for the HTTP server.
      enable_server: Force-enable/disable the HTTP server.  When *None*
        (default), the server starts if any Webhook triggers exist.
      dev: Enable development mode with hot reload and Swagger docs.
    """
    if dev:
      from definable.runtime._dev import is_dev_child, run_dev_mode

      if not is_dev_child():
        run_dev_mode()
        return

    asyncio.run(
      self.aserve(
        name=name,
        host=host,
        port=port,
        enable_server=enable_server,
        dev=dev,
      )
    )

  # --- Manifest ---

  def export_manifest(self) -> Dict[str, Any]:
    """Export agent configuration as a manifest for platform deployment.

    Returns a dictionary with agent name, model, tools, and instruction
    summary — used by Definable Cloud's platform dashboard and packager.

    Returns:
      Dict with agent metadata.
    """
    return {
      "agent_name": self.agent_name,
      "agent_id": self.agent_id,
      "model": self.model.id if self.model else None,
      "tools": [{"name": t.name, "description": t.description} for t in self.tools],
      "instructions_summary": (self.instructions[:200] + "...") if self.instructions and len(self.instructions) > 200 else self.instructions,
      "has_memory": self.memory is not None,
      "has_knowledge": self._knowledge is not None,
    }

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
