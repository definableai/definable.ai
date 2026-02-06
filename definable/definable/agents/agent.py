"""Agent class - production-grade wrapper around model execution."""

import asyncio
import contextlib
from typing import (
  TYPE_CHECKING,
  Any,
  AsyncIterator,
  Dict,
  Iterator,
  List,
  Optional,
  Type,
  Union,
)
from uuid import uuid4

from definable.agents.config import AgentConfig
from definable.agents.middleware import Middleware
from definable.agents.toolkit import Toolkit
from definable.agents.tracing.base import TraceWriter
from definable.media import Audio, File, Image, Video
from definable.models.message import Message
from definable.models.metrics import Metrics
from definable.models.response import ToolExecution
from definable.run.agent import (
  RunCompletedEvent,
  RunContentEvent,
  RunErrorEvent,
  RunInput,
  RunOutput,
  RunOutputEvent,
  RunStartedEvent,
  ToolCallCompletedEvent,
  ToolCallStartedEvent,
)
from definable.run.base import BaseRunOutputEvent, RunContext, RunStatus
from definable.tools.function import Function
from definable.utils.tools import get_function_call_for_tool_call
from pydantic import BaseModel

if TYPE_CHECKING:
  from definable.compression import CompressionManager
  from definable.models.base import Model


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
      from definable.agents import Agent, AgentConfig
      from definable.models import OpenAIChat

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
    # Primary attributes - directly accessible
    model: "Model",
    tools: Optional[List[Function]] = None,
    toolkits: Optional[List[Toolkit]] = None,
    instructions: Optional[str] = None,
    # Optional advanced configuration
    config: Optional[AgentConfig] = None,
  ):
    """
    Initialize the agent.

    Args:
        model: Model instance to use for generation (required).
        tools: List of tools (Function objects) available to the agent.
        toolkits: List of toolkits providing additional tools.
        instructions: System instructions for the agent.
        config: Optional advanced configuration settings.
    """
    # Direct attributes
    self.model = model
    self.tools = tools or []
    self.toolkits = toolkits or []
    self.instructions = instructions

    # Optional config for advanced settings
    self.config = config or AgentConfig()

    # Internal state
    self._tools_dict: Dict[str, Function] = self._flatten_tools()
    self._trace_writer: Optional[TraceWriter] = self._init_tracing()
    self._compression_manager: Optional["CompressionManager"] = self._init_compression()
    self._middleware: List[Middleware] = []
    self._started = False

    # Auto-register knowledge middleware if configured
    if self.config.knowledge and self.config.knowledge.enabled:
      from definable.agents.middleware import KnowledgeMiddleware

      self._middleware.insert(0, KnowledgeMiddleware(self.config.knowledge))

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
    if self._trace_writer:
      self._trace_writer.shutdown()
    self._started = False

  async def _ashutdown(self) -> None:
    """Async cleanup."""
    self._shutdown()

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

  # --- Run Methods ---

  def run(
    self,
    instruction: Union[str, Message, List[Message]],
    *,
    messages: Optional[List[Message]] = None,
    session_id: Optional[str] = None,
    run_id: Optional[str] = None,
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
    images: Optional[List[Image]] = None,
    videos: Optional[List[Video]] = None,
    audio: Optional[List[Audio]] = None,
    files: Optional[List[File]] = None,
    output_schema: Optional[Type[BaseModel]] = None,
  ) -> RunOutput:
    """
    Async run with middleware chain execution.

    Args:
        instruction: New user message (string, Message, or list).
        messages: Optional conversation history for multi-turn.
        session_id: Session identifier (auto-generated if not provided).
        run_id: Run identifier (auto-generated if not provided).
        images: Images to include with the instruction.
        videos: Videos to include with the instruction.
        audio: Audio to include with the instruction.
        files: Files to include with the instruction.
        output_schema: Optional Pydantic model for structured output.

    Returns:
        RunOutput with response, metrics, tool executions, and messages.
    """
    run_id = run_id or str(uuid4())
    session_id = session_id or str(uuid4())

    # Normalize instruction to messages
    new_messages = self._normalize_instruction(instruction, images, videos, audio, files)
    all_messages = (messages or []) + new_messages

    # Build context with messages in metadata for middleware access
    context = RunContext(
      run_id=run_id,
      session_id=session_id,
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

    # Build the execution chain with middleware
    async def core_handler(ctx: RunContext) -> RunOutput:
      return await self._execute_run(ctx, all_messages, run_input)

    # Wrap with middleware (innermost to outermost)
    handler = core_handler
    for middleware in reversed(self._middleware):
      prev_handler = handler

      async def wrapped_handler(ctx: RunContext, mw=middleware, h=prev_handler) -> RunOutput:
        return await mw(ctx, h)

      handler = wrapped_handler

    # Execute
    return await handler(context)

  def run_stream(
    self,
    instruction: Union[str, Message, List[Message]],
    *,
    messages: Optional[List[Message]] = None,
    session_id: Optional[str] = None,
    run_id: Optional[str] = None,
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
      timeout_seconds = 300.0
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
    images: Optional[List[Image]] = None,
    output_schema: Optional[Type[BaseModel]] = None,
  ) -> AsyncIterator[RunOutputEvent]:
    """
    Async streaming run that yields events with full agent loop support.

    Args:
        instruction: New user message.
        messages: Optional conversation history.
        session_id: Session identifier.
        run_id: Run identifier.
        images: Images to include.
        output_schema: Optional structured output schema.

    Yields:
        RunOutputEvent instances as the run progresses.
    """
    run_id = run_id or str(uuid4())
    session_id = session_id or str(uuid4())

    # Normalize instruction to messages
    new_messages = self._normalize_instruction(instruction, images)
    all_messages = (messages or []) + new_messages

    # Build context with messages in metadata for middleware access
    context = RunContext(
      run_id=run_id,
      session_id=session_id,
      dependencies=self.config.dependencies,
      session_state=dict(self.config.session_state or {}),
      output_schema=output_schema,
      metadata={"_messages": all_messages},
    )

    # Run knowledge middleware if configured (streaming doesn't use middleware chain)
    if self.config.knowledge and self.config.knowledge.enabled:
      from definable.agents.middleware import KnowledgeMiddleware

      km = KnowledgeMiddleware(self.config.knowledge)

      # Create a dummy next_handler since we just need the retrieval side effect
      async def _dummy_handler(ctx: "RunContext") -> "RunOutput":
        return RunOutput(run_id=ctx.run_id, session_id=ctx.session_id, status=RunStatus.completed)

      await km(context, _dummy_handler)

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
      model_provider=self.model.provider,
      run_input=run_input,
    )
    self._emit(started_event)
    yield started_event

    try:
      # Build messages list with system message prepended if instructions exist
      invoke_messages = all_messages.copy()

      # Build system message with optional knowledge context
      system_content = self.instructions or ""
      if context.knowledge_context:
        position = (context.metadata or {}).get("_knowledge_position", "system")
        if position == "system":
          if system_content:
            system_content = f"{system_content}\n\n{context.knowledge_context}"
          else:
            system_content = context.knowledge_context

      if system_content:
        invoke_messages.insert(0, Message(role="system", content=system_content))

      # Convert tools to dicts for the model API (OpenAI format)
      tools_dicts = [{"type": "function", "function": t.to_dict()} for t in tools.values()] if tools else None

      all_tool_executions: List[ToolExecution] = []
      final_content = ""
      total_metrics: Optional[Metrics] = None

      # STREAMING AGENT LOOP - continues until model gives final answer
      while True:
        # Compress tool results if needed (before model call to stay within context limits)
        if self._compression_manager is not None:
          if await self._compression_manager.ashould_compress(invoke_messages, tools_dicts, model=self.model):
            await self._compression_manager.acompress(invoke_messages)

        # Accumulate response while streaming
        accumulated_content = ""
        accumulated_tool_calls: List[Dict[str, Any]] = []
        accumulated_metrics: Optional[Metrics] = None

        # Create assistant message (required by model.ainvoke_stream)
        assistant_message = Message(role="assistant")

        # Stream model response
        async for chunk in self.model.ainvoke_stream(
          messages=invoke_messages,
          assistant_message=assistant_message,
          tools=tools_dicts,
          response_format=context.output_schema,
        ):
          # Stream text content tokens immediately
          if hasattr(chunk, "content") and chunk.content:
            accumulated_content += chunk.content
            content_event = RunContentEvent(
              run_id=context.run_id,
              session_id=context.session_id,
              agent_id=self.agent_id,
              agent_name=self.agent_name,
              content=chunk.content,
            )
            # Not traced - RunCompleted contains full content
            yield content_event

          # Accumulate tool calls from stream (they come in deltas)
          if hasattr(chunk, "tool_calls") and chunk.tool_calls:
            accumulated_tool_calls = self._merge_tool_call_deltas(accumulated_tool_calls, chunk.tool_calls)

          # Accumulate metrics from chunks (usually in final chunk with usage info)
          if hasattr(chunk, "response_usage") and chunk.response_usage is not None:
            if accumulated_metrics is None:
              accumulated_metrics = chunk.response_usage
            else:
              accumulated_metrics = accumulated_metrics + chunk.response_usage

        # After streaming complete, add assistant message to history
        assistant_msg = Message(
          role="assistant",
          content=accumulated_content or None,
          tool_calls=accumulated_tool_calls or None,
        )
        # Attach accumulated metrics to this message
        if accumulated_metrics is not None:
          assistant_msg.metrics = accumulated_metrics
          # Aggregate into total_metrics for RunCompletedEvent
          if total_metrics is None:
            total_metrics = accumulated_metrics
          else:
            total_metrics = total_metrics + accumulated_metrics
        invoke_messages.append(assistant_msg)

        # Check if no tool calls - model is done
        if not accumulated_tool_calls:
          final_content = accumulated_content
          break

        # Execute tools and yield events
        for tool_call in accumulated_tool_calls:
          # Parse tool call
          function_call = get_function_call_for_tool_call(tool_call, tools)

          # Create ToolExecution for tracking
          tool_execution = ToolExecution(
            tool_call_id=tool_call.get("id"),
            tool_name=tool_call.get("function", {}).get("name"),
            tool_args=function_call.arguments if function_call else None,
          )

          # Yield tool started event
          tool_started_event = ToolCallStartedEvent(
            run_id=context.run_id,
            session_id=context.session_id,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tool=tool_execution,
          )
          self._emit(tool_started_event)
          yield tool_started_event

          # Execute the tool
          if function_call:
            result = await function_call.aexecute()
            tool_execution.result = str(result.result) if result.status == "success" else str(result.error)
            tool_execution.tool_call_error = result.status == "failure"
          else:
            tool_name = tool_call.get("function", {}).get("name", "unknown")
            tool_execution.result = f"Tool '{tool_name}' not found"
            tool_execution.tool_call_error = True

          all_tool_executions.append(tool_execution)

          # Yield tool completed event
          tool_completed_event = ToolCallCompletedEvent(
            run_id=context.run_id,
            session_id=context.session_id,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            tool=tool_execution,
            content=tool_execution.result,
          )
          self._emit(tool_completed_event)
          yield tool_completed_event

          # Add tool result to messages
          invoke_messages.append(
            Message(
              role="tool",
              content=tool_execution.result,
              tool_call_id=tool_call.get("id"),
              name=tool_call.get("function", {}).get("name"),
            )
          )

        # Loop continues - next iteration will stream model's response to tool results

      # Final completed event
      completed_event = RunCompletedEvent(
        run_id=context.run_id,
        session_id=context.session_id,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
        content=final_content,
        metrics=total_metrics,
      )
      self._emit(completed_event)
      yield completed_event

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

  def _merge_tool_call_deltas(self, existing: List[Dict[str, Any]], new_deltas: List[Any]) -> List[Dict[str, Any]]:
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

      # Get delta values (handle both dict and object formats)
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

  # --- Internal Methods ---

  async def _execute_run(
    self,
    context: RunContext,
    messages: List[Message],
    run_input: Optional[RunInput] = None,
  ) -> RunOutput:
    """Core execution logic with agent loop for tool calls."""
    # Prepare tools with injected context
    tools = self._prepare_tools_for_run(context)

    # Emit RunStarted
    self._emit(
      RunStartedEvent(
        run_id=context.run_id,
        session_id=context.session_id,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
        model=self.model.id,
        model_provider=self.model.provider,
        run_input=run_input,
      )
    )

    try:
      # Build messages list with system message prepended if instructions exist
      invoke_messages = messages.copy()

      # Build system message with optional knowledge context
      system_content = self.instructions or ""
      if context.knowledge_context:
        position = (context.metadata or {}).get("_knowledge_position", "system")
        if position == "system":
          # Append knowledge context to system message
          if system_content:
            system_content = f"{system_content}\n\n{context.knowledge_context}"
          else:
            system_content = context.knowledge_context

      if system_content:
        invoke_messages.insert(0, Message(role="system", content=system_content))

      # Convert tools to dicts for the model API (OpenAI format)
      tools_dicts = [{"type": "function", "function": t.to_dict()} for t in tools.values()] if tools else None

      all_tool_executions: List[ToolExecution] = []
      final_response = None

      # AGENT LOOP - continues until model gives final answer (no tool calls)
      while True:
        # Compress tool results if needed (before model call to stay within context limits)
        if self._compression_manager is not None:
          if await self._compression_manager.ashould_compress(invoke_messages, tools_dicts, model=self.model):
            await self._compression_manager.acompress(invoke_messages)

        # Create assistant message (required by model.ainvoke)
        assistant_message = Message(role="assistant")

        # Call model
        response = await self.model.ainvoke(
          messages=invoke_messages,
          assistant_message=assistant_message,
          tools=tools_dicts,
          response_format=context.output_schema,
        )

        # Add assistant message to conversation history
        assistant_msg = Message(
          role="assistant",
          content=response.content,
          tool_calls=response.tool_calls or None,
        )
        # Attach metrics from model response to this message
        if response.response_usage is not None:
          assistant_msg.metrics = response.response_usage
        invoke_messages.append(assistant_msg)

        # Check if no tool calls - model is done, exit loop
        if not response.tool_calls:
          final_response = response
          break

        # Execute each tool call
        for tool_call in response.tool_calls:
          # Parse tool call
          function_call = get_function_call_for_tool_call(tool_call, tools)

          # Create ToolExecution for tracking
          tool_execution = ToolExecution(
            tool_call_id=tool_call.get("id"),
            tool_name=tool_call.get("function", {}).get("name"),
            tool_args=function_call.arguments if function_call else None,
          )

          # Emit ToolCallStarted event
          self._emit(
            ToolCallStartedEvent(
              run_id=context.run_id,
              session_id=context.session_id,
              agent_id=self.agent_id,
              agent_name=self.agent_name,
              tool=tool_execution,
            )
          )

          # Execute the tool
          if function_call:
            result = await function_call.aexecute()
            tool_execution.result = str(result.result) if result.status == "success" else str(result.error)
            tool_execution.tool_call_error = result.status == "failure"
          else:
            tool_name = tool_call.get("function", {}).get("name", "unknown")
            tool_execution.result = f"Tool '{tool_name}' not found"
            tool_execution.tool_call_error = True

          all_tool_executions.append(tool_execution)

          # Emit ToolCallCompleted event
          self._emit(
            ToolCallCompletedEvent(
              run_id=context.run_id,
              session_id=context.session_id,
              agent_id=self.agent_id,
              agent_name=self.agent_name,
              tool=tool_execution,
              content=tool_execution.result,
            )
          )

          # Add tool result message to conversation
          invoke_messages.append(
            Message(
              role="tool",
              content=tool_execution.result,
              tool_call_id=tool_call.get("id"),
              name=tool_call.get("function", {}).get("name"),
            )
          )

        # Loop continues - model will be called again with tool results

      # Build output messages (excluding system message)
      output_messages = [m for m in invoke_messages if m.role != "system"]

      # Build RunOutput with final response content
      output = RunOutput(
        run_id=context.run_id,
        session_id=context.session_id,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
        content=final_response.content if final_response else None,
        tools=all_tool_executions,
        metrics=final_response.response_usage if final_response else None,
        messages=output_messages,
        model=self.model.id,
        model_provider=self.model.provider,
        status=RunStatus.completed,
        session_state=context.session_state,
        reasoning_content=final_response.reasoning_content if final_response else None,
        citations=final_response.citations if final_response else None,
        images=final_response.images if final_response else None,
        videos=final_response.videos if final_response else None,
        audio=final_response.audios if final_response else None,
      )

      # Emit RunCompleted
      self._emit(
        RunCompletedEvent(
          run_id=context.run_id,
          session_id=context.session_id,
          agent_id=self.agent_id,
          agent_name=self.agent_name,
          content=output.content,
          metrics=output.metrics,
        )
      )

      return output

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

  def _flatten_tools(self) -> Dict[str, Function]:
    """
    Flatten tools from toolkits and direct tools into a single dict.

    Toolkit tools are processed first, then direct tools (which can override).
    """
    result: Dict[str, Function] = {}

    # Process toolkits first
    for toolkit in self.toolkits:
      for fn in toolkit.tools:
        # Merge toolkit dependencies with tool's existing dependencies
        if toolkit.dependencies:
          existing_deps = getattr(fn, "_dependencies", None) or {}
          fn._dependencies = {**existing_deps, **toolkit.dependencies}
        result[fn.name] = fn

    # Process direct tools (can override toolkit tools)
    for fn in self.tools:
      result[fn.name] = fn

    return result

  def _init_tracing(self) -> Optional[TraceWriter]:
    """Initialize trace writer if tracing is configured."""
    if self.config.tracing and self.config.tracing.exporters:
      return TraceWriter(self.config.tracing)
    return None

  def _init_compression(self) -> Optional["CompressionManager"]:
    """Initialize compression manager if compression is configured."""
    if self.config.compression and self.config.compression.enabled:
      from definable.compression import CompressionManager

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

  def __repr__(self) -> str:
    return f"Agent(model={self.model.id!r}, tools={len(self._tools_dict)}, name={self.agent_name!r})"
