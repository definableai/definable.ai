"""ClaudeCodeAgent — wraps Claude Code CLI with the full Definable ecosystem.

Communicates directly with the ``claude`` CLI via subprocess + JSONL protocol.
No external SDK dependency — the transport layer is built-in.

Example::

    from definable.claude_code import ClaudeCodeAgent

    agent = ClaudeCodeAgent(
        model="claude-sonnet-4-6",
        instructions="Senior backend developer.",
        allowed_tools=["Read", "Write", "Edit", "Bash"],
        cwd="/workspace/my-app",
    )
    result = await agent.arun("Fix the auth bug in the login endpoint")
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import (
  TYPE_CHECKING,
  Any,
  AsyncIterator,
  Dict,
  List,
  Optional,
  Type,
  Union,
)
from uuid import uuid4

from definable.claude_code.bridge import ToolBridge
from definable.claude_code.parser import message_to_event, parse_to_run_output
from definable.claude_code.tool_server import ToolServer
from definable.claude_code.transport import SubprocessTransport
from definable.claude_code.types import (
  ControlRequest,
  Message,
  ResultMessage,
  parse_message,
)
from definable.agent.events import (
  RunCompletedEvent,
  RunContext,
  RunErrorEvent,
  RunOutput,
  RunOutputEvent,
  RunStartedEvent,
  RunStatus,
)
from definable.tool.function import Function
from definable.utils.log import log_debug, log_error, log_warning

if TYPE_CHECKING:
  from definable.agent.config import AgentConfig
  from definable.agent.guardrail.base import Guardrails
  from definable.agent.tracing import Tracing
  from definable.knowledge.base import Knowledge
  from definable.memory.manager import Memory, MemoryManager
  from pydantic import BaseModel


@dataclass
class ClaudeCodeAgent:
  """Agent wrapping Claude Code CLI with Definable features.

  Communicates directly with the ``claude`` CLI via subprocess + JSONL.
  No external SDK dependency — the transport layer is built-in.

  Supports: memory, knowledge/RAG, custom tools (via MCP), guardrails,
  middleware, tracing, skills, structured output, extended thinking,
  and multi-turn sessions.
  """

  # --- Claude Code configuration ---
  model: str = "claude-sonnet-4-6"
  instructions: Optional[str] = None
  allowed_tools: Optional[List[str]] = None
  disallowed_tools: Optional[List[str]] = None
  permission_mode: str = "bypassPermissions"
  max_turns: Optional[int] = None
  max_budget_usd: Optional[float] = None
  cwd: Optional[str] = None
  cli_path: Optional[str] = None
  env: Optional[Dict[str, str]] = None

  # --- Agent identity ---
  agent_id: Optional[str] = None
  agent_name: Optional[str] = None

  # --- Definable features ---
  memory: Optional[Union["Memory", "MemoryManager", bool]] = None
  knowledge: Optional["Knowledge"] = None
  guardrails: Optional["Guardrails"] = None
  middleware: Optional[List[Any]] = None
  tools: Optional[List[Function]] = None
  toolkits: Optional[List[Any]] = None
  skills: Optional[List[Any]] = None
  tracing: Optional[Union["Tracing", bool]] = None

  # --- Extended thinking ---
  thinking_budget_tokens: Optional[int] = None

  # --- Session ---
  continue_conversation: bool = False

  # --- Config ---
  config: Optional["AgentConfig"] = None

  # --- Internal state ---
  _tool_bridge: Optional[ToolBridge] = field(default=None, repr=False)
  _tool_server: Optional[ToolServer] = field(default=None, repr=False)
  _memory_manager: Optional[Any] = field(default=None, repr=False)
  _knowledge_instance: Optional[Any] = field(default=None, repr=False)
  _knowledge_config: Optional[Any] = field(default=None, repr=False)
  _tracing_config: Optional[Any] = field(default=None, repr=False)
  _event_handlers: List[Any] = field(default_factory=list, repr=False)
  _initialized: bool = field(default=False, repr=False)

  def __post_init__(self) -> None:
    if not self.agent_id:
      self.agent_id = f"claude-code-{uuid4().hex[:8]}"
    if not self.agent_name:
      self.agent_name = "ClaudeCodeAgent"

  def _ensure_initialized(self) -> None:
    """Lazy initialization of internal components."""
    if self._initialized:
      return

    # Tool bridge
    all_tools = list(self.tools or [])
    self._tool_bridge = ToolBridge(tools=all_tools, skills=self.skills)

    # Memory resolution
    self._resolve_memory()

    # Knowledge resolution
    self._resolve_knowledge()

    # Tracing resolution
    self._resolve_tracing()

    self._initialized = True

  def _resolve_memory(self) -> None:
    """Resolve memory config into a MemoryManager instance."""
    if self.memory is None or self.memory is False:
      return

    if self.memory is True:
      from definable.memory.manager import MemoryManager
      from definable.memory.store.in_memory import InMemoryStore

      self._memory_manager = MemoryManager(store=InMemoryStore())
      return

    # Check if it's already a Memory / MemoryManager instance
    from definable.memory.manager import Memory, MemoryManager

    if isinstance(self.memory, (Memory, MemoryManager)):
      self._memory_manager = self.memory
      return

    log_warning(f"Unknown memory type: {type(self.memory)}")  # type: ignore[unreachable]

  def _resolve_knowledge(self) -> None:
    """Resolve knowledge config into a Knowledge instance."""
    if self.knowledge is None:
      return

    from definable.knowledge.base import Knowledge

    if isinstance(self.knowledge, Knowledge):
      self._knowledge_instance = self.knowledge
    else:
      log_warning(f"Unknown knowledge type: {type(self.knowledge)}")  # type: ignore[unreachable]

  def _resolve_tracing(self) -> None:
    """Resolve tracing config."""
    if self.tracing is None or self.tracing is False:
      return

    if self.tracing is True:
      from definable.agent.tracing import Tracing

      self._tracing_config = Tracing()
      return

    self._tracing_config = self.tracing

  # ---------------------------------------------------------------------------
  # System prompt construction
  # ---------------------------------------------------------------------------

  def _build_system_prompt(
    self,
    knowledge_ctx: Optional[str] = None,
    memory_ctx: Optional[str] = None,
  ) -> str:
    """Compose the system prompt from instructions, skills, knowledge, and memory."""
    parts: List[str] = []

    # Base instructions
    if self.instructions:
      parts.append(self.instructions)

    # Skill instructions
    if self.skills:
      for skill in self.skills:
        skill_instructions = getattr(skill, "instructions", None)
        if skill_instructions:
          parts.append(f"\n## Skill: {getattr(skill, 'name', 'unnamed')}\n{skill_instructions}")

    # Knowledge context (RAG results)
    if knowledge_ctx:
      parts.append(f"\n<knowledge_context>\n{knowledge_ctx}\n</knowledge_context>")

    # Memory context (user memories)
    if memory_ctx:
      parts.append(f"\n<user_memory>\n{memory_ctx}\n</user_memory>")

    return "\n\n".join(parts) if parts else ""

  # ---------------------------------------------------------------------------
  # CLI args construction
  # ---------------------------------------------------------------------------

  def _build_cli_args(
    self,
    system_prompt: str,
    output_schema: Optional[Union[Type["BaseModel"], Dict[str, Any]]] = None,
    session_id: Optional[str] = None,
  ) -> List[str]:
    """Build the CLI command-line arguments."""
    args = [
      "-p",  # Required for --input-format and --output-format to work
      "--verbose",  # Required for stream-json output with --print
      "--output-format",
      "stream-json",
      "--input-format",
      "stream-json",
      "--model",
      self.model,
      "--permission-mode",
      self.permission_mode,
    ]

    if system_prompt:
      args.extend(["--system-prompt", system_prompt])

    if self.max_turns is not None:
      args.extend(["--max-turns", str(self.max_turns)])

    if self.max_budget_usd is not None:
      args.extend(["--max-budget-usd", str(self.max_budget_usd)])

    if self.thinking_budget_tokens is not None:
      args.extend(["--max-thinking-tokens", str(self.thinking_budget_tokens)])

    # Structured output via JSON schema
    if output_schema is not None:
      schema = self._resolve_output_schema(output_schema)
      if schema:
        args.extend(["--output-format", "json", "--json-schema", json.dumps(schema)])

    # Session resume
    if session_id and self.continue_conversation:
      args.extend(["--resume", session_id])

    # MCP tool server config
    if self._tool_server and self._tool_server.is_running:
      mcp_config = self._tool_server.get_mcp_config_path()
      if mcp_config:
        args.extend(["--mcp-config", mcp_config])

    # Tool allowlist / blocklist
    all_allowed = list(self.allowed_tools or [])
    if self._tool_bridge and self._tool_bridge.tool_count > 0:
      all_allowed.extend(self._tool_bridge.get_tool_names())
    if all_allowed:
      args.extend(["--allowedTools", ",".join(all_allowed)])

    if self.disallowed_tools:
      args.extend(["--disallowedTools", ",".join(self.disallowed_tools)])

    return args

  @staticmethod
  def _resolve_output_schema(schema: Union[Type["BaseModel"], Dict[str, Any]]) -> Optional[dict]:
    """Convert a Pydantic model or dict to a JSON Schema dict."""
    if isinstance(schema, dict):
      return schema
    if hasattr(schema, "model_json_schema"):
      return schema.model_json_schema()
    return None

  # ---------------------------------------------------------------------------
  # Init config (MCP tools, hooks)
  # ---------------------------------------------------------------------------

  async def _start_tool_server(self) -> None:
    """Start the MCP tool server if custom tools are registered."""
    if not self._tool_bridge or self._tool_bridge.tool_count == 0:
      return
    if self._tool_server and self._tool_server.is_running:
      return

    self._tool_server = ToolServer(self._tool_bridge)
    await self._tool_server.start()

  async def _stop_tool_server(self) -> None:
    """Stop the MCP tool server."""
    if self._tool_server:
      await self._tool_server.stop()
      self._tool_server = None

  def _build_user_message(self, prompt: str, messages: Optional[List[Any]] = None) -> dict:
    """Build the user message to send to the CLI."""
    content = prompt
    # If previous messages are provided, include as context
    if messages:
      context_parts = []
      for msg in messages:
        role = getattr(msg, "role", "unknown")
        msg_content = getattr(msg, "content", str(msg))
        if isinstance(msg_content, str):
          context_parts.append(f"<{role}>{msg_content}</{role}>")
      if context_parts:
        content = "<conversation_history>\n" + "\n".join(context_parts) + "\n</conversation_history>\n\n" + prompt

    return {
      "type": "user",
      "message": {"role": "user", "content": content},
    }

  # ---------------------------------------------------------------------------
  # Knowledge and memory helpers
  # ---------------------------------------------------------------------------

  async def _knowledge_retrieve(self, context: RunContext, prompt: str) -> Optional[str]:
    """Retrieve relevant knowledge for the prompt."""
    if not self._knowledge_instance:
      return None

    try:
      top_k = 5
      if self._knowledge_config:
        top_k = getattr(self._knowledge_config, "top_k", 5)

      results = self._knowledge_instance.search(prompt, top_k=top_k)
      if not results:
        return None

      # Format results
      parts = []
      for i, doc in enumerate(results, 1):
        content = getattr(doc, "content", str(doc))
        parts.append(f"[{i}] {content}")
      return "\n\n".join(parts)
    except Exception as exc:
      log_warning(f"Knowledge retrieval failed: {exc}")
      return None

  async def _memory_recall(self, context: RunContext, user_id: Optional[str]) -> Optional[str]:
    """Recall user memories for context injection."""
    if not self._memory_manager or not user_id:
      return None

    try:
      store = getattr(self._memory_manager, "store", None)
      if store is None:
        return None

      memories = await store.get_user_memories(user_id=user_id)
      if not memories:
        return None

      parts = []
      for mem in memories:
        content = getattr(mem, "memory", str(mem))
        parts.append(f"- {content}")
      return "\n".join(parts)
    except Exception as exc:
      log_warning(f"Memory recall failed: {exc}")
      return None

  async def _memory_store(self, prompt: str, response: Optional[str], user_id: Optional[str]) -> None:
    """Fire-and-forget memory storage after a run."""
    if not self._memory_manager or not user_id:
      return

    try:
      store = getattr(self._memory_manager, "store", None)
      if store is None:
        return
      # MemoryManager's update logic is model-driven, but for Claude Code
      # we do a simple store since the memory model may not be available.
      # In practice, users would configure a full MemoryManager with a model.
      log_debug("Memory store: skipping auto-update (no memory model configured for Claude Code agent)")
    except Exception as exc:
      log_warning(f"Memory store failed: {exc}")

  # ---------------------------------------------------------------------------
  # Event emission
  # ---------------------------------------------------------------------------

  def _emit(self, event: Any) -> None:
    """Emit a tracing/lifecycle event."""
    for handler in self._event_handlers:
      try:
        handler(event)
      except Exception as exc:
        log_warning(f"Event handler error: {exc}")

  def _emit_for(self, msg: Message, context: RunContext) -> None:
    """Convert a CLI message to a Definable event and emit it."""
    event = message_to_event(
      msg,
      context,
      agent_id=self.agent_id or "",
      agent_name=self.agent_name or "",
    )
    if event:
      self._emit(event)

  # ---------------------------------------------------------------------------
  # Guardrail runners
  # ---------------------------------------------------------------------------

  async def _run_input_guardrails(self, context: RunContext, prompt: str) -> Optional[RunOutput]:
    """Run input guardrails. Returns a blocked RunOutput if blocked, else None."""
    if not self.guardrails or not self.guardrails.input:
      return None

    results = await self.guardrails.run_input_checks(prompt, context)
    for result in results:
      if result.action == "block":
        return RunOutput(
          run_id=context.run_id,
          session_id=context.session_id,
          agent_id=self.agent_id,
          agent_name=self.agent_name,
          content=f"Input blocked: {result.message}",
          status=RunStatus.blocked,
          model=self.model,
          model_provider="Anthropic",
        )
    return None

  async def _run_output_guardrails(self, context: RunContext, output: RunOutput) -> Optional[RunOutput]:
    """Run output guardrails. Returns modified RunOutput if modified/blocked, else None."""
    if not self.guardrails or not self.guardrails.output:
      return None
    if not output.content or not isinstance(output.content, str):
      return None

    results = await self.guardrails.run_output_checks(output.content, context)
    for result in results:
      if result.action == "block":
        output.content = f"Output blocked: {result.message}"
        output.status = RunStatus.blocked
        return output
      if result.action == "modify" and result.modified_text:
        output.content = result.modified_text
        return output
    return None

  # ---------------------------------------------------------------------------
  # Control protocol handler
  # ---------------------------------------------------------------------------

  async def _handle_control(self, request: ControlRequest, context: RunContext) -> dict:
    """Handle control requests from CLI (permissions, hooks, MCP tools)."""

    if request.subtype == "can_use_tool":
      # Check tool guardrails
      if self.guardrails and self.guardrails.tool:
        results = await self.guardrails.run_tool_checks(
          request.tool_name or "",
          request.input or {},
          context,
        )
        for result in results:
          if result.action == "block":
            return {
              "type": "control_response",
              "id": request.id,
              "behavior": "deny",
              "message": result.message or "Blocked by guardrail",
            }
      return {"type": "control_response", "id": request.id, "behavior": "allow"}

    elif request.subtype == "tool_call":
      # Execute Definable MCP tool
      if self._tool_bridge:
        tool_result = await self._tool_bridge.execute(request.tool_name or "", request.input or {})
        return {"type": "control_response", "id": request.id, "result": tool_result}
      return {
        "type": "control_response",
        "id": request.id,
        "result": {"content": [{"type": "text", "text": "No tool bridge configured"}], "isError": True},
      }

    # Unknown control request — allow by default
    log_debug(f"Unknown control request subtype: {request.subtype}")
    return {"type": "control_response", "id": request.id, "behavior": "allow"}

  # ---------------------------------------------------------------------------
  # Main execution
  # ---------------------------------------------------------------------------

  async def arun(
    self,
    prompt: str,
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    output_schema: Optional[Union[Type["BaseModel"], Dict[str, Any]]] = None,
    messages: Optional[List[Any]] = None,
    **kwargs: Any,
  ) -> RunOutput:
    """Run the Claude Code agent with the given prompt.

    Args:
      prompt: The user message to send.
      user_id: User identifier for memory operations.
      session_id: Session ID for multi-turn conversations.
      output_schema: Pydantic model or JSON Schema for structured output.
      messages: Previous messages for context (from a prior RunOutput.messages).

    Returns:
      RunOutput with content, metrics, tools, and session info.
    """
    self._ensure_initialized()

    # 1. Build RunContext
    context = RunContext(
      run_id=str(uuid4()),
      session_id=session_id or str(uuid4()),
      user_id=user_id,
      output_schema=output_schema,
    )

    # 2. Input guardrails
    block = await self._run_input_guardrails(context, prompt)
    if block:
      return block

    # 3. Build + execute through middleware chain
    async def core_handler(ctx: RunContext) -> RunOutput:
      return await self._execute_run(
        ctx,
        prompt,
        user_id=user_id,
        session_id=session_id,
        output_schema=output_schema,
        messages=messages,
      )

    handler = core_handler
    for mw in reversed(self.middleware or []):
      prev = handler

      async def wrapped(ctx: RunContext, m: Any = mw, h: Any = prev) -> RunOutput:
        return await m(ctx, h)

      handler = wrapped

    return await handler(context)

  async def _execute_run(
    self,
    context: RunContext,
    prompt: str,
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    output_schema: Optional[Union[Type["BaseModel"], Dict[str, Any]]] = None,
    messages: Optional[List[Any]] = None,
  ) -> RunOutput:
    """Core execution: spawn CLI, communicate via JSONL, return RunOutput."""
    self._emit(
      RunStartedEvent(
        agent_id=self.agent_id or "",
        agent_name=self.agent_name or "",
        run_id=context.run_id,
        session_id=context.session_id,
        model=self.model,
        model_provider="Anthropic",
      )
    )

    # 1. Pre-pipeline: knowledge + memory
    knowledge_ctx = await self._knowledge_retrieve(context, prompt)
    memory_ctx = await self._memory_recall(context, user_id)

    # 2. Build system prompt
    system_prompt = self._build_system_prompt(knowledge_ctx, memory_ctx)

    # 3. Start MCP tool server (if custom tools registered)
    await self._start_tool_server()

    # 4. Build CLI args (must be after tool server start for --mcp-config)
    cli_args = self._build_cli_args(system_prompt, output_schema, session_id)

    # 5. Create transport and connect
    transport = SubprocessTransport(
      cli_path=self.cli_path,
      cwd=self.cwd,
      env=self.env,
    )

    try:
      await transport.connect(cli_args)
    except (FileNotFoundError, RuntimeError) as exc:
      await self._stop_tool_server()
      self._emit(
        RunErrorEvent(
          agent_id=self.agent_id or "",
          agent_name=self.agent_name or "",
          run_id=context.run_id,
          session_id=context.session_id,
          content=str(exc),
        )
      )
      return RunOutput(
        run_id=context.run_id,
        session_id=context.session_id,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
        content=f"Failed to start Claude Code CLI: {exc}",
        status=RunStatus.error,
        model=self.model,
        model_provider="Anthropic",
      )

    try:
      # 6. Send user prompt
      user_msg = self._build_user_message(prompt, messages)
      await transport.send(user_msg)

      # 7. Read response messages, handle control protocol
      sdk_messages: List[Message] = []
      async for raw_msg in transport.receive():
        try:
          msg = parse_message(raw_msg)
        except ValueError:
          continue

        if isinstance(msg, ControlRequest):
          response = await self._handle_control(msg, context)
          await transport.send(response)
          continue

        sdk_messages.append(msg)
        self._emit_for(msg, context)

        if isinstance(msg, ResultMessage):
          break

      # 8. Parse into RunOutput
      result = parse_to_run_output(
        sdk_messages,
        context,
        self.model,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
      )

      # 9. Output guardrails
      modified = await self._run_output_guardrails(context, result)
      if modified:
        result = modified

      # 10. Memory store (fire-and-forget)
      if self._memory_manager and user_id:
        asyncio.get_event_loop().create_task(self._memory_store(prompt, result.content if isinstance(result.content, str) else None, user_id))

      self._emit(
        RunCompletedEvent(
          agent_id=self.agent_id or "",
          agent_name=self.agent_name or "",
          run_id=context.run_id,
          session_id=context.session_id,
          content=result.content,
          metrics=result.metrics,
        )
      )

      return result

    except Exception as exc:
      log_error(f"Claude Code agent run failed: {exc}")
      self._emit(
        RunErrorEvent(
          agent_id=self.agent_id or "",
          agent_name=self.agent_name or "",
          run_id=context.run_id,
          session_id=context.session_id,
          content=str(exc),
        )
      )
      return RunOutput(
        run_id=context.run_id,
        session_id=context.session_id,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
        content=f"Error: {exc}",
        status=RunStatus.error,
        model=self.model,
        model_provider="Anthropic",
      )
    finally:
      await transport.close()
      await self._stop_tool_server()

  # ---------------------------------------------------------------------------
  # Streaming
  # ---------------------------------------------------------------------------

  async def arun_stream(
    self,
    prompt: str,
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    output_schema: Optional[Union[Type["BaseModel"], Dict[str, Any]]] = None,
    messages: Optional[List[Any]] = None,
    **kwargs: Any,
  ) -> AsyncIterator[RunOutputEvent]:
    """Stream events from a Claude Code agent run.

    Yields RunOutputEvent instances as the CLI produces output.
    Control requests are handled inline (not yielded).
    """
    self._ensure_initialized()

    context = RunContext(
      run_id=str(uuid4()),
      session_id=session_id or str(uuid4()),
      user_id=user_id,
      output_schema=output_schema,
    )

    # Input guardrails
    block = await self._run_input_guardrails(context, prompt)
    if block:
      yield RunErrorEvent(
        agent_id=self.agent_id or "",
        agent_name=self.agent_name or "",
        run_id=context.run_id,
        session_id=context.session_id,
        content=block.content,
      )
      return

    # Pre-pipeline
    knowledge_ctx = await self._knowledge_retrieve(context, prompt)
    memory_ctx = await self._memory_recall(context, user_id)

    system_prompt = self._build_system_prompt(knowledge_ctx, memory_ctx)

    # Start MCP tool server
    await self._start_tool_server()

    cli_args = self._build_cli_args(system_prompt, output_schema, session_id)

    transport = SubprocessTransport(
      cli_path=self.cli_path,
      cwd=self.cwd,
      env=self.env,
    )

    try:
      await transport.connect(cli_args)
    except (FileNotFoundError, RuntimeError) as exc:
      await self._stop_tool_server()
      yield RunErrorEvent(
        agent_id=self.agent_id or "",
        agent_name=self.agent_name or "",
        run_id=context.run_id,
        session_id=context.session_id,
        content=str(exc),
      )
      return

    try:
      yield RunStartedEvent(
        agent_id=self.agent_id or "",
        agent_name=self.agent_name or "",
        run_id=context.run_id,
        session_id=context.session_id,
        model=self.model,
        model_provider="Anthropic",
      )

      user_msg = self._build_user_message(prompt, messages)
      await transport.send(user_msg)

      async for raw_msg in transport.receive():
        try:
          msg = parse_message(raw_msg)
        except ValueError:
          continue

        if isinstance(msg, ControlRequest):
          response = await self._handle_control(msg, context)
          await transport.send(response)
          continue

        event = message_to_event(
          msg,
          context,
          agent_id=self.agent_id or "",
          agent_name=self.agent_name or "",
        )
        if event:
          yield event  # type: ignore[misc]

        if isinstance(msg, ResultMessage):
          break

    except Exception as exc:
      log_error(f"Claude Code stream failed: {exc}")
      yield RunErrorEvent(
        agent_id=self.agent_id or "",
        agent_name=self.agent_name or "",
        run_id=context.run_id,
        session_id=context.session_id,
        content=str(exc),
      )
    finally:
      await transport.close()
      await self._stop_tool_server()

  # ---------------------------------------------------------------------------
  # Lifecycle
  # ---------------------------------------------------------------------------

  async def __aenter__(self) -> "ClaudeCodeAgent":
    self._ensure_initialized()
    return self

  async def __aexit__(self, *args: Any) -> None:
    await self._stop_tool_server()

  def on_event(self, handler: Any) -> None:
    """Register an event handler for tracing/lifecycle events."""
    self._event_handlers.append(handler)
