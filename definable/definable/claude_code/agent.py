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
import os
from dataclasses import dataclass, field
from typing import (
  TYPE_CHECKING,
  Any,
  AsyncIterator,
  Callable,
  Dict,
  List,
  Optional,
  Type,
  Union,
  cast,
)
from uuid import uuid4

from definable.agent.events import (
  RunCompletedEvent,
  RunContext,
  RunErrorEvent,
  RunOutput,
  RunOutputEvent,
  RunPausedEvent,
  RunStartedEvent,
  RunStatus,
)
from definable.agent.run.requirement import RunRequirement
from definable.model.response import ToolExecution
from definable.claude_code.bridge import ToolBridge
from definable.claude_code.parser import message_to_events, parse_to_run_output
from definable.claude_code.tool_server import ToolServer
from definable.claude_code.transport import SubprocessTransport
from definable.claude_code.types import (
  ControlRequest,
  Message,
  ResultMessage,
  parse_message,
)
from definable.tool.function import Function
from definable.utils.log import log_debug, log_error, log_warning

if TYPE_CHECKING:
  from pydantic import BaseModel

  from definable.agent.config import AgentConfig
  from definable.agent.guardrail.base import Guardrails
  from definable.agent.middleware import Middleware
  from definable.agent.tracing import Tracing
  from definable.knowledge.base import Knowledge
  from definable.memory.manager import Memory
  from definable.skill.base import Skill
  from definable.toolkit import Toolkit

# Tools the framework needs regardless of user config.
# AskUserQuestion for HITL questioning, task tools for CLI sticky panel.
_FRAMEWORK_TOOLS = frozenset({
  "AskUserQuestion",
  "TodoWrite",
  "TaskCreate",
  "TaskUpdate",
  "TaskGet",
  "TaskList",
})


@dataclass
class _HitlState:
  """Internal state for a paused HITL run — holds transport + context across pause."""

  transport: SubprocessTransport
  control_request_id: str
  context: RunContext
  sdk_messages: List["Message"]
  prompt: str
  mcp_passthrough_paths: List[str] = field(default_factory=list)
  ask_user_input: Optional[Dict[str, Any]] = None


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
  session_id: Optional[str] = None

  # --- Definable features ---
  memory: Optional[Union["Memory", bool]] = None
  knowledge: Optional["Knowledge"] = None
  guardrails: Optional["Guardrails"] = None
  middleware: Optional[List["Middleware"]] = None
  tools: Optional[List[Function]] = None
  toolkits: Optional[List["Toolkit"]] = None
  skills: Optional[List["Skill"]] = None
  tracing: Optional[Union["Tracing", bool]] = None

  # --- HITL ---
  confirm_tools: Optional[List[str]] = None

  # --- Extended thinking ---
  thinking_budget_tokens: Optional[int] = None

  # --- Session ---
  continue_conversation: bool = False

  # --- Config ---
  config: Optional["AgentConfig"] = None

  # --- Internal state ---
  _tool_bridge: Optional[ToolBridge] = field(default=None, repr=False)
  _tool_server: Optional[ToolServer] = field(default=None, repr=False)
  _memory_manager: Optional["Memory"] = field(default=None, repr=False)
  _knowledge_instance: Optional["Knowledge"] = field(default=None, repr=False)
  _tracing_config: Optional["Tracing"] = field(default=None, repr=False)
  _event_handlers: List[Callable[[object], None]] = field(default_factory=list, repr=False)
  _initialized: bool = field(default=False, repr=False)
  _pending_tasks: list[asyncio.Task[object]] = field(default_factory=list, repr=False)
  _agent_owned_toolkits: list[Any] = field(default_factory=list, repr=False)
  _mcp_config_temps: list[str] = field(default_factory=list, repr=False)
  _hitl_states: Dict[str, _HitlState] = field(default_factory=dict, repr=False)

  def __post_init__(self) -> None:
    if not self.agent_id:
      self.agent_id = f"claude-code-{uuid4().hex[:8]}"
    if not self.agent_name:
      self.agent_name = "ClaudeCodeAgent"
    if not self.session_id:
      self.session_id = str(uuid4())

  def _ensure_initialized(self) -> None:
    """Lazy initialization of internal components."""
    if self._initialized:
      return

    # Skill setup (must happen before ToolBridge so skill tools are available)
    self._init_skills()

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

  def _init_skills(self) -> None:
    """Call setup() on each skill, log errors but don't block."""
    if not self.skills:
      return
    for skill in self.skills:
      try:
        skill.setup()
      except Exception as exc:
        log_warning(f"Skill '{getattr(skill, 'name', 'unknown')}' setup failed: {exc}")

  def _resolve_memory(self) -> None:
    """Resolve memory config into a Memory instance."""
    if self.memory is None or self.memory is False:
      return

    if self.memory is True:
      from definable.memory.manager import Memory
      from definable.memory.store.in_memory import InMemoryStore

      self._memory_manager = Memory(store=InMemoryStore())
      return

    # Check if it's already a Memory instance
    from definable.memory.manager import Memory

    if isinstance(self.memory, Memory):
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

  async def _ensure_toolkits_initialized(self) -> None:
    """Initialize AsyncLifecycleToolkit instances and merge their tools into the bridge.

    MCPToolkit instances are skipped here — they are passed through to the CLI
    via ``--mcp-config`` instead of being double-proxied through ToolBridge.
    See ``_collect_mcp_configs()``.
    """
    if not self.toolkits:
      return

    from definable.mcp.toolkit import MCPToolkit

    for toolkit in self.toolkits:
      # MCPToolkit goes through passthrough (--mcp-config), not ToolBridge
      if isinstance(toolkit, MCPToolkit):
        continue

      try:
        # Check for AsyncLifecycleToolkit protocol
        if hasattr(toolkit, "initialize") and hasattr(toolkit, "_initialized"):
          if not toolkit._initialized:
            await toolkit.initialize()
            self._agent_owned_toolkits.append(toolkit)

        # Merge toolkit tools into the bridge
        toolkit_tools = getattr(toolkit, "tools", None)
        if toolkit_tools and self._tool_bridge:
          tools_to_add = [t for t in toolkit_tools if isinstance(t, Function)]
          if tools_to_add:
            self._tool_bridge._register_tools(tools_to_add)
      except Exception as exc:
        log_warning(f"Toolkit initialization failed: {exc}")

  # ---------------------------------------------------------------------------
  # MCP passthrough (MCPToolkit → --mcp-config)
  # ---------------------------------------------------------------------------

  def _collect_mcp_configs(self) -> list[str]:
    """Convert MCPToolkit server configs into Claude CLI ``--mcp-config`` files.

    For each MCPToolkit in ``self.toolkits``, extracts server definitions and
    writes them to a temp JSON file in the format the Claude CLI expects::

        {"mcpServers": {"name": {"command": ..., "args": ..., "env": ...}}}

    Returns a list of temp file paths (tracked for cleanup).
    """
    if not self.toolkits:
      return []

    import tempfile

    from definable.mcp.toolkit import MCPToolkit

    paths: list[str] = []
    for toolkit in self.toolkits:
      if not isinstance(toolkit, MCPToolkit):
        continue

      servers: Dict[str, Any] = {}
      for server in toolkit.config.servers:
        entry: Dict[str, Any] = {}
        if server.transport == "stdio":
          entry["command"] = server.command or ""
          if server.args:
            entry["args"] = server.args
          if server.env:
            entry["env"] = server.env
        elif server.transport in ("sse", "http"):
          entry["type"] = server.transport
          entry["url"] = server.url or ""
          if server.headers:
            entry["headers"] = server.headers
        servers[server.name] = entry

      if not servers:
        continue

      # Write to temp file
      fd, path = tempfile.mkstemp(prefix="definable_mcp_", suffix=".json")
      try:
        with os.fdopen(fd, "w") as f:
          json.dump({"mcpServers": servers}, f)
      except Exception:
        os.close(fd)
        raise
      paths.append(path)
      self._mcp_config_temps.append(path)

    return paths

  def _cleanup_mcp_temps(self) -> None:
    """Remove temporary MCP config files created by ``_collect_mcp_configs``."""
    import contextlib

    for path in self._mcp_config_temps:
      with contextlib.suppress(OSError):
        os.remove(path)
    self._mcp_config_temps.clear()

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
        try:
          skill_instructions = skill.get_instructions()
        except Exception:
          skill_instructions = getattr(skill, "instructions", None)  # type: ignore[assignment]
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
    mcp_config_paths: Optional[List[str]] = None,
  ) -> List[str]:
    """Build the CLI command-line arguments."""
    # Determine output format upfront — structured output overrides default
    output_format = "stream-json"
    schema_json: Optional[str] = None
    if output_schema is not None:
      schema = self._resolve_output_schema(output_schema)
      if schema:
        output_format = "json"
        schema_json = json.dumps(schema)

    args = [
      "-p",  # Required for --input-format and --output-format to work
      "--verbose",  # Required for stream-json output with --print
      "--output-format",
      output_format,
      "--input-format",
      "stream-json",
      "--model",
      self.model,
      "--permission-mode",
      self.permission_mode,
    ]

    if schema_json:
      args.extend(["--json-schema", schema_json])

    if system_prompt:
      args.extend(["--system-prompt", system_prompt])

    if self.max_turns is not None:
      args.extend(["--max-turns", str(self.max_turns)])

    if self.max_budget_usd is not None:
      args.extend(["--max-budget-usd", str(self.max_budget_usd)])

    if self.thinking_budget_tokens is not None:
      args.extend(["--max-thinking-tokens", str(self.thinking_budget_tokens)])

    # Session resume — passing session_id implies intent to resume
    if session_id:
      args.extend(["--resume", session_id])

    # MCP tool server config (custom @tool functions via ToolBridge)
    if self._tool_server and self._tool_server.is_running:
      mcp_config = self._tool_server.get_mcp_config_path()
      if mcp_config:
        args.extend(["--mcp-config", mcp_config])

    # MCP passthrough configs (MCPToolkit servers passed directly to CLI)
    for path in mcp_config_paths or []:
      args.extend(["--mcp-config", path])

    # Tool allowlist / blocklist
    all_allowed = list(self.allowed_tools or [])
    if self._tool_bridge and self._tool_bridge.tool_count > 0:
      all_allowed.extend(self._tool_bridge.get_tool_names())
    if all_allowed:
      # Inject framework-required tools (AskUserQuestion for HITL, task tools
      # for CLI panel).  Deduplicate in case user already listed them.
      existing = set(all_allowed)
      for fw_tool in _FRAMEWORK_TOOLS:
        if fw_tool not in existing:
          all_allowed.append(fw_tool)
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
      top_k = getattr(self._knowledge_instance, "top_k", 5)
      results = await self._knowledge_instance.asearch(prompt, top_k=top_k)
      if not results:
        return None

      # Use Knowledge's own formatter (respects context_format setting)
      if hasattr(self._knowledge_instance, "format_context"):
        return self._knowledge_instance.format_context(results)

      # Fallback for custom knowledge implementations
      parts = []
      for i, doc in enumerate(results, 1):
        content = getattr(doc, "content", str(doc))
        parts.append(f"[{i}] {content}")
      return "\n\n".join(parts)
    except Exception as exc:
      log_warning(f"Knowledge retrieval failed: {exc}")
      return None

  async def _memory_recall(self, context: RunContext) -> Optional[str]:
    """Recall session history for context injection."""
    if not self._memory_manager:
      return None

    try:
      session_id = context.session_id or "default"
      user_id = context.user_id or "default"
      await self._memory_manager._optimize_if_needed(session_id, user_id)
      entries = await self._memory_manager.get_entries(session_id, user_id, limit=50)
      if not entries:
        return None

      parts = []
      for entry in entries:
        if entry.role == "summary":
          parts.append(f"[Summary]: {entry.content}")
        else:
          parts.append(f"{entry.role}: {entry.content}")
      return "<conversation_history>\n" + "\n".join(parts) + "\n</conversation_history>"
    except Exception as exc:
      log_warning(f"Memory recall failed: {exc}")
      return None

  async def _memory_store(self, context: RunContext, prompt: str, response: Optional[str]) -> None:
    """Store messages in session memory after a run."""
    if not self._memory_manager:
      return

    try:
      from definable.model.message import Message as DefMessage

      session_id = context.session_id or "default"
      user_id = context.user_id or "default"
      await self._memory_manager._ensure_initialized()
      await self._memory_manager.add(DefMessage(role="user", content=prompt), session_id=session_id, user_id=user_id)
      if response:
        await self._memory_manager.add(DefMessage(role="assistant", content=response), session_id=session_id, user_id=user_id)
    except Exception as exc:
      log_warning(f"Memory store failed: {exc}")

  # ---------------------------------------------------------------------------
  # Event emission
  # ---------------------------------------------------------------------------

  def _emit(self, event: object) -> None:
    """Emit a tracing/lifecycle event."""
    for handler in self._event_handlers:
      try:
        handler(event)
      except Exception as exc:
        log_warning(f"Event handler error: {exc}")

  def _emit_for(self, msg: Message, context: RunContext) -> None:
    """Convert a CLI message to Definable events and emit them."""
    events = message_to_events(
      msg,
      context,
      agent_id=self.agent_id or "",
      agent_name=self.agent_name or "",
    )
    for event in events:
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

  async def _handle_control(self, request: ControlRequest, context: RunContext) -> Union[dict, RunPausedEvent]:
    """Handle control requests from CLI (permissions, hooks, MCP tools).

    Returns a dict control response to send back to the CLI, or a
    ``RunPausedEvent`` when HITL confirmation is required (``confirm_tools``).
    """
    log_debug(f"ControlRequest: subtype={request.subtype}, tool={request.tool_name}, id={request.id}")

    if request.subtype == "can_use_tool":
      # 0. AskUserQuestion — pause for interactive user input
      if request.tool_name == "AskUserQuestion":
        raw_input = request.input or {}
        questions = raw_input.get("questions", [])
        if not questions:
          # Empty questions — auto-allow with empty answers
          return {
            "type": "control_response",
            "id": request.id,
            "behavior": "allow",
            "updatedInput": {"questions": [], "answers": {}},
          }
        tool_exec = ToolExecution(
          tool_name="AskUserQuestion",
          tool_args=raw_input,
          requires_user_input=True,
        )
        requirement = RunRequirement(tool_exec)
        return RunPausedEvent(
          run_id=context.run_id,
          session_id=context.session_id,
          agent_id=self.agent_id or "",
          agent_name=self.agent_name or "",
          tools=[tool_exec],
          requirements=[requirement],
        )

      # 1. Guardrails take precedence — can block outright
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

      # 2. HITL check — pause for user confirmation
      if self.confirm_tools and (request.tool_name or "") in self.confirm_tools:
        tool_exec = ToolExecution(
          tool_name=request.tool_name or "",
          tool_args=request.input or {},
          requires_confirmation=True,
        )
        requirement = RunRequirement(tool_exec)
        return RunPausedEvent(
          run_id=context.run_id,
          session_id=context.session_id,
          agent_id=self.agent_id or "",
          agent_name=self.agent_name or "",
          tools=[tool_exec],
          requirements=[requirement],
        )

      # 3. Auto-allow
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
      session_id=session_id or self.session_id or str(uuid4()),
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
    memory_ctx = await self._memory_recall(context)

    # 2. Build system prompt
    system_prompt = self._build_system_prompt(knowledge_ctx, memory_ctx)

    # 3. Initialize toolkits + start MCP tool server
    await self._ensure_toolkits_initialized()
    await self._start_tool_server()

    # 3b. Collect MCP passthrough configs (MCPToolkit → --mcp-config)
    mcp_passthrough_paths = self._collect_mcp_configs()

    # 4. Build CLI args (must be after tool server start for --mcp-config)
    cli_args = self._build_cli_args(system_prompt, output_schema, session_id, mcp_passthrough_paths)

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

    paused = False
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

          # HITL: _handle_control returned a pause event
          if isinstance(response, RunPausedEvent):
            self._hitl_states[context.run_id] = _HitlState(
              transport=transport,
              control_request_id=msg.id,
              context=context,
              sdk_messages=sdk_messages,
              prompt=prompt,
              mcp_passthrough_paths=mcp_passthrough_paths,
              ask_user_input=msg.input if msg.tool_name == "AskUserQuestion" else None,
            )
            result = parse_to_run_output(
              sdk_messages,
              context,
              self.model,
              agent_id=self.agent_id,
              agent_name=self.agent_name,
            )
            result.status = RunStatus.paused
            result.requirements = response.requirements
            self._emit(response)
            paused = True
            return result

          await transport.send(response)
          continue

        sdk_messages.append(msg)

        # Skip _emit_for on ResultMessage — the explicit RunCompletedEvent
        # below (after guardrails) has richer data (content + metrics).
        if isinstance(msg, ResultMessage):
          break

        self._emit_for(msg, context)

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

      # 10. Memory store (fire-and-forget with tracking)
      if self._memory_manager:
        task = asyncio.create_task(self._memory_store(context, prompt, result.content if isinstance(result.content, str) else None))
        self._pending_tasks.append(task)
        task.add_done_callback(lambda t: self._pending_tasks.remove(t) if t in self._pending_tasks else None)

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
      if not paused:
        await transport.close()
        await self._stop_tool_server()
        self._cleanup_mcp_temps()

  # ---------------------------------------------------------------------------
  # Streaming execution
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
    """Stream events from a Claude Code CLI run.

    Yields ``RunOutputEvent`` instances (``RunContentEvent``,
    ``ToolCallStartedEvent``, ``RunPausedEvent``, ``RunCompletedEvent``, etc.)
    as they arrive from the CLI subprocess.

    No middleware wrapping — matches the regular Agent streaming contract.

    When ``confirm_tools`` is set and the CLI asks permission for a matching
    tool, a ``RunPausedEvent`` is yielded and the generator returns.  Call
    ``continue_run_stream()`` to resume.
    """
    self._ensure_initialized()

    context = RunContext(
      run_id=str(uuid4()),
      session_id=session_id or self.session_id or str(uuid4()),
      user_id=user_id,
      output_schema=output_schema,
    )

    # Input guardrails
    block = await self._run_input_guardrails(context, prompt)
    if block:
      yield RunCompletedEvent(
        agent_id=self.agent_id or "",
        agent_name=self.agent_name or "",
        run_id=context.run_id,
        session_id=context.session_id,
        content=block.content,
      )
      return

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

    # Pre-pipeline
    knowledge_ctx = await self._knowledge_retrieve(context, prompt)
    memory_ctx = await self._memory_recall(context)
    system_prompt = self._build_system_prompt(knowledge_ctx, memory_ctx)

    await self._ensure_toolkits_initialized()
    await self._start_tool_server()
    mcp_passthrough_paths = self._collect_mcp_configs()
    cli_args = self._build_cli_args(system_prompt, output_schema, session_id, mcp_passthrough_paths)

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

    paused = False
    try:
      user_msg = self._build_user_message(prompt, messages)
      await transport.send(user_msg)

      sdk_messages: List[Message] = []
      async for raw_msg in transport.receive():
        try:
          msg = parse_message(raw_msg)
        except ValueError:
          continue

        if isinstance(msg, ControlRequest):
          response = await self._handle_control(msg, context)

          if isinstance(response, RunPausedEvent):
            self._hitl_states[context.run_id] = _HitlState(
              transport=transport,
              control_request_id=msg.id,
              context=context,
              sdk_messages=sdk_messages,
              prompt=prompt,
              mcp_passthrough_paths=mcp_passthrough_paths,
              ask_user_input=msg.input if msg.tool_name == "AskUserQuestion" else None,
            )
            self._emit(response)
            paused = True
            yield response
            return

          await transport.send(response)
          continue

        sdk_messages.append(msg)

        if isinstance(msg, ResultMessage):
          break

        # Yield events to caller
        events = message_to_events(msg, context, agent_id=self.agent_id or "", agent_name=self.agent_name or "")
        for event in events:
          self._emit(event)
          yield cast(RunOutputEvent, event)

      # Final: parse RunOutput for guardrails + memory + completion event
      result = parse_to_run_output(
        sdk_messages,
        context,
        self.model,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
      )

      modified = await self._run_output_guardrails(context, result)
      if modified:
        result = modified

      if self._memory_manager:
        task = asyncio.create_task(self._memory_store(context, prompt, result.content if isinstance(result.content, str) else None))
        self._pending_tasks.append(task)
        task.add_done_callback(lambda t: self._pending_tasks.remove(t) if t in self._pending_tasks else None)

      completed_event = RunCompletedEvent(
        agent_id=self.agent_id or "",
        agent_name=self.agent_name or "",
        run_id=context.run_id,
        session_id=context.session_id,
        content=result.content,
        metrics=result.metrics,
      )
      self._emit(completed_event)
      yield completed_event

    except Exception as exc:
      log_error(f"Claude Code agent stream failed: {exc}")
      error_event = RunErrorEvent(
        agent_id=self.agent_id or "",
        agent_name=self.agent_name or "",
        run_id=context.run_id,
        session_id=context.session_id,
        content=str(exc),
      )
      self._emit(error_event)
      yield error_event
    finally:
      if not paused:
        await transport.close()
        await self._stop_tool_server()
        self._cleanup_mcp_temps()

  # ---------------------------------------------------------------------------
  # HITL continue methods
  # ---------------------------------------------------------------------------

  async def continue_run(self, *, run_output: RunOutput) -> RunOutput:
    """Resume a paused HITL run after user has confirmed/rejected requirements.

    Args:
      run_output: The paused ``RunOutput`` (``is_paused`` must be ``True``).

    Returns:
      A new ``RunOutput`` — may be paused again if another tool needs confirmation.
    """
    if not run_output.is_paused:
      raise ValueError("RunOutput is not paused — nothing to continue")

    run_id = run_output.run_id or ""
    state = self._hitl_states.pop(run_id, None)
    if state is None:
      raise ValueError(f"No HITL state found for run_id={run_id}")

    # Build control response — AskUserQuestion gets updatedInput with answers
    if state.ask_user_input is not None:
      # Extract answers from the resolved requirement's tool_args
      answers: Dict[str, str] = {}
      if run_output.requirements:
        for req in run_output.requirements:
          if req.tool_execution and req.tool_execution.tool_args:
            answers = req.tool_execution.tool_args.get("_answers", {})
            break
      control_response = {
        "type": "control_response",
        "id": state.control_request_id,
        "behavior": "allow",
        "updatedInput": {
          "questions": state.ask_user_input.get("questions", []),
          "answers": answers,
        },
      }
    else:
      # Determine allow/deny from resolved requirements
      behavior = "allow"
      if run_output.requirements:
        for req in run_output.requirements:
          if req.confirmation is False:
            behavior = "deny"
            break
      control_response = {
        "type": "control_response",
        "id": state.control_request_id,
        "behavior": behavior,
      }

    paused = False
    try:
      await state.transport.send(control_response)

      # Continue reading from the same transport
      async for raw_msg in state.transport.receive():
        try:
          msg = parse_message(raw_msg)
        except ValueError:
          continue

        if isinstance(msg, ControlRequest):
          response = await self._handle_control(msg, state.context)

          if isinstance(response, RunPausedEvent):
            self._hitl_states[run_id] = _HitlState(
              transport=state.transport,
              control_request_id=msg.id,
              context=state.context,
              sdk_messages=state.sdk_messages,
              prompt=state.prompt,
              mcp_passthrough_paths=state.mcp_passthrough_paths,
              ask_user_input=msg.input if msg.tool_name == "AskUserQuestion" else None,
            )
            result = parse_to_run_output(
              state.sdk_messages,
              state.context,
              self.model,
              agent_id=self.agent_id,
              agent_name=self.agent_name,
            )
            result.status = RunStatus.paused
            result.requirements = response.requirements
            self._emit(response)
            paused = True
            return result

          await state.transport.send(response)
          continue

        state.sdk_messages.append(msg)
        if isinstance(msg, ResultMessage):
          break
        self._emit_for(msg, state.context)

      result = parse_to_run_output(
        state.sdk_messages,
        state.context,
        self.model,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
      )

      modified = await self._run_output_guardrails(state.context, result)
      if modified:
        result = modified

      if self._memory_manager:
        task = asyncio.create_task(self._memory_store(state.context, state.prompt, result.content if isinstance(result.content, str) else None))
        self._pending_tasks.append(task)
        task.add_done_callback(lambda t: self._pending_tasks.remove(t) if t in self._pending_tasks else None)

      self._emit(
        RunCompletedEvent(
          agent_id=self.agent_id or "",
          agent_name=self.agent_name or "",
          run_id=state.context.run_id,
          session_id=state.context.session_id,
          content=result.content,
          metrics=result.metrics,
        )
      )
      return result

    except Exception as exc:
      log_error(f"Claude Code agent continue_run failed: {exc}")
      self._emit(
        RunErrorEvent(
          agent_id=self.agent_id or "",
          agent_name=self.agent_name or "",
          run_id=state.context.run_id,
          session_id=state.context.session_id,
          content=str(exc),
        )
      )
      return RunOutput(
        run_id=state.context.run_id,
        session_id=state.context.session_id,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
        content=f"Error: {exc}",
        status=RunStatus.error,
        model=self.model,
        model_provider="Anthropic",
      )
    finally:
      if not paused:
        await state.transport.close()
        await self._stop_tool_server()
        self._cleanup_mcp_temps()

  async def continue_run_stream(self, *, run_output: Union[RunOutput, RunPausedEvent]) -> AsyncIterator[RunOutputEvent]:
    """Resume a paused HITL run as a streaming generator.

    Args:
      run_output: The paused ``RunOutput`` or ``RunPausedEvent``.

    Yields:
      ``RunOutputEvent`` instances until completion or another pause.
    """
    # Extract run_id from either RunOutput or RunPausedEvent
    if isinstance(run_output, RunPausedEvent):
      run_id = run_output.run_id or ""
      requirements = run_output.requirements
    else:
      if not run_output.is_paused:
        raise ValueError("RunOutput is not paused — nothing to continue")
      run_id = run_output.run_id or ""
      requirements = run_output.requirements

    state = self._hitl_states.pop(run_id, None)
    if state is None:
      raise ValueError(f"No HITL state found for run_id={run_id}")

    # Build control response — AskUserQuestion gets updatedInput with answers
    if state.ask_user_input is not None:
      answers_dict: Dict[str, str] = {}
      if requirements:
        for req in requirements:
          if req.tool_execution and req.tool_execution.tool_args:
            answers_dict = req.tool_execution.tool_args.get("_answers", {})
            break
      control_response = {
        "type": "control_response",
        "id": state.control_request_id,
        "behavior": "allow",
        "updatedInput": {
          "questions": state.ask_user_input.get("questions", []),
          "answers": answers_dict,
        },
      }
    else:
      behavior = "allow"
      if requirements:
        for req in requirements:
          if req.confirmation is False:
            behavior = "deny"
            break
      control_response = {
        "type": "control_response",
        "id": state.control_request_id,
        "behavior": behavior,
      }

    paused = False
    try:
      await state.transport.send(control_response)

      async for raw_msg in state.transport.receive():
        try:
          msg = parse_message(raw_msg)
        except ValueError:
          continue

        if isinstance(msg, ControlRequest):
          response = await self._handle_control(msg, state.context)

          if isinstance(response, RunPausedEvent):
            self._hitl_states[run_id] = _HitlState(
              transport=state.transport,
              control_request_id=msg.id,
              context=state.context,
              sdk_messages=state.sdk_messages,
              prompt=state.prompt,
              mcp_passthrough_paths=state.mcp_passthrough_paths,
              ask_user_input=msg.input if msg.tool_name == "AskUserQuestion" else None,
            )
            self._emit(response)
            paused = True
            yield response
            return

          await state.transport.send(response)
          continue

        state.sdk_messages.append(msg)
        if isinstance(msg, ResultMessage):
          break

        events = message_to_events(msg, state.context, agent_id=self.agent_id or "", agent_name=self.agent_name or "")
        for event in events:
          self._emit(event)
          yield cast(RunOutputEvent, event)

      result = parse_to_run_output(
        state.sdk_messages,
        state.context,
        self.model,
        agent_id=self.agent_id,
        agent_name=self.agent_name,
      )

      modified = await self._run_output_guardrails(state.context, result)
      if modified:
        result = modified

      if self._memory_manager:
        task = asyncio.create_task(self._memory_store(state.context, state.prompt, result.content if isinstance(result.content, str) else None))
        self._pending_tasks.append(task)
        task.add_done_callback(lambda t: self._pending_tasks.remove(t) if t in self._pending_tasks else None)

      completed_event = RunCompletedEvent(
        agent_id=self.agent_id or "",
        agent_name=self.agent_name or "",
        run_id=state.context.run_id,
        session_id=state.context.session_id,
        content=result.content,
        metrics=result.metrics,
      )
      self._emit(completed_event)
      yield completed_event

    except Exception as exc:
      log_error(f"Claude Code agent continue_run_stream failed: {exc}")
      error_event = RunErrorEvent(
        agent_id=self.agent_id or "",
        agent_name=self.agent_name or "",
        run_id=state.context.run_id,
        session_id=state.context.session_id,
        content=str(exc),
      )
      self._emit(error_event)
      yield error_event
    finally:
      if not paused:
        await state.transport.close()
        await self._stop_tool_server()
        self._cleanup_mcp_temps()

  # ---------------------------------------------------------------------------
  # Lifecycle
  # ---------------------------------------------------------------------------

  async def __aenter__(self) -> "ClaudeCodeAgent":
    self._ensure_initialized()
    return self

  async def __aexit__(self, *args: object) -> None:
    # Close any dangling HITL transports (abandoned paused runs)
    for state in self._hitl_states.values():
      try:
        await state.transport.close()
      except Exception as exc:
        log_warning(f"HITL transport cleanup failed: {exc}")
    self._hitl_states.clear()

    # Drain pending memory tasks
    await self._drain_pending_tasks()

    # Stop tool server
    await self._stop_tool_server()

    # Clean up MCP passthrough temp files
    self._cleanup_mcp_temps()

    # Shutdown agent-owned toolkits
    for toolkit in self._agent_owned_toolkits:
      try:
        await toolkit.shutdown()
      except Exception as exc:
        log_warning(f"Toolkit shutdown failed: {exc}")
    self._agent_owned_toolkits.clear()

    # Close memory store
    if self._memory_manager:
      try:
        await self._memory_manager.close()
      except Exception as exc:
        log_warning(f"Memory close failed: {exc}")

    # Teardown skills
    if self.skills:
      for skill in self.skills:
        try:
          skill.teardown()
        except Exception as exc:
          log_warning(f"Skill '{getattr(skill, 'name', 'unknown')}' teardown failed: {exc}")

  async def _drain_pending_tasks(self) -> None:
    """Wait for all pending fire-and-forget tasks to complete."""
    if self._pending_tasks:
      await asyncio.gather(*self._pending_tasks, return_exceptions=True)
      self._pending_tasks.clear()

  def on_event(self, handler: Callable[[object], None]) -> None:
    """Register an event handler for tracing/lifecycle events."""
    self._event_handlers.append(handler)
