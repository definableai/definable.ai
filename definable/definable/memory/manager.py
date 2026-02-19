"""MemoryManager — LLM-driven agentic memory orchestrator.

The MemoryManager lets an LLM decide what to remember about users.
Instead of heuristic scoring, the model is given tools (add_memory,
update_memory, delete_memory, clear_memory) and decides which memories
to create, update, or remove based on the conversation.

Inspired by Agno's MemoryManager pattern.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

from definable.memory.types import UserMemory
from definable.utils.log import log_debug, log_warning


# Default system prompt for the memory LLM
_DEFAULT_SYSTEM_MESSAGE = """\
You are a Memory Manager responsible for managing information about the user.
Your job is to analyze the conversation and decide what facts about the user
are worth remembering for future interactions.

{memory_capture_instructions}

<existing_memories>
{existing_memories}
</existing_memories>

Rules:
- Extract factual information about the user: preferences, personal details,
  goals, decisions, and anything they explicitly state about themselves.
- Do NOT store transient conversation details (greetings, filler, etc.).
- Do NOT store information that is already captured in existing memories.
- If new information contradicts an existing memory, UPDATE the existing one.
- If information is no longer relevant, DELETE it.
- Keep memories concise — one clear fact per memory.
- Assign 1-3 topic tags to each memory for easy filtering.
- You may call multiple tools in a single response.
- If there is nothing worth remembering, respond with "No new memories to add."

{additional_instructions}"""


@dataclass
class Memory:
  """LLM-driven memory orchestrator — a composable lego block.

  The model decides what to remember via tool calls (add/update/delete).
  Store defaults to InMemoryStore. Model defaults to the agent's model.

  Memory works standalone or snaps directly into an Agent:

      agent = Agent(model=model, memory=Memory(store=SQLiteStore("./memory.db")))

  Attributes:
    store: Backend store for persistence. None = InMemoryStore.
    model: LLM for memory extraction. None = uses agent's model at runtime.
    system_message: Custom system prompt for the memory LLM.
    memory_capture_instructions: What kind of information to capture.
    additional_instructions: Extra instructions appended to the system prompt.
    add_memories: Whether the LLM can add new memories.
    update_memories: Whether the LLM can update existing memories.
    delete_memories: Whether the LLM can delete memories (off by default).
    clear_memories: Whether the LLM can clear all memories (off by default).
    debug_mode: Whether to log debug info during memory operations.
    trigger: When to activate memory recall ("always", "auto", "never").
    update_on_run: Whether to process memories after each agent run.
    decision_prompt: Custom YES/NO prompt for trigger="auto".
    description: Description shown in the layer guide (system prompt).
    routing_model: Cheap model for trigger="auto" gate decisions.
  """

  store: Optional[Any] = None
  model: Optional[Any] = None
  system_message: Optional[str] = None
  memory_capture_instructions: Optional[str] = None
  additional_instructions: Optional[str] = None

  # Capability flags
  add_memories: bool = True
  update_memories: bool = True
  delete_memories: bool = False
  clear_memories: bool = False

  debug_mode: bool = False

  # Agent integration fields (absorbed from MemoryConfig)
  trigger: Literal["always", "auto", "never"] = "always"
  update_on_run: bool = True
  decision_prompt: Optional[str] = None
  description: Optional[str] = None
  routing_model: Optional[Any] = None

  _initialized: bool = field(default=False, repr=False)

  async def _ensure_initialized(self) -> None:
    """Lazy-initialize the store on first use."""
    if self._initialized:
      return
    if self.store is None:
      from definable.memory.store.in_memory import InMemoryStore

      self.store = InMemoryStore()
    await self.store.initialize()
    self._initialized = True

  async def close(self) -> None:
    """Close the underlying store."""
    if self.store and self._initialized:
      await self.store.close()
      self._initialized = False

  # --- Public API ---

  async def aget_user_memories(self, user_id: Optional[str] = None) -> List[UserMemory]:
    """Read all memories for a user."""
    await self._ensure_initialized()
    assert self.store is not None  # guaranteed by _ensure_initialized
    return await self.store.get_user_memories(user_id=user_id)

  async def acreate_user_memories(
    self,
    message: Optional[str] = None,
    *,
    messages: Optional[List[Any]] = None,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
  ) -> str:
    """Extract and store memories from a message or messages using the LLM.

    The LLM analyzes the conversation and calls add_memory/update_memory/delete_memory
    tools as appropriate.

    Args:
      message: A single user message string.
      messages: A list of Message objects (takes precedence over message).
      user_id: The user ID for scoping memories.
      agent_id: The agent ID to tag memories with.

    Returns:
      The LLM's response text (usually a summary of actions taken).
    """
    await self._ensure_initialized()

    if self.model is None:
      raise ValueError("MemoryManager requires a model. Set model= or use with an Agent (which provides its model).")

    # Build conversation content — include all roles so the memory LLM sees full context
    if messages:
      parts = []
      for m in messages:
        if hasattr(m, "role") and m.content:
          content = m.content if isinstance(m.content, str) else str(m.content)
          parts.append(f"{m.role}: {content}")
      user_content = "\n".join(parts)
    elif message:
      user_content = message
    else:
      return "No input provided."

    if not user_content.strip():
      return "No user content to process."

    # Get existing memories for context
    assert self.store is not None  # guaranteed by _ensure_initialized
    existing = await self.store.get_user_memories(user_id=user_id)
    existing_text = self.format_memories_for_prompt(existing) if existing else "No existing memories."

    # Build system prompt
    sys_msg = (self.system_message or _DEFAULT_SYSTEM_MESSAGE).format(
      memory_capture_instructions=self.memory_capture_instructions or "Remember important facts about the user.",
      existing_memories=existing_text,
      additional_instructions=self.additional_instructions or "",
    )

    # Build tools
    tool_functions, tool_map = self._build_memory_tools(user_id=user_id, agent_id=agent_id)

    # Convert functions to OpenAI tool format
    tools_dicts = [{"type": "function", "function": f} for f in tool_functions]

    # Run the tool dispatch loop
    return await self._run_tool_loop(
      system_message=sys_msg,
      user_content=user_content,
      tools_dicts=tools_dicts,
      tool_map=tool_map,
    )

  async def aupdate_memory_task(
    self,
    task: str,
    *,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
  ) -> str:
    """Run an arbitrary memory task via the LLM.

    Example tasks:
      - "Remember that the user prefers dark mode"
      - "Forget the user's old address"
      - "Update the user's job title to Senior Engineer"

    Args:
      task: Natural language description of the memory operation.
      user_id: The user ID for scoping memories.
      agent_id: The agent ID to tag memories with.

    Returns:
      The LLM's response text.
    """
    return await self.acreate_user_memories(message=task, user_id=user_id, agent_id=agent_id)

  async def aoptimize_memories(
    self,
    user_id: Optional[str] = None,
    strategy: Optional[str] = None,
  ) -> List[UserMemory]:
    """Run a memory optimization strategy.

    Args:
      user_id: The user whose memories to optimize.
      strategy: Strategy name (currently only "summarize").

    Returns:
      The optimized list of memories.
    """
    await self._ensure_initialized()

    from definable.memory.strategies.types import StrategyFactory, StrategyType

    strategy_type = StrategyType(strategy or "summarize")
    strategy_impl = StrategyFactory.create(strategy_type)

    assert self.store is not None  # guaranteed by _ensure_initialized
    memories = await self.store.get_user_memories(user_id=user_id)
    if not memories:
      return []

    if self.model is None:
      raise ValueError("MemoryManager requires a model for optimization.")

    optimized = await strategy_impl.aoptimize(memories, self.model)

    # Replace all memories with optimized versions
    await self.store.clear_user_memories(user_id=user_id)
    for mem in optimized:
      mem.user_id = user_id
      await self.store.upsert_user_memory(mem)

    return optimized

  def format_memories_for_prompt(self, memories: List[UserMemory]) -> str:
    """Format memories for injection into a system prompt.

    Returns an XML-formatted string suitable for ``<memories_from_previous_interactions>`` tags.
    """
    if not memories:
      return ""

    lines = []
    for i, mem in enumerate(memories, 1):
      topics_str = f" [{', '.join(mem.topics)}]" if mem.topics else ""
      lines.append(f"  {i}. [{mem.memory_id}]{topics_str} {mem.memory}")

    return "\n".join(lines)

  # --- Internal: Tool Dispatch Loop ---

  async def _run_tool_loop(
    self,
    system_message: str,
    user_content: str,
    tools_dicts: List[Dict[str, Any]],
    tool_map: Dict[str, Callable],
    max_rounds: int = 5,
  ) -> str:
    """Run the model with tools until it produces a final answer (no tool calls).

    Args:
      system_message: System prompt for the memory LLM.
      user_content: User message to analyze.
      tools_dicts: OpenAI-format tool definitions.
      tool_map: Mapping from tool name to async callable.
      max_rounds: Maximum tool call rounds to prevent runaway loops.

    Returns:
      The model's final text response.
    """
    from definable.model.message import Message
    from definable.model.response import ModelResponse

    messages: List[Message] = [
      Message(role="system", content=system_message),
      Message(role="user", content=user_content),
    ]

    assert self.model is not None  # caller must set model before _run_tool_loop

    for round_num in range(max_rounds):
      assistant_message = Message(role="assistant", content="")

      response: ModelResponse = await self.model.ainvoke(
        messages=messages,
        assistant_message=assistant_message,
        tools=tools_dicts,
      )

      if self.debug_mode:
        log_debug(f"MemoryManager round {round_num + 1}: content={response.content!r}, tool_calls={len(response.tool_calls)}")

      # If no tool calls, we're done
      if not response.tool_calls:
        return response.content or "Done."

      # Execute tool calls
      # First, add the assistant message with tool calls to history
      messages.append(
        Message(
          role="assistant",
          content=response.content or "",
          tool_calls=response.tool_calls,
        )
      )

      for tc in response.tool_calls:
        tool_name = tc.get("function", {}).get("name", "")
        tool_args_str = tc.get("function", {}).get("arguments", "{}")
        tool_call_id = tc.get("id", "")

        try:
          tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
        except json.JSONDecodeError:
          tool_args = {}

        # Execute the tool
        tool_fn = tool_map.get(tool_name)
        if tool_fn is None:
          result = f"Error: Unknown tool '{tool_name}'"
        else:
          try:
            result = await tool_fn(**tool_args)
          except Exception as e:
            result = f"Error: {e}"
            log_warning(f"MemoryManager tool {tool_name} failed: {e}")

        if self.debug_mode:
          log_debug(f"  Tool {tool_name}({tool_args}) -> {result}")

        # Add tool result message
        messages.append(
          Message(
            role="tool",
            tool_call_id=tool_call_id,
            content=str(result),
          )
        )

    return "Memory processing complete (max rounds reached)."

  # --- Internal: Tool Builders ---

  def _build_memory_tools(
    self,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
  ) -> tuple:
    """Build tool function dicts and callable map for the memory LLM.

    Returns:
      (tool_function_dicts, tool_callable_map) — ready for ainvoke.
    """
    tool_functions: List[Dict[str, Any]] = []
    tool_map: Dict[str, Callable] = {}
    assert self.store is not None  # caller must ensure store is set
    store = self.store

    if self.add_memories:

      async def _add_memory(memory: str, topics: Optional[List[str]] = None) -> str:
        mem = UserMemory(
          memory=memory,
          topics=topics or [],
          user_id=user_id,
          agent_id=agent_id,
        )
        await store.upsert_user_memory(mem)
        return f"Memory added: {mem.memory_id}"

      tool_functions.append({
        "name": "add_memory",
        "description": "Add a new memory about the user. Use this to store facts, preferences, or important information.",
        "parameters": {
          "type": "object",
          "properties": {
            "memory": {"type": "string", "description": "The memory content — a clear, concise fact about the user."},
            "topics": {
              "type": "array",
              "items": {"type": "string"},
              "description": "1-3 topic tags for this memory (e.g. ['preferences', 'work']).",
            },
          },
          "required": ["memory"],
        },
      })
      tool_map["add_memory"] = _add_memory

    if self.update_memories:

      async def _update_memory(memory_id: str, memory: str, topics: Optional[List[str]] = None) -> str:
        existing = await store.get_user_memory(memory_id, user_id=user_id)
        if existing is None:
          return f"Error: Memory {memory_id} not found."
        existing.memory = memory
        if topics is not None:
          existing.topics = topics
        existing.updated_at = time.time()
        await store.upsert_user_memory(existing)
        return f"Memory updated: {memory_id}"

      tool_functions.append({
        "name": "update_memory",
        "description": "Update an existing memory. Use this when new information contradicts or extends a previous memory.",
        "parameters": {
          "type": "object",
          "properties": {
            "memory_id": {"type": "string", "description": "The ID of the memory to update (from existing_memories)."},
            "memory": {"type": "string", "description": "The updated memory content."},
            "topics": {
              "type": "array",
              "items": {"type": "string"},
              "description": "Updated topic tags (optional).",
            },
          },
          "required": ["memory_id", "memory"],
        },
      })
      tool_map["update_memory"] = _update_memory

    if self.delete_memories:

      async def _delete_memory(memory_id: str) -> str:
        await store.delete_user_memory(memory_id, user_id=user_id)
        return f"Memory deleted: {memory_id}"

      tool_functions.append({
        "name": "delete_memory",
        "description": "Delete a memory that is no longer relevant or accurate.",
        "parameters": {
          "type": "object",
          "properties": {
            "memory_id": {"type": "string", "description": "The ID of the memory to delete."},
          },
          "required": ["memory_id"],
        },
      })
      tool_map["delete_memory"] = _delete_memory

    if self.clear_memories:

      async def _clear_memory() -> str:
        await store.clear_user_memories(user_id=user_id)
        return "All memories cleared."

      tool_functions.append({
        "name": "clear_memory",
        "description": "Delete ALL memories for the user. Use with extreme caution.",
        "parameters": {
          "type": "object",
          "properties": {},
        },
      })
      tool_map["clear_memory"] = _clear_memory

    return tool_functions, tool_map


# Backward-compat alias — old code using MemoryManager still works
MemoryManager = Memory
