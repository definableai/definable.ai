"""Memory — session-history buffer with auto-summarization.

The Memory block manages conversation history per session. Messages are
stored as MemoryEntry objects. When the history exceeds max_messages and
a model is available, middle entries are automatically summarized.

Quick Start:
    from definable.memory import Memory, SQLiteStore

    memory = Memory(store=SQLiteStore("./memory.db"))
    agent = Agent(model=model, memory=memory)
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from definable.memory.types import MemoryEntry
from definable.utils.log import log_debug


@dataclass
class Memory:
  """Session-history memory block — a composable lego block.

  Stores conversation messages per session with auto-summarization.
  Store defaults to InMemoryStore. Model is required for auto-optimization.

  Memory snaps directly into Agent:

      agent = Agent(model=model, memory=Memory(store=SQLiteStore("./memory.db")))

  Attributes:
    store: Backend store. None = InMemoryStore (auto-created).
    model: LLM for summarization. None = uses agent's model at runtime.
    enabled: Whether memory is active.
    max_messages: Threshold for auto-optimization.
    pin_count: How many initial messages to preserve during optimization.
    recent_count: How many recent messages to preserve during optimization.
    description: Description shown in the agent layer guide.
  """

  store: Optional[Any] = None
  model: Optional[Any] = None
  enabled: bool = True
  max_messages: int = 100
  pin_count: int = 2
  recent_count: int = 5
  description: Optional[str] = None

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

  async def add(self, message: Any, session_id: str = "default", user_id: str = "default") -> None:
    """Add a message to session memory.

    Converts the Message to a MemoryEntry and stores it.
    Triggers auto-optimization if count exceeds max_messages.

    Args:
      message: A Message object (with role, content, optionally tool_calls).
      session_id: Session to store in.
      user_id: User scope.
    """
    if not self.enabled:
      return

    await self._ensure_initialized()

    # Build message_data for full reconstruction
    msg_data: Optional[dict] = None
    if hasattr(message, "tool_calls") and message.tool_calls:
      msg_data = {
        "role": message.role,
        "content": message.content if isinstance(message.content, str) else str(message.content or ""),
        "tool_calls": message.tool_calls,
      }
    elif hasattr(message, "tool_call_id") and message.tool_call_id:
      msg_data = {
        "role": message.role,
        "content": message.content if isinstance(message.content, str) else str(message.content or ""),
        "tool_call_id": message.tool_call_id,
      }

    content = message.content if isinstance(message.content, str) else str(message.content or "")
    entry = MemoryEntry(
      session_id=session_id,
      user_id=user_id,
      role=getattr(message, "role", "user"),
      content=content,
      message_data=msg_data,
    )

    assert self.store is not None
    await self.store.add(entry)

    # Auto-optimize if threshold exceeded
    await self._optimize_if_needed(session_id, user_id)

  async def get_entries(self, session_id: str, user_id: str = "default") -> List[MemoryEntry]:
    """Get all entries for a session."""
    await self._ensure_initialized()
    assert self.store is not None
    return await self.store.get_entries(session_id, user_id)

  async def get_context_messages(self, session_id: str, user_id: str = "default") -> List[Any]:
    """Get entries as Message objects for injection into a conversation.

    Summary entries are converted to system messages.
    Entries with message_data are reconstructed with full tool_calls.
    """
    from definable.model.message import Message

    entries = await self.get_entries(session_id, user_id)
    messages: List[Message] = []
    for entry in entries:
      if entry.role == "summary":
        messages.append(Message(role="system", content=entry.content))
      elif entry.message_data:
        # Reconstruct full message from stored data
        msg_kwargs: dict = {
          "role": entry.message_data.get("role", entry.role),
          "content": entry.message_data.get("content", entry.content),
        }
        if "tool_calls" in entry.message_data:
          msg_kwargs["tool_calls"] = entry.message_data["tool_calls"]
        if "tool_call_id" in entry.message_data:
          msg_kwargs["tool_call_id"] = entry.message_data["tool_call_id"]
        messages.append(Message(**msg_kwargs))
      else:
        messages.append(Message(role=entry.role, content=entry.content))

    return messages

  async def update(self, memory_id: str, content: str) -> None:
    """Update the content of an entry."""
    await self._ensure_initialized()
    assert self.store is not None
    entry = await self.store.get_entry(memory_id)
    if entry is not None:
      entry.content = content
      await self.store.update(entry)

  async def delete(self, memory_id: str) -> None:
    """Delete an entry by ID."""
    await self._ensure_initialized()
    assert self.store is not None
    await self.store.delete(memory_id)

  async def clear(self, session_id: str) -> None:
    """Clear all entries for a session."""
    await self._ensure_initialized()
    assert self.store is not None
    await self.store.delete_session(session_id)

  # --- Auto-optimization ---

  async def _optimize_if_needed(self, session_id: str, user_id: str) -> None:
    """Run summarize strategy if entry count exceeds max_messages."""
    if self.model is None:
      return  # No model = can't summarize

    assert self.store is not None
    count = await self.store.count(session_id, user_id)
    if count <= self.max_messages:
      return

    from definable.memory.strategies.summarize import SummarizeStrategy

    strategy = SummarizeStrategy(pin_count=self.pin_count, recent_count=self.recent_count)
    entries = await self.store.get_entries(session_id, user_id)
    optimized = await strategy.optimize(entries, self.model)

    if len(optimized) < len(entries):
      # Replace entries with optimized version
      await self.store.delete_session(session_id, user_id)
      for entry in optimized:
        await self.store.add(entry)
      log_debug(f"Memory optimized: {len(entries)} entries -> {len(optimized)} (session={session_id})")

  # --- Lifecycle ---

  async def __aenter__(self) -> "Memory":
    await self._ensure_initialized()
    return self

  async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    await self.close()
