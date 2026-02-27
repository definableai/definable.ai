"""Memory — session-history buffer with configurable optimization strategies.

The Memory block manages conversation history per session. Messages are
stored as MemoryEntry objects. When the history exceeds max_messages and
a model is available, the configured strategy optimizes the entries.

Composition: Memory = Strategy + Store + Optional[Embedder].

Strategies:
    - SummarizeStrategy (default): pin first N + summarize middle + keep recent M.
    - SemanticStrategy: extract atomic, self-contained facts from conversation.

When an embedder is provided, the recall path becomes similarity-ranked
instead of chronological. Atoms carry their own embedding vectors, and
queries are embedded at recall time for cosine-similarity ranking.

Quick Start:
    from definable.memory import Memory, SQLiteStore

    memory = Memory(store=SQLiteStore("./memory.db"))
    agent = Agent(model=model, memory=memory)

    # With semantic extraction + smart recall:
    from definable.memory import SemanticStrategy
    from definable.embedder import OpenAIEmbedder
    memory = Memory(
        store=SQLiteStore("./memory.db"),
        strategy=SemanticStrategy(),
        embedder=OpenAIEmbedder(),
    )
"""

import math
from types import TracebackType
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
  from definable.knowledge.embedder.base import Embedder
  from definable.memory.consolidation import ConsolidationPolicy
  from definable.memory.store.base import MemoryStore
  from definable.memory.strategies.base import MemoryStrategy
  from definable.model.base import Model
  from definable.model.message import Message

from definable.memory.types import MemoryEntry
from definable.utils.log import log_debug, log_warning


@dataclass
class Memory:
  """Session-history memory block — a composable lego block.

  Composition: Memory = Strategy + Store + Optional[Embedder].

  Store defaults to InMemoryStore. Strategy defaults to SummarizeStrategy.
  Model is required for auto-optimization.

  When ``embedder`` is set, two things happen:
    1. Atoms produced by SemanticStrategy are auto-embedded (vectors stored on entries).
    2. Recall uses similarity-ranked retrieval: recent messages (STM) + top-K
       atoms by cosine similarity (LTM), merged into context.

  Memory snaps directly into Agent:

      agent = Agent(model=model, memory=Memory(store=SQLiteStore("./memory.db")))

      # Full semantic memory (extraction + smart recall):
      agent = Agent(model=model, memory=Memory(
          store=SQLiteStore("./memory.db"),
          strategy=SemanticStrategy(),
          embedder=OpenAIEmbedder(),
      ))

  Attributes:
    store: Backend store. None = InMemoryStore (auto-created).
    strategy: Optimization strategy. None = SummarizeStrategy (auto-created).
    embedder: Embedding model for similarity-ranked recall. None = chronological.
    consolidation: Policy for atom decay/merge/prune. None = disabled.
    model: LLM for strategy optimization. None = uses agent's model at runtime.
    enabled: Whether memory is active.
    max_messages: Threshold for auto-optimization.
    pin_count: How many initial messages to preserve during optimization.
    recent_count: How many recent messages to preserve during optimization.
    search_top_k: Max atoms returned by similarity search.
    description: Description shown in the agent layer guide.
  """

  store: Optional["MemoryStore"] = None
  strategy: Optional["MemoryStrategy"] = None
  embedder: Optional["Embedder"] = None
  consolidation: Optional["ConsolidationPolicy"] = None
  model: Optional["Model"] = None
  enabled: bool = True
  max_messages: int = 100
  pin_count: int = 2
  recent_count: int = 5
  search_top_k: int = 10
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

  async def add(self, message: "Message", session_id: str = "default", user_id: str = "default") -> None:
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

  async def get_entries(self, session_id: str, user_id: str = "default", limit: Optional[int] = None) -> List[MemoryEntry]:
    """Get all entries for a session, optionally limited to the last N."""
    await self._ensure_initialized()
    assert self.store is not None
    entries = await self.store.get_entries(session_id, user_id)
    if limit is not None:
      return entries[-limit:]
    return entries

  async def get_context_messages(self, session_id: str, user_id: str = "default") -> list["Message"]:
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

  # --- Semantic Search ---

  async def search(
    self,
    query: str,
    session_id: str,
    user_id: str = "default",
    top_k: Optional[int] = None,
  ) -> List[MemoryEntry]:
    """Similarity-ranked retrieval of atom entries.

    Embeds the query, computes cosine similarity against all atom entries
    that have stored vectors, and returns top-K ranked results.

    Falls back to returning all atoms chronologically if no embedder or
    no atoms have vectors.

    Args:
      query: The search query text.
      session_id: Session to search in.
      user_id: User scope.
      top_k: Max results. Defaults to self.search_top_k.

    Returns:
      List of MemoryEntry atoms ranked by relevance.
    """
    await self._ensure_initialized()
    assert self.store is not None
    k = top_k or self.search_top_k

    # Get all entries, filter to atoms.
    entries = await self.store.get_entries(session_id, user_id)
    atoms = [e for e in entries if e.entry_type == "atom" and e.superseded_by is None]

    if not atoms:
      return []

    # If no embedder, return chronological atoms (best effort).
    if self.embedder is None:
      return atoms[-k:]

    # Filter to atoms that have stored vectors.
    vectorized = [a for a in atoms if a.vector]
    if not vectorized:
      return atoms[-k:]

    # Embed the query.
    try:
      query_vector = await self.embedder.async_get_embedding(query)
    except Exception as exc:
      log_warning(f"Memory search: embedding query failed: {exc}")
      return atoms[-k:]

    if not query_vector:
      return atoms[-k:]

    # Rank by cosine similarity.
    scored = []
    for atom in vectorized:
      sim = _cosine_similarity(query_vector, atom.vector)  # type: ignore[arg-type]
      scored.append((sim, atom))
    scored.sort(key=lambda x: x[0], reverse=True)

    return [atom for _, atom in scored[:k]]

  @property
  def has_semantic_search(self) -> bool:
    """Whether this Memory instance supports similarity-ranked recall."""
    return self.embedder is not None

  # --- Auto-optimization ---

  def _resolve_strategy(self) -> "MemoryStrategy":
    """Return the configured strategy, defaulting to SummarizeStrategy."""
    if self.strategy is not None:
      return self.strategy
    from definable.memory.strategies.summarize import SummarizeStrategy

    return SummarizeStrategy(pin_count=self.pin_count, recent_count=self.recent_count)

  async def _optimize_if_needed(self, session_id: str, user_id: str) -> None:
    """Run the configured strategy if entry count exceeds max_messages."""
    if self.model is None:
      return  # No model = can't optimize

    assert self.store is not None
    count = await self.store.count(session_id, user_id)
    if count <= self.max_messages:
      return

    strategy = self._resolve_strategy()
    entries = await self.store.get_entries(session_id, user_id)
    optimized = await strategy.optimize(entries, self.model)

    if len(optimized) < len(entries):
      # Embed any new atom entries before storing.
      if self.embedder is not None:
        await self._embed_atoms(optimized)

      # Consolidate atoms (decay, merge, prune).
      await self._consolidate_atoms(optimized)

      # Replace entries with optimized version.
      await self.store.delete_session(session_id, user_id)
      for entry in optimized:
        await self.store.add(entry)
      log_debug(f"Memory optimized: {len(entries)} entries -> {len(optimized)} (session={session_id}, strategy={type(strategy).__name__})")

  async def _embed_atoms(self, entries: List[MemoryEntry]) -> None:
    """Embed atom entries that don't yet have vectors."""
    assert self.embedder is not None
    for entry in entries:
      if entry.entry_type == "atom" and not entry.vector:
        text = entry.lossless_content or entry.content
        if not text:
          continue
        try:
          vector = await self.embedder.async_get_embedding(text)
          if vector:
            entry.vector = vector
        except Exception as exc:
          log_warning(f"Memory: failed to embed atom: {exc}")

  # --- Consolidation ---

  async def _consolidate_atoms(self, entries: List[MemoryEntry]) -> None:
    """Run consolidation on atom entries if a policy is configured."""
    if self.consolidation is None:
      return
    from definable.memory.consolidation import consolidate

    atom_entries = [e for e in entries if e.entry_type == "atom"]
    if atom_entries:
      await consolidate(atom_entries, self.consolidation)

  async def consolidate_session(self, session_id: str, user_id: str = "default") -> int:
    """Manually trigger consolidation on a session's atoms.

    Useful for maintenance or batch jobs outside the normal optimization cycle.

    Returns the number of atoms that were soft-deleted.
    """
    if self.consolidation is None:
      from definable.memory.consolidation import ConsolidationPolicy

      policy = ConsolidationPolicy()
    else:
      policy = self.consolidation

    await self._ensure_initialized()
    assert self.store is not None

    from definable.memory.consolidation import consolidate

    entries = await self.store.get_entries(session_id, user_id)
    atom_entries = [e for e in entries if e.entry_type == "atom"]
    if not atom_entries:
      return 0

    before_active = sum(1 for a in atom_entries if a.superseded_by is None)
    await consolidate(atom_entries, policy)
    after_active = sum(1 for a in atom_entries if a.superseded_by is None)

    # Persist changes.
    for atom in atom_entries:
      await self.store.update(atom)

    deleted = before_active - after_active
    if deleted:
      log_debug(f"Manual consolidation: {deleted} atoms soft-deleted (session={session_id})")
    return deleted

  # --- Lifecycle ---

  async def __aenter__(self) -> "Memory":
    await self._ensure_initialized()
    return self

  async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None:
    await self.close()


def _cosine_similarity(a: List[float], b: List[float]) -> float:
  """Compute cosine similarity between two vectors."""
  dot = sum(x * y for x, y in zip(a, b))
  norm_a = math.sqrt(sum(x * x for x in a))
  norm_b = math.sqrt(sum(x * x for x in b))
  if norm_a == 0.0 or norm_b == 0.0:
    return 0.0
  return dot / (norm_a * norm_b)
