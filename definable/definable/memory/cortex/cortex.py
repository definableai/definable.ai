"""CortexMemory — next-generation memory layer for Definable AI agents.

Duck-types the Memory interface so Agent.py needs zero modifications.
Cortex does more under the hood: multi-representation ingestion,
5-layer retrieval cascade, cascade-aware updates, and behavioral learning.

Quick Start:
    from definable.memory.cortex import CortexMemory

    memory = CortexMemory()
    agent = Agent(model=model, memory=memory)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING, Any, Dict, Optional

from definable.memory.cortex.config import CortexConfig
from definable.memory.cortex.record.scratchpad import Scratchpad
from definable.memory.cortex.record.types import MemoryRecord, MemorySource
from definable.memory.cortex.retrieval.result import RetrievalResult
from definable.memory.cortex.store import CortexStore
from definable.utils.log import log_debug, log_warning

if TYPE_CHECKING:
  from definable.knowledge.embedder.base import Embedder
  from definable.memory.cortex.index.graph import GraphIndex
  from definable.memory.cortex.index.signature import SignatureBuilder, SignatureIndex
  from definable.memory.cortex.index.tags import TagIndex
  from definable.memory.cortex.ingestion.pipeline import IngestionPipeline
  from definable.memory.cortex.learning.inferencer import TraitInferencer
  from definable.memory.cortex.learning.observer import BehavioralObserver
  from definable.memory.cortex.learning.user_model import UserModel
  from definable.memory.cortex.learning.validator import ModelValidator
  from definable.memory.cortex.retrieval.engine import RetrievalEngine
  from definable.memory.cortex.update.cascade import CascadePropagator
  from definable.memory.cortex.update.consolidator import BackgroundConsolidator
  from definable.memory.cortex.update.engine import UpdateEngine
  from definable.model.base import Model
  from definable.model.message import Message
  from definable.tool.function import Function


@dataclass
class CortexMemory:
  """Next-generation memory layer with multi-representation storage and behavioral learning.

  Duck-types the Memory interface (enabled, embedder, model, add, get_entries,
  get_context_messages, search, close, has_semantic_search) so Agent.py needs
  no modifications.

  Also provides a richer native interface:
    - remember(content, source) → str
    - recall(query, top_k) → RetrievalResult
    - update(memory_id, new_content, reason)
    - forget(memory_id, reason)
    - get_state() → Scratchpad
    - set_belief(key, value)
    - as_tools() → list[Function]

  Attributes:
    config: CortexConfig with all tunables.
    model: LLM for slow-path ingestion and observation. None = fast-path only.
    embedder: Embedding model for similarity search. None = no embedding.
    enabled: Whether memory is active.
    description: Description for the agent layer guide.
  """

  config: CortexConfig = field(default_factory=CortexConfig)
  model: Optional["Model"] = None
  embedder: Optional["Embedder"] = None
  enabled: bool = True
  description: Optional[str] = None
  recent_count: int = 20  # Duck-type Memory.recent_count for Agent compatibility

  # Internal state
  _store: Optional[CortexStore] = field(default=None, repr=False)
  _initialized: bool = field(default=False, repr=False)

  # Indexes
  _sig_builder: Optional["SignatureBuilder"] = field(default=None, repr=False)
  _sig_index: Optional["SignatureIndex"] = field(default=None, repr=False)
  _graph_index: Optional["GraphIndex"] = field(default=None, repr=False)
  _tag_index: Optional["TagIndex"] = field(default=None, repr=False)

  # Engines
  _ingestion: Optional["IngestionPipeline"] = field(default=None, repr=False)
  _retrieval: Optional["RetrievalEngine"] = field(default=None, repr=False)
  _update_engine: Optional["UpdateEngine"] = field(default=None, repr=False)
  _cascade: Optional["CascadePropagator"] = field(default=None, repr=False)
  _consolidator: Optional["BackgroundConsolidator"] = field(default=None, repr=False)

  # Learning
  _observer: Optional["BehavioralObserver"] = field(default=None, repr=False)
  _inferencer: Optional["TraitInferencer"] = field(default=None, repr=False)
  _user_models: Dict[str, "UserModel"] = field(default_factory=dict, repr=False)
  _validator: Optional["ModelValidator"] = field(default=None, repr=False)

  # Shared db connection for indexes
  _index_db: Any = field(default=None, repr=False)

  async def _ensure_initialized(self) -> None:
    """Lazy-initialize all subsystems on first use."""
    if self._initialized:
      return

    # Store
    self._store = CortexStore(db_path=self.config.db_path)
    await self._store.initialize()

    # Shared index db (reuse the store's db for simplicity)
    import aiosqlite

    db_path = self._store.db_path
    self._index_db = await aiosqlite.connect(db_path)

    # Signature index
    if self.config.enable_signatures:
      from definable.memory.cortex.index.signature import SignatureBuilder, SignatureIndex

      self._sig_builder = SignatureBuilder(dims=self.config.signature_dims, nnz=self.config.signature_nnz)
      self._sig_index = SignatureIndex()
      await self._sig_index.initialize(self._index_db)

    # Graph index
    if self.config.enable_graph:
      from definable.memory.cortex.index.graph import GraphIndex

      self._graph_index = GraphIndex()
      await self._graph_index.initialize(self._index_db)

    # Tag index
    if self.config.enable_tags:
      from definable.memory.cortex.index.tags import TagIndex

      self._tag_index = TagIndex(separator=self.config.tag_separator)
      await self._tag_index.initialize(self._index_db)

    # Cascade propagator
    if self.config.enable_graph and self._graph_index:
      from definable.memory.cortex.update.cascade import CascadePropagator

      self._cascade = CascadePropagator(store=self._store, graph_index=self._graph_index, config=self.config)

    # Ingestion pipeline
    from definable.memory.cortex.ingestion.pipeline import IngestionPipeline

    self._ingestion = IngestionPipeline(
      store=self._store,
      config=self.config,
      model=self.model,
      embedder=self.embedder,
      signature_builder=self._sig_builder,
      signature_index=self._sig_index,
      graph_index=self._graph_index,
      tag_index=self._tag_index,
    )

    # Retrieval engine
    from definable.memory.cortex.retrieval.engine import RetrievalEngine

    self._retrieval = RetrievalEngine(
      store=self._store,
      config=self.config,
      embedder=self.embedder,
      signature_builder=self._sig_builder,
      signature_index=self._sig_index,
      graph_index=self._graph_index,
      tag_index=self._tag_index,
    )

    # Update engine
    from definable.memory.cortex.update.engine import UpdateEngine

    self._update_engine = UpdateEngine(
      store=self._store,
      config=self.config,
      graph_index=self._graph_index,
      tag_index=self._tag_index,
      cascade=self._cascade,
    )

    # Learning subsystem
    if self.config.enable_learning:
      from definable.memory.cortex.learning.inferencer import TraitInferencer
      from definable.memory.cortex.learning.observer import BehavioralObserver
      from definable.memory.cortex.learning.validator import ModelValidator

      self._observer = BehavioralObserver(model=self.model)
      self._inferencer = TraitInferencer(config=self.config)
      self._validator = ModelValidator()

    # Background consolidation
    if self.config.enable_consolidation:
      from definable.memory.cortex.update.consolidator import BackgroundConsolidator

      self._consolidator = BackgroundConsolidator(store=self._store, config=self.config, embedder=self.embedder)
      self._consolidator.start()

    self._initialized = True
    log_debug("CortexMemory initialized", log_level=2)

  # ================================================================
  # Memory-compatible interface (Agent calls these)
  # ================================================================

  @property
  def has_semantic_search(self) -> bool:
    """Always True — Cortex supports semantic search natively."""
    return True

  async def add(self, message: "Message", session_id: str = "default", user_id: str = "default") -> None:
    """Add a message to Cortex memory. Memory-compatible interface."""
    if not self.enabled:
      return
    await self._ensure_initialized()

    content = message.content if isinstance(message.content, str) else str(message.content or "")
    role = getattr(message, "role", "user")

    # Ingest through the pipeline
    assert self._ingestion is not None
    record = await self._ingestion.ingest(
      content=content,
      role=role,
      source=MemorySource.CONVERSATION,
      session_id=session_id,
      user_id=user_id,
    )

    # Fire learning observer in background
    if self._observer and self._inferencer and role == "user":
      asyncio.create_task(self._observe_and_learn(content, role, record.record_id, user_id))

  async def get_entries(self, session_id: str, user_id: str = "default", limit: Optional[int] = None) -> list:
    """Get raw entries for a session. Memory-compatible interface."""
    await self._ensure_initialized()
    assert self._store is not None
    records = await self._store.get_records(session_id, user_id, active_only=True, limit=limit)
    # Convert to MemoryEntry-like dicts for compatibility
    from definable.memory.types import MemoryEntry

    entries = []
    for r in records:
      entries.append(
        MemoryEntry(
          session_id=r.session_id,
          memory_id=r.record_id,
          user_id=r.user_id,
          role=r.role,
          content=r.raw_content,
          created_at=r.created_at,
          updated_at=r.updated_at,
          entry_type="message" if r.source == MemorySource.CONVERSATION else "atom",
        )
      )
    return entries

  async def get_context_messages(self, session_id: str, user_id: str = "default") -> list:
    """Get entries as Message objects. Memory-compatible interface."""
    from definable.model.message import Message

    entries = await self.get_entries(session_id, user_id)
    messages = []
    for entry in entries:
      messages.append(Message(role=entry.role, content=entry.content))
    return messages

  async def search(
    self,
    query: str,
    session_id: str,
    user_id: str = "default",
    top_k: Optional[int] = None,
  ) -> list:
    """Similarity-ranked retrieval. Memory-compatible interface."""
    result = await self.recall(query, top_k=top_k, session_id=session_id, user_id=user_id)
    # Convert to MemoryEntry list for compatibility
    from definable.memory.types import MemoryEntry

    entries = []
    for sm in result.memories:
      r = sm.record
      entries.append(
        MemoryEntry(
          session_id=r.session_id,
          memory_id=r.record_id,
          user_id=r.user_id,
          role=r.role,
          content=r.raw_content,
          created_at=r.created_at,
          updated_at=r.updated_at,
        )
      )
    return entries

  async def close(self) -> None:
    """Shutdown all subsystems."""
    if self._consolidator:
      await self._consolidator.stop()
    if self._ingestion:
      await self._ingestion.wait_for_background()
    if self._index_db:
      await self._index_db.close()
      self._index_db = None
    if self._store:
      await self._store.close()
    self._initialized = False

  # ================================================================
  # Cortex-native interface (richer)
  # ================================================================

  async def remember(
    self,
    content: str,
    source: MemorySource = MemorySource.CONVERSATION,
    session_id: str = "default",
    user_id: str = "default",
    role: str = "user",
  ) -> str:
    """Store a new memory. Returns the record ID."""
    await self._ensure_initialized()
    assert self._ingestion is not None
    record = await self._ingestion.ingest(
      content=content,
      role=role,
      source=source,
      session_id=session_id,
      user_id=user_id,
    )
    return record.record_id

  async def recall(
    self,
    query: str,
    top_k: Optional[int] = None,
    session_id: str = "default",
    user_id: str = "default",
  ) -> RetrievalResult:
    """Recall memories using the 5-layer cascade."""
    await self._ensure_initialized()
    assert self._retrieval is not None
    return await self._retrieval.recall(query, session_id=session_id, user_id=user_id, top_k=top_k)

  async def update(self, memory_id: str, new_content: str, reason: str = "") -> Optional[MemoryRecord]:
    """Update a memory's content (soft-update: old is superseded, new is created)."""
    await self._ensure_initialized()
    assert self._update_engine is not None
    return await self._update_engine.update_content(memory_id, new_content, reason)

  async def forget(self, memory_id: str, reason: str = "") -> bool:
    """Soft-delete a memory."""
    await self._ensure_initialized()
    assert self._update_engine is not None
    return await self._update_engine.forget(memory_id, reason)

  async def get_state(self, session_id: str = "default", user_id: str = "default") -> Scratchpad:
    """Get the current scratchpad state."""
    await self._ensure_initialized()
    assert self._store is not None
    return await self._store.get_scratchpad(session_id, user_id)

  async def set_belief(self, key: str, value: Any, session_id: str = "default", user_id: str = "default") -> None:
    """Set a scratchpad belief."""
    await self._ensure_initialized()
    assert self._update_engine is not None
    await self._update_engine.set_belief(key, value, session_id, user_id)

  async def get_user_model(self, user_id: str = "default") -> "UserModel":
    """Get or create the user model for a user."""
    await self._ensure_initialized()
    if user_id not in self._user_models:
      from definable.memory.cortex.learning.user_model import UserModel

      self._user_models[user_id] = UserModel(user_id=user_id)
    return self._user_models[user_id]

  def as_tools(self) -> list["Function"]:
    """Return Cortex operations as agent tools.

    Allows the agent to actively use memory:
      - cortex_remember: Store a memory
      - cortex_recall: Search memory
      - cortex_set_belief: Update scratchpad
      - cortex_forget: Remove a memory
    """
    from definable.tool.decorator import tool

    cortex = self

    @tool
    async def cortex_remember(content: str, source: str = "conversation") -> str:
      """Store a new memory in Cortex. Returns the memory ID."""
      mem_source = MemorySource(source) if source in [s.value for s in MemorySource] else MemorySource.CONVERSATION
      record_id = await cortex.remember(content, source=mem_source)
      return f"Stored memory: {record_id[:8]}"

    @tool
    async def cortex_recall(query: str, top_k: int = 5) -> str:
      """Search Cortex memory for relevant information."""
      result = await cortex.recall(query, top_k=top_k)
      if not result.memories:
        return "No relevant memories found."
      parts = []
      for sm in result.memories:
        parts.append(f"[{sm.score:.2f}] {sm.record.raw_content[:200]}")
      return "\n".join(parts)

    @tool
    async def cortex_set_belief(key: str, value: str) -> str:
      """Set a belief in the Cortex scratchpad."""
      await cortex.set_belief(key, value)
      return f"Belief '{key}' set."

    @tool
    async def cortex_forget(memory_id: str, reason: str = "") -> str:
      """Forget (soft-delete) a memory from Cortex."""
      success = await cortex.forget(memory_id, reason)
      return f"Memory {'forgotten' if success else 'not found'}."

    return [cortex_remember, cortex_recall, cortex_set_belief, cortex_forget]

  # ================================================================
  # Internal helpers
  # ================================================================

  async def _observe_and_learn(self, text: str, role: str, record_id: str, user_id: str) -> None:
    """Background task: observe interaction and update user model."""
    try:
      assert self._observer is not None
      assert self._inferencer is not None
      observations = await self._observer.observe(text, role=role, record_id=record_id)
      if observations:
        model = await self.get_user_model(user_id)
        self._inferencer.process(observations, model)
        self._inferencer.check_contradictions(observations, model)
    except Exception as exc:
      log_warning(f"Cortex learning error: {exc}")

  # ================================================================
  # Lifecycle
  # ================================================================

  async def __aenter__(self) -> "CortexMemory":
    await self._ensure_initialized()
    return self

  async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None:
    await self.close()
