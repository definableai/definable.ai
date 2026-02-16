"""Mem0-backed memory store."""

import json
import os
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from definable.memory.types import Episode, KnowledgeAtom, Procedure, TopicTransition
from definable.utils.log import log_debug


class Mem0MemoryStore:
  """Async Mem0 memory store using the hosted Mem0 API.

  Uses ``AsyncMemoryClient`` with ``infer=False`` so Mem0 stores content
  verbatim (no AI extraction/merging).  This makes Mem0 behave as a
  structured metadata store, matching the protocol's CRUD semantics.

  **Known limitation**: ``search_episodes_by_embedding()`` and
  ``search_atoms_by_embedding()`` return empty lists.  Mem0 does not
  accept raw embedding vectors — it only supports text-query semantic
  search.  The retrieval pipeline has 5 parallel paths and handles
  empty results gracefully.
  """

  # System-level user_id for data without a real user.
  # Mem0's hosted API silently drops data stored with ``agent_id`` when
  # ``infer=False``, so we use a sentinel user_id instead.
  _SYSTEM_USER_ID = "__definable_system__"

  def __init__(
    self,
    api_key: str = "",
    org_id: Optional[str] = None,
    project_id: Optional[str] = None,
  ):
    self._api_key = api_key or os.environ.get("MEM0_API_KEY", "")
    self._org_id = org_id
    self._project_id = project_id
    self._client: Any = None
    self._initialized = False
    # definable_id → mem0 memory ID
    self._id_cache: Dict[str, str] = {}
    # definable_id → full Mem0 memory dict (for merge-update)
    self._memory_cache: Dict[str, dict] = {}
    # Track user_ids seen during this session (for cross-user queries)
    self._known_user_ids: set = set()
    # Track deleted user_ids to short-circuit reads during eventual consistency
    self._deleted_user_ids: set = set()

  async def initialize(self) -> None:
    if self._initialized:
      return

    try:
      from mem0 import AsyncMemoryClient
    except ImportError as e:
      raise ImportError("mem0ai is required for Mem0MemoryStore. Install it with: pip install definable[mem0-memory]") from e

    if not self._api_key:
      raise ValueError("Mem0 API key is required. Set api_key or MEM0_API_KEY environment variable.")

    kwargs: Dict[str, Any] = {"api_key": self._api_key}
    if self._org_id:
      kwargs["org_id"] = self._org_id
    if self._project_id:
      kwargs["project_id"] = self._project_id

    self._client = AsyncMemoryClient(**kwargs)
    self._initialized = True
    log_debug("Mem0MemoryStore initialized", log_level=2)

  async def close(self) -> None:
    self._client = None
    self._initialized = False
    self._id_cache.clear()
    self._memory_cache.clear()
    self._known_user_ids.clear()
    self._deleted_user_ids.clear()

  async def _ensure_initialized(self) -> None:
    if not self._initialized:
      await self.initialize()

  # ------------------------------------------------------------------
  # Cache helpers
  # ------------------------------------------------------------------

  def _cache_memory(self, memory: dict) -> None:
    """Cache a Mem0 memory for later lookup by definable_id."""
    metadata = memory.get("metadata") or {}
    definable_id = metadata.get("definable_id")
    if definable_id:
      mem0_id = memory.get("id", "")
      if mem0_id:
        self._id_cache[definable_id] = mem0_id
        self._memory_cache[definable_id] = memory

  # ------------------------------------------------------------------
  # Metadata ↔ Episode
  # ------------------------------------------------------------------

  def _episode_to_metadata(self, episode: Episode) -> dict:
    return {
      "record_type": "episode",
      "definable_id": episode.id,
      "user_id": episode.user_id or "",
      "session_id": episode.session_id,
      "role": episode.role,
      "topics": json.dumps(episode.topics),
      "sentiment": episode.sentiment,
      "token_count": episode.token_count,
      "compression_stage": episode.compression_stage,
      "created_at": episode.created_at,
      "last_accessed_at": episode.last_accessed_at,
      "access_count": episode.access_count,
      "_has_embedding": episode.embedding is not None,
    }

  def _metadata_to_episode(self, memory: dict) -> Episode:
    metadata = memory.get("metadata") or {}
    topics_raw = metadata.get("topics", "[]")
    topics = json.loads(topics_raw) if isinstance(topics_raw, str) else topics_raw
    return Episode(
      id=metadata.get("definable_id", ""),
      user_id=metadata.get("user_id") or None,
      session_id=metadata.get("session_id", ""),
      role=metadata.get("role", ""),
      content=memory.get("memory", ""),
      embedding=None,
      topics=topics,
      sentiment=float(metadata.get("sentiment", 0.0)),
      token_count=int(metadata.get("token_count", 0)),
      compression_stage=int(metadata.get("compression_stage", 0)),
      created_at=float(metadata.get("created_at", 0.0)),
      last_accessed_at=float(metadata.get("last_accessed_at", 0.0)),
      access_count=int(metadata.get("access_count", 0)),
    )

  # ------------------------------------------------------------------
  # Metadata ↔ KnowledgeAtom
  # ------------------------------------------------------------------

  def _atom_to_metadata(self, atom: KnowledgeAtom) -> dict:
    return {
      "record_type": "atom",
      "definable_id": atom.id,
      "user_id": atom.user_id or "",
      "subject": atom.subject,
      "subject_lower": atom.subject.lower(),
      "predicate": atom.predicate,
      "predicate_lower": atom.predicate.lower(),
      "object": atom.object,
      "confidence": atom.confidence,
      "reinforcement_count": atom.reinforcement_count,
      "topics": json.dumps(atom.topics),
      "token_count": atom.token_count,
      "source_episode_ids": json.dumps(atom.source_episode_ids),
      "created_at": atom.created_at,
      "last_accessed_at": atom.last_accessed_at,
      "last_reinforced_at": atom.last_reinforced_at,
      "access_count": atom.access_count,
      "_has_embedding": atom.embedding is not None,
    }

  def _metadata_to_atom(self, memory: dict) -> KnowledgeAtom:
    metadata = memory.get("metadata") or {}
    topics_raw = metadata.get("topics", "[]")
    topics = json.loads(topics_raw) if isinstance(topics_raw, str) else topics_raw
    source_raw = metadata.get("source_episode_ids", "[]")
    source_ids = json.loads(source_raw) if isinstance(source_raw, str) else source_raw
    return KnowledgeAtom(
      id=metadata.get("definable_id", ""),
      user_id=metadata.get("user_id") or None,
      subject=metadata.get("subject", ""),
      predicate=metadata.get("predicate", ""),
      object=metadata.get("object", ""),
      content=memory.get("memory", ""),
      embedding=None,
      confidence=float(metadata.get("confidence", 1.0)),
      reinforcement_count=int(metadata.get("reinforcement_count", 0)),
      topics=topics,
      token_count=int(metadata.get("token_count", 0)),
      source_episode_ids=source_ids,
      created_at=float(metadata.get("created_at", 0.0)),
      last_accessed_at=float(metadata.get("last_accessed_at", 0.0)),
      last_reinforced_at=float(metadata.get("last_reinforced_at", 0.0)),
      access_count=int(metadata.get("access_count", 0)),
    )

  # ------------------------------------------------------------------
  # Metadata ↔ Procedure
  # ------------------------------------------------------------------

  def _procedure_to_metadata(self, procedure: Procedure) -> dict:
    return {
      "record_type": "procedure",
      "definable_id": procedure.id,
      "user_id": procedure.user_id or "",
      "trigger": procedure.trigger,
      "action": procedure.action,
      "confidence": procedure.confidence,
      "observation_count": procedure.observation_count,
      "created_at": procedure.created_at,
      "last_accessed_at": procedure.last_accessed_at,
    }

  def _metadata_to_procedure(self, memory: dict) -> Procedure:
    metadata = memory.get("metadata") or {}
    return Procedure(
      id=metadata.get("definable_id", ""),
      user_id=metadata.get("user_id") or None,
      trigger=metadata.get("trigger", ""),
      action=metadata.get("action", ""),
      confidence=float(metadata.get("confidence", 0.5)),
      observation_count=int(metadata.get("observation_count", 1)),
      created_at=float(metadata.get("created_at", 0.0)),
      last_accessed_at=float(metadata.get("last_accessed_at", 0.0)),
    )

  # ------------------------------------------------------------------
  # Private helpers
  # ------------------------------------------------------------------

  async def _add_record(
    self,
    text: str,
    metadata: dict,
    user_id: Optional[str],
  ) -> str:
    """Add a record to Mem0 with ``infer=False``.  Returns the Mem0 memory ID."""
    kwargs: Dict[str, Any] = {
      "metadata": metadata,
      "infer": False,
      "async_mode": False,  # Ensure immediate consistency for subsequent reads
    }
    if user_id:
      kwargs["user_id"] = user_id
      self._known_user_ids.add(user_id)
    else:
      kwargs["user_id"] = self._SYSTEM_USER_ID

    result = await self._client.add(
      messages=[{"role": "user", "content": text}],
      **kwargs,
    )

    # Extract Mem0 memory ID from response (handles both list and dict forms)
    events: list = []
    if isinstance(result, dict):
      events = result.get("results", [])
    elif isinstance(result, list):
      events = result

    mem0_id = ""
    if events:
      mem0_id = events[0].get("id", "")

    # Cache mapping
    definable_id = metadata.get("definable_id", "")
    if definable_id and mem0_id:
      self._id_cache[definable_id] = mem0_id
      self._memory_cache[definable_id] = {
        "id": mem0_id,
        "memory": text,
        "metadata": metadata,
      }

    return mem0_id

  def _matches_filters(self, metadata: dict, filters: Dict[str, Any]) -> bool:
    """Check if *metadata* satisfies every condition in *filters*."""
    for key, value in filters.items():
      if isinstance(value, dict):
        mem_val = metadata.get(key)
        if mem_val is None:
          return False
        for op, threshold in value.items():
          mem_float = float(mem_val)
          if op == "$eq" and mem_float != threshold:
            return False
          elif op == "$ne" and mem_float == threshold:
            return False
          elif op == "$gte" and mem_float < threshold:
            return False
          elif op == "$lte" and mem_float > threshold:
            return False
          elif op == "$gt" and mem_float <= threshold:
            return False
          elif op == "$lt" and mem_float >= threshold:
            return False
      else:
        if metadata.get(key) != value:
          return False
    return True

  async def _get_all_by_type(
    self,
    record_type: str,
    user_id: Optional[str],
    extra_filters: Optional[Dict[str, Any]] = None,
    limit: int = 10000,
    global_search: bool = False,
  ) -> List[dict]:
    """Fetch all memories of *record_type*, scoped by user.

    When *user_id* is ``None`` and *global_search* is ``False`` the
    request is scoped with the system user_id (shared scope).
    When *global_search* is ``True`` the method queries the system scope
    **plus** every user scope seen so far, covering cross-user
    operations like distillation and pruning.

    After the API fetch, locally-cached entries that were not returned
    (read-after-write consistency gap) are merged in so that recent
    writes are always visible.
    """
    # Build the list of API scopes to query, then exclude any that
    # were deleted during this session (Mem0 eventual consistency means
    # the API may still return stale data for recently-deleted users).
    if global_search:
      scopes: List[Dict[str, str]] = [{"user_id": self._SYSTEM_USER_ID}]
      for uid in sorted(self._known_user_ids):
        scopes.append({"user_id": uid})
    elif user_id is not None:
      scopes = [{"user_id": user_id}]
    else:
      scopes = [{"user_id": self._SYSTEM_USER_ID}]
    scopes = [s for s in scopes if s.get("user_id") not in self._deleted_user_ids]

    all_memories: List[dict] = []
    seen_mem0_ids: set = set()

    # --- Phase 1: fetch from API across all scopes ---
    for scope in scopes:
      if len(all_memories) >= limit:
        break
      page = 1
      page_size = 100
      while True:
        kwargs: Dict[str, Any] = {"filters": scope, "page": page, "page_size": page_size}
        result = await self._client.get_all(**kwargs)

        memories: list = []
        if isinstance(result, dict):
          memories = result.get("results", [])
        elif isinstance(result, list):
          memories = result

        if not memories:
          break

        for mem in memories:
          mem0_id = mem.get("id", "")

          # Prefer locally-cached version over API result — the cache
          # may contain updates (e.g. incremented counts) not yet
          # reflected by the API due to indexing lag.
          api_md = mem.get("metadata") or {}
          did = api_md.get("definable_id")
          if did and did in self._memory_cache:
            cached = self._memory_cache[did]
            if cached.get("id") == mem0_id:
              mem = cached
          else:
            self._cache_memory(mem)

          if mem0_id in seen_mem0_ids:
            continue
          seen_mem0_ids.add(mem0_id)

          md = mem.get("metadata") or {}
          if md.get("record_type") != record_type:
            continue
          if extra_filters and not self._matches_filters(md, extra_filters):
            continue

          all_memories.append(mem)
          if len(all_memories) >= limit:
            break

        if len(all_memories) >= limit or len(memories) < page_size:
          break
        page += 1

    # --- Phase 2: supplement with locally-cached entries ---
    # Handles read-after-write consistency gaps (e.g. topic transition
    # increments where the API hasn't indexed the latest write yet).
    for cached_mem in list(self._memory_cache.values()):
      if len(all_memories) >= limit:
        break
      mem0_id = cached_mem.get("id", "")
      if mem0_id in seen_mem0_ids:
        continue
      md = cached_mem.get("metadata") or {}
      if md.get("record_type") != record_type:
        continue
      # Scope check
      if not global_search:
        cached_uid = md.get("user_id") or ""
        if user_id is not None:
          if cached_uid != user_id:
            continue
        else:
          if cached_uid:
            continue  # user-scoped entry, skip for system scope
      if extra_filters and not self._matches_filters(md, extra_filters):
        continue
      seen_mem0_ids.add(mem0_id)
      all_memories.append(cached_mem)

    return all_memories

  async def _find_memory(self, definable_id: str, record_type: str) -> Optional[dict]:
    """Look up a Mem0 memory by *definable_id*.  Checks cache first."""
    if definable_id in self._memory_cache:
      return self._memory_cache[definable_id]

    # Cache miss — the memory must exist under *some* user scope that we
    # fetched earlier but the cache was cleared, or the store was
    # re-instantiated.  Try the global scope as a fallback.
    memories = await self._get_all_by_type(
      record_type,
      user_id=None,
      extra_filters={"definable_id": definable_id},
      limit=1,
    )
    if memories:
      return memories[0]

    log_debug(
      f"Mem0MemoryStore: could not locate memory for definable_id={definable_id}",
      log_level=2,
    )
    return None

  # ------------------------------------------------------------------
  # Episodes
  # ------------------------------------------------------------------

  async def store_episode(self, episode: Episode) -> str:
    await self._ensure_initialized()
    if not episode.id:
      episode.id = str(uuid4())
    now = time.time()
    if episode.created_at == 0.0:
      episode.created_at = now
    if episode.last_accessed_at == 0.0:
      episode.last_accessed_at = now

    metadata = self._episode_to_metadata(episode)
    await self._add_record(episode.content, metadata, episode.user_id)
    return episode.id

  async def get_episodes(
    self,
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = 50,
    min_stage: Optional[int] = None,
    max_stage: Optional[int] = None,
  ) -> List[Episode]:
    await self._ensure_initialized()

    extra_filters: Dict[str, Any] = {}
    if session_id is not None:
      extra_filters["session_id"] = session_id
    if min_stage is not None:
      extra_filters.setdefault("compression_stage", {})
      extra_filters["compression_stage"]["$gte"] = min_stage
    if max_stage is not None:
      extra_filters.setdefault("compression_stage", {})
      extra_filters["compression_stage"]["$lte"] = max_stage

    memories = await self._get_all_by_type(
      "episode",
      user_id=user_id,
      extra_filters=extra_filters or None,
    )

    episodes = [self._metadata_to_episode(mem) for mem in memories]
    episodes.sort(key=lambda e: e.created_at, reverse=True)
    return episodes[:limit]

  async def update_episode(self, episode_id: str, **fields: Any) -> None:
    await self._ensure_initialized()
    if not fields:
      return

    memory = await self._find_memory(episode_id, "episode")
    if not memory:
      return

    mem0_id = memory.get("id")
    if not mem0_id:
      return

    current_metadata = dict(memory.get("metadata") or {})
    text = memory.get("memory", "")

    for key, value in fields.items():
      if key == "embedding":
        current_metadata["_has_embedding"] = value is not None
      elif key in ("topics", "source_episode_ids"):
        current_metadata[key] = json.dumps(value if value is not None else [])
      elif key == "content":
        text = value
      else:
        current_metadata[key] = value

    await self._client.update(mem0_id, text=text, metadata=current_metadata)

    # Refresh cache
    memory["memory"] = text
    memory["metadata"] = current_metadata
    self._memory_cache[episode_id] = memory

  async def get_episodes_for_distillation(self, stage: int, older_than: float) -> List[Episode]:
    await self._ensure_initialized()

    extra_filters: Dict[str, Any] = {
      "compression_stage": {"$eq": stage},
      "created_at": {"$lt": older_than},
    }

    # Distillation is cross-user, so we search global scope.
    memories = await self._get_all_by_type(
      "episode",
      user_id=None,
      extra_filters=extra_filters,
      global_search=True,
    )

    episodes = [self._metadata_to_episode(mem) for mem in memories]
    episodes.sort(key=lambda e: e.created_at)
    return episodes[:50]

  # ------------------------------------------------------------------
  # Knowledge Atoms
  # ------------------------------------------------------------------

  async def store_atom(self, atom: KnowledgeAtom) -> str:
    await self._ensure_initialized()
    if not atom.id:
      atom.id = str(uuid4())
    now = time.time()
    if atom.created_at == 0.0:
      atom.created_at = now
    if atom.last_accessed_at == 0.0:
      atom.last_accessed_at = now

    metadata = self._atom_to_metadata(atom)
    await self._add_record(atom.content, metadata, atom.user_id)
    return atom.id

  async def get_atoms(
    self,
    *,
    user_id: Optional[str] = None,
    min_confidence: float = 0.1,
    limit: int = 50,
  ) -> List[KnowledgeAtom]:
    await self._ensure_initialized()

    extra_filters: Dict[str, Any] = {
      "confidence": {"$gte": min_confidence},
    }

    memories = await self._get_all_by_type(
      "atom",
      user_id=user_id,
      extra_filters=extra_filters,
    )

    atoms = [self._metadata_to_atom(mem) for mem in memories]
    atoms.sort(key=lambda a: a.last_accessed_at, reverse=True)
    return atoms[:limit]

  async def find_similar_atom(
    self,
    subject: str,
    predicate: str,
    user_id: Optional[str] = None,
  ) -> Optional[KnowledgeAtom]:
    await self._ensure_initialized()

    extra_filters: Dict[str, Any] = {
      "subject_lower": subject.lower(),
      "predicate_lower": predicate.lower(),
    }

    memories = await self._get_all_by_type(
      "atom",
      user_id=user_id,
      extra_filters=extra_filters,
      limit=1,
    )

    if memories:
      return self._metadata_to_atom(memories[0])
    return None

  async def update_atom(self, atom_id: str, **fields: Any) -> None:
    await self._ensure_initialized()
    if not fields:
      return

    memory = await self._find_memory(atom_id, "atom")
    if not memory:
      return

    mem0_id = memory.get("id")
    if not mem0_id:
      return

    current_metadata = dict(memory.get("metadata") or {})
    text = memory.get("memory", "")

    for key, value in fields.items():
      if key == "embedding":
        current_metadata["_has_embedding"] = value is not None
      elif key in ("topics", "source_episode_ids"):
        current_metadata[key] = json.dumps(value if value is not None else [])
      elif key == "subject":
        current_metadata["subject"] = value
        current_metadata["subject_lower"] = value.lower() if isinstance(value, str) else value
      elif key == "predicate":
        current_metadata["predicate"] = value
        current_metadata["predicate_lower"] = value.lower() if isinstance(value, str) else value
      elif key == "content":
        text = value
      else:
        current_metadata[key] = value

    await self._client.update(mem0_id, text=text, metadata=current_metadata)

    memory["memory"] = text
    memory["metadata"] = current_metadata
    self._memory_cache[atom_id] = memory

  async def prune_atoms(self, min_confidence: float) -> int:
    await self._ensure_initialized()

    extra_filters: Dict[str, Any] = {
      "confidence": {"$lt": min_confidence},
    }

    # Prune across global scope (distillation context).
    memories = await self._get_all_by_type(
      "atom",
      user_id=None,
      extra_filters=extra_filters,
      global_search=True,
    )

    count = len(memories)
    for mem in memories:
      mem0_id = mem.get("id")
      if mem0_id:
        await self._client.delete(mem0_id)
        # Evict from cache
        md = mem.get("metadata") or {}
        definable_id = md.get("definable_id")
        if definable_id:
          self._id_cache.pop(definable_id, None)
          self._memory_cache.pop(definable_id, None)

    return count

  # ------------------------------------------------------------------
  # Procedures
  # ------------------------------------------------------------------

  async def store_procedure(self, procedure: Procedure) -> str:
    await self._ensure_initialized()
    if not procedure.id:
      procedure.id = str(uuid4())
    now = time.time()
    if procedure.created_at == 0.0:
      procedure.created_at = now
    if procedure.last_accessed_at == 0.0:
      procedure.last_accessed_at = now

    metadata = self._procedure_to_metadata(procedure)
    text = f"{procedure.trigger} -> {procedure.action}"
    await self._add_record(text, metadata, procedure.user_id)
    return procedure.id

  async def get_procedures(
    self,
    *,
    user_id: Optional[str] = None,
    min_confidence: float = 0.3,
  ) -> List[Procedure]:
    await self._ensure_initialized()

    extra_filters: Dict[str, Any] = {
      "confidence": {"$gte": min_confidence},
    }

    memories = await self._get_all_by_type(
      "procedure",
      user_id=user_id,
      extra_filters=extra_filters,
    )

    procedures = [self._metadata_to_procedure(mem) for mem in memories]
    procedures.sort(key=lambda p: p.confidence, reverse=True)
    return procedures

  async def find_similar_procedure(
    self,
    trigger: str,
    user_id: Optional[str] = None,
  ) -> Optional[Procedure]:
    await self._ensure_initialized()

    trigger_words = set(trigger.lower().split())
    procedures = await self.get_procedures(user_id=user_id, min_confidence=0.1)

    best_match: Optional[Procedure] = None
    best_overlap = 0

    for proc in procedures:
      proc_words = set(proc.trigger.lower().split())
      overlap = len(trigger_words & proc_words)
      if overlap > best_overlap:
        best_overlap = overlap
        best_match = proc

    return best_match if best_overlap > 0 else None

  async def update_procedure(self, procedure_id: str, **fields: Any) -> None:
    await self._ensure_initialized()
    if not fields:
      return

    memory = await self._find_memory(procedure_id, "procedure")
    if not memory:
      return

    mem0_id = memory.get("id")
    if not mem0_id:
      return

    current_metadata = dict(memory.get("metadata") or {})

    for key, value in fields.items():
      current_metadata[key] = value

    # Rebuild text from (possibly updated) trigger/action
    trigger = current_metadata.get("trigger", "")
    action = current_metadata.get("action", "")
    text = f"{trigger} -> {action}"

    await self._client.update(mem0_id, text=text, metadata=current_metadata)

    memory["memory"] = text
    memory["metadata"] = current_metadata
    self._memory_cache[procedure_id] = memory

  # ------------------------------------------------------------------
  # Topics
  # ------------------------------------------------------------------

  async def store_topic_transition(
    self,
    from_topic: str,
    to_topic: str,
    user_id: Optional[str] = None,
  ) -> None:
    await self._ensure_initialized()

    uid_value = user_id or ""
    extra_filters: Dict[str, Any] = {
      "from_topic": from_topic,
      "to_topic": to_topic,
      "user_id": uid_value,
    }

    memories = await self._get_all_by_type(
      "transition",
      user_id=user_id,
      extra_filters=extra_filters,
      limit=1,
    )

    if memories:
      # Increment count
      existing = memories[0]
      mem0_id = existing.get("id")
      if mem0_id:
        md = dict(existing.get("metadata") or {})
        md["count"] = int(md.get("count", 0)) + 1
        text = f"{from_topic} -> {to_topic}"
        await self._client.update(mem0_id, text=text, metadata=md)
        # Refresh cache
        existing["metadata"] = md
        definable_id = md.get("definable_id")
        if definable_id:
          self._memory_cache[definable_id] = existing
    else:
      # Insert new transition
      point_id = str(uuid4())
      metadata: dict = {
        "record_type": "transition",
        "definable_id": point_id,
        "user_id": uid_value,
        "from_topic": from_topic,
        "to_topic": to_topic,
        "count": 1,
      }
      text = f"{from_topic} -> {to_topic}"
      await self._add_record(text, metadata, user_id)

  async def get_topic_transitions(
    self,
    from_topic: str,
    user_id: Optional[str] = None,
    min_count: int = 3,
  ) -> List[TopicTransition]:
    await self._ensure_initialized()

    extra_filters: Dict[str, Any] = {
      "from_topic": from_topic,
      "count": {"$gte": min_count},
    }
    if user_id is not None:
      extra_filters["user_id"] = user_id

    memories = await self._get_all_by_type(
      "transition",
      user_id=user_id,
      extra_filters=extra_filters,
    )

    raw = []
    for mem in memories:
      md = mem.get("metadata") or {}
      raw.append((
        md.get("from_topic", ""),
        md.get("to_topic", ""),
        int(md.get("count", 0)),
      ))

    raw.sort(key=lambda x: x[2], reverse=True)

    total = sum(r[2] for r in raw)
    transitions = []
    for from_t, to_t, count in raw:
      prob = count / total if total > 0 else 0.0
      transitions.append(
        TopicTransition(
          from_topic=from_t,
          to_topic=to_t,
          count=count,
          probability=prob,
        )
      )
    return transitions

  # ------------------------------------------------------------------
  # Vector search (not supported — Mem0 uses text-query search only)
  # ------------------------------------------------------------------

  async def search_episodes_by_embedding(
    self,
    embedding: List[float],
    *,
    user_id: Optional[str] = None,
    top_k: int = 20,
  ) -> List[Episode]:
    await self._ensure_initialized()
    log_debug(
      "Mem0MemoryStore: search_episodes_by_embedding returns [] (Mem0 does not accept raw embedding vectors)",
      log_level=2,
    )
    return []

  async def search_atoms_by_embedding(
    self,
    embedding: List[float],
    *,
    user_id: Optional[str] = None,
    top_k: int = 20,
  ) -> List[KnowledgeAtom]:
    await self._ensure_initialized()
    log_debug(
      "Mem0MemoryStore: search_atoms_by_embedding returns [] (Mem0 does not accept raw embedding vectors)",
      log_level=2,
    )
    return []

  # ------------------------------------------------------------------
  # Deletion
  # ------------------------------------------------------------------

  async def delete_user_data(self, user_id: str) -> None:
    await self._ensure_initialized()
    await self._client.delete_users(user_id=user_id)

    # Mark user as deleted so subsequent reads return [] immediately
    # (Mem0 delete_users is eventually consistent).
    self._deleted_user_ids.add(user_id)
    self._known_user_ids.discard(user_id)

    # Evict matching entries from cache
    to_remove = [did for did, mem in self._memory_cache.items() if (mem.get("metadata") or {}).get("user_id") == user_id]
    for did in to_remove:
      self._id_cache.pop(did, None)
      self._memory_cache.pop(did, None)

  async def delete_session_data(self, session_id: str) -> None:
    await self._ensure_initialized()

    # Mem0 has no session-level bulk delete — find matching episodes
    # across all users, then delete each one individually.
    memories = await self._get_all_by_type(
      "episode",
      user_id=None,
      extra_filters={"session_id": session_id},
      global_search=True,
    )

    for mem in memories:
      mem0_id = mem.get("id")
      if mem0_id:
        await self._client.delete(mem0_id)
        md = mem.get("metadata") or {}
        definable_id = md.get("definable_id")
        if definable_id:
          self._id_cache.pop(definable_id, None)
          self._memory_cache.pop(definable_id, None)

  # ------------------------------------------------------------------
  # Context manager
  # ------------------------------------------------------------------

  async def __aenter__(self) -> "Mem0MemoryStore":
    await self.initialize()
    return self

  async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    await self.close()
