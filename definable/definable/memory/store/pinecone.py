"""Pinecone-backed memory store."""

import asyncio
import os
import time
from typing import Any, List, Optional
from uuid import uuid4

from definable.memory.types import Episode, KnowledgeAtom, Procedure, TopicTransition
from definable.utils.log import log_debug


class PineconeMemoryStore:
  """Async Pinecone memory store.

  Uses a single Pinecone index with four namespaces: episodes, atoms,
  procedures, transitions. All metadata is stored in Pinecone metadata
  dicts. The Pinecone SDK v5 is synchronous, so all operations are
  wrapped in ``asyncio.to_thread()``.
  """

  def __init__(
    self,
    api_key: str = "",
    index_name: str = "memory",
    environment: Optional[str] = None,
    vector_size: int = 1536,
  ):
    self._api_key = api_key or os.environ.get("PINECONE_API_KEY", "")
    self._index_name = index_name
    self._environment = environment
    self._vector_size = vector_size
    self._index: Any = None
    self._initialized = False

  async def initialize(self) -> None:
    if self._initialized:
      return

    try:
      from pinecone import Pinecone
    except ImportError as e:
      raise ImportError("pinecone is required for PineconeMemoryStore. Install it with: pip install definable[pinecone-memory]") from e

    if not self._api_key:
      raise ValueError("Pinecone API key is required. Set api_key or PINECONE_API_KEY environment variable.")

    def _setup() -> Any:
      pc = Pinecone(api_key=self._api_key)
      existing = [idx.name for idx in pc.list_indexes()]
      if self._index_name not in existing:
        from pinecone import ServerlessSpec

        pc.create_index(
          name=self._index_name,
          dimension=self._vector_size,
          metric="cosine",
          spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
      return pc.Index(self._index_name)

    self._index = await asyncio.to_thread(_setup)
    self._initialized = True
    log_debug("PineconeMemoryStore initialized", log_level=2)

  async def close(self) -> None:
    # Pinecone client does not require explicit closing.
    self._index = None
    self._initialized = False

  async def _ensure_initialized(self) -> None:
    if not self._initialized:
      await self.initialize()

  # --- Helper methods ---

  def _zero_vector(self) -> List[float]:
    return [0.0] * self._vector_size

  def _episode_to_metadata(self, episode: Episode) -> dict:
    return {
      "id": episode.id,
      "user_id": episode.user_id or "",
      "session_id": episode.session_id,
      "role": episode.role,
      "content": episode.content,
      "topics": episode.topics,
      "sentiment": episode.sentiment,
      "token_count": episode.token_count,
      "compression_stage": episode.compression_stage,
      "created_at": episode.created_at,
      "last_accessed_at": episode.last_accessed_at,
      "access_count": episode.access_count,
      "_has_embedding": episode.embedding is not None,
      "_record_type": "episode",
    }

  def _metadata_to_episode(self, metadata: dict, vector: Optional[List[float]] = None) -> Episode:
    has_embedding = metadata.get("_has_embedding", False)
    embedding = vector if has_embedding and vector else None
    return Episode(
      id=metadata.get("id", ""),
      user_id=metadata.get("user_id") or None,
      session_id=metadata.get("session_id", ""),
      role=metadata.get("role", ""),
      content=metadata.get("content", ""),
      embedding=embedding,
      topics=metadata.get("topics", []),
      sentiment=metadata.get("sentiment", 0.0),
      token_count=int(metadata.get("token_count", 0)),
      compression_stage=int(metadata.get("compression_stage", 0)),
      created_at=metadata.get("created_at", 0.0),
      last_accessed_at=metadata.get("last_accessed_at", 0.0),
      access_count=int(metadata.get("access_count", 0)),
    )

  def _atom_to_metadata(self, atom: KnowledgeAtom) -> dict:
    return {
      "id": atom.id,
      "user_id": atom.user_id or "",
      "subject": atom.subject,
      "subject_lower": atom.subject.lower(),
      "predicate": atom.predicate,
      "predicate_lower": atom.predicate.lower(),
      "object": atom.object,
      "content": atom.content,
      "confidence": atom.confidence,
      "reinforcement_count": atom.reinforcement_count,
      "topics": atom.topics,
      "token_count": atom.token_count,
      "source_episode_ids": atom.source_episode_ids,
      "created_at": atom.created_at,
      "last_accessed_at": atom.last_accessed_at,
      "last_reinforced_at": atom.last_reinforced_at,
      "access_count": atom.access_count,
      "_has_embedding": atom.embedding is not None,
      "_record_type": "atom",
    }

  def _metadata_to_atom(self, metadata: dict, vector: Optional[List[float]] = None) -> KnowledgeAtom:
    has_embedding = metadata.get("_has_embedding", False)
    embedding = vector if has_embedding and vector else None
    return KnowledgeAtom(
      id=metadata.get("id", ""),
      user_id=metadata.get("user_id") or None,
      subject=metadata.get("subject", ""),
      predicate=metadata.get("predicate", ""),
      object=metadata.get("object", ""),
      content=metadata.get("content", ""),
      embedding=embedding,
      confidence=metadata.get("confidence", 1.0),
      reinforcement_count=int(metadata.get("reinforcement_count", 0)),
      topics=metadata.get("topics", []),
      token_count=int(metadata.get("token_count", 0)),
      source_episode_ids=metadata.get("source_episode_ids", []),
      created_at=metadata.get("created_at", 0.0),
      last_accessed_at=metadata.get("last_accessed_at", 0.0),
      last_reinforced_at=metadata.get("last_reinforced_at", 0.0),
      access_count=int(metadata.get("access_count", 0)),
    )

  def _procedure_to_metadata(self, procedure: Procedure) -> dict:
    return {
      "id": procedure.id,
      "user_id": procedure.user_id or "",
      "trigger": procedure.trigger,
      "action": procedure.action,
      "confidence": procedure.confidence,
      "observation_count": procedure.observation_count,
      "created_at": procedure.created_at,
      "last_accessed_at": procedure.last_accessed_at,
      "_record_type": "procedure",
    }

  def _metadata_to_procedure(self, metadata: dict) -> Procedure:
    return Procedure(
      id=metadata.get("id", ""),
      user_id=metadata.get("user_id") or None,
      trigger=metadata.get("trigger", ""),
      action=metadata.get("action", ""),
      confidence=metadata.get("confidence", 0.5),
      observation_count=int(metadata.get("observation_count", 1)),
      created_at=metadata.get("created_at", 0.0),
      last_accessed_at=metadata.get("last_accessed_at", 0.0),
    )

  def _build_filter(self, conditions: dict) -> dict:
    """Build a Pinecone metadata filter from key-value pairs.

    Supports simple equality via ``{"key": value}`` and range operators
    via ``{"key": {"$gte": v}}`` style dicts (passed through as-is).
    """
    parts: List[dict] = []
    for key, value in conditions.items():
      if isinstance(value, dict):
        parts.append({key: value})
      else:
        parts.append({key: {"$eq": value}})
    if len(parts) == 1:
      return parts[0]
    return {"$and": parts}

  # --- Episodes ---

  async def store_episode(self, episode: Episode) -> str:
    await self._ensure_initialized()
    if not episode.id:
      episode.id = str(uuid4())
    now = time.time()
    if episode.created_at == 0.0:
      episode.created_at = now
    if episode.last_accessed_at == 0.0:
      episode.last_accessed_at = now

    vector = episode.embedding or self._zero_vector()
    metadata = self._episode_to_metadata(episode)

    await asyncio.to_thread(
      self._index.upsert,
      vectors=[(episode.id, vector, metadata)],
      namespace="episodes",
    )
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

    conditions: dict = {}
    if user_id is not None:
      conditions["user_id"] = user_id
    if session_id is not None:
      conditions["session_id"] = session_id
    if min_stage is not None:
      conditions["compression_stage"] = conditions.get("compression_stage", {})
      if isinstance(conditions["compression_stage"], dict):
        conditions["compression_stage"]["$gte"] = min_stage
      else:
        conditions["compression_stage"] = {"$gte": min_stage}
    if max_stage is not None:
      if "compression_stage" in conditions and isinstance(conditions["compression_stage"], dict):
        conditions["compression_stage"]["$lte"] = max_stage
      else:
        conditions["compression_stage"] = {"$lte": max_stage}

    metadata_filter = self._build_filter(conditions) if conditions else None

    # Fetch a large batch and sort in Python (Pinecone lacks ORDER BY created_at)
    fetch_limit = max(limit * 10, 10000)

    result = await asyncio.to_thread(
      self._index.query,
      vector=self._zero_vector(),
      top_k=fetch_limit,
      filter=metadata_filter,
      namespace="episodes",
      include_metadata=True,
    )

    episodes = [self._metadata_to_episode(match["metadata"]) for match in result.get("matches", [])]
    episodes.sort(key=lambda e: e.created_at, reverse=True)
    return episodes[:limit]

  async def update_episode(self, episode_id: str, **fields: Any) -> None:
    await self._ensure_initialized()
    if not fields:
      return

    # Fetch current metadata
    fetched = await asyncio.to_thread(
      self._index.fetch,
      ids=[episode_id],
      namespace="episodes",
    )
    vectors = fetched.get("vectors", {})
    if episode_id not in vectors:
      return

    current = vectors[episode_id]
    metadata = dict(current.get("metadata", {}))
    vector = list(current.get("values", self._zero_vector()))

    for key, value in fields.items():
      if key == "embedding":
        if value is not None:
          vector = value
          metadata["_has_embedding"] = True
        else:
          vector = self._zero_vector()
          metadata["_has_embedding"] = False
      elif key in ("topics", "source_episode_ids"):
        metadata[key] = value if value is not None else []
      else:
        metadata[key] = value

    await asyncio.to_thread(
      self._index.upsert,
      vectors=[(episode_id, vector, metadata)],
      namespace="episodes",
    )

  async def get_episodes_for_distillation(self, stage: int, older_than: float) -> List[Episode]:
    await self._ensure_initialized()

    metadata_filter = self._build_filter({
      "compression_stage": stage,
      "created_at": {"$lt": older_than},
    })

    result = await asyncio.to_thread(
      self._index.query,
      vector=self._zero_vector(),
      top_k=10000,
      filter=metadata_filter,
      namespace="episodes",
      include_metadata=True,
    )

    episodes = [self._metadata_to_episode(match["metadata"]) for match in result.get("matches", [])]
    episodes.sort(key=lambda e: e.created_at)
    return episodes[:50]

  # --- Knowledge Atoms ---

  async def store_atom(self, atom: KnowledgeAtom) -> str:
    await self._ensure_initialized()
    if not atom.id:
      atom.id = str(uuid4())
    now = time.time()
    if atom.created_at == 0.0:
      atom.created_at = now
    if atom.last_accessed_at == 0.0:
      atom.last_accessed_at = now

    vector = atom.embedding or self._zero_vector()
    metadata = self._atom_to_metadata(atom)

    await asyncio.to_thread(
      self._index.upsert,
      vectors=[(atom.id, vector, metadata)],
      namespace="atoms",
    )
    return atom.id

  async def get_atoms(
    self,
    *,
    user_id: Optional[str] = None,
    min_confidence: float = 0.1,
    limit: int = 50,
  ) -> List[KnowledgeAtom]:
    await self._ensure_initialized()

    conditions: dict = {
      "confidence": {"$gte": min_confidence},
    }
    if user_id is not None:
      conditions["user_id"] = user_id

    metadata_filter = self._build_filter(conditions)
    fetch_limit = max(limit * 10, 10000)

    result = await asyncio.to_thread(
      self._index.query,
      vector=self._zero_vector(),
      top_k=fetch_limit,
      filter=metadata_filter,
      namespace="atoms",
      include_metadata=True,
    )

    atoms = [self._metadata_to_atom(match["metadata"]) for match in result.get("matches", [])]
    atoms.sort(key=lambda a: a.last_accessed_at, reverse=True)
    return atoms[:limit]

  async def find_similar_atom(
    self,
    subject: str,
    predicate: str,
    user_id: Optional[str] = None,
  ) -> Optional[KnowledgeAtom]:
    await self._ensure_initialized()

    # Use lowered fields for case-insensitive matching
    conditions: dict = {
      "subject_lower": subject.lower(),
      "predicate_lower": predicate.lower(),
    }
    if user_id is not None:
      conditions["user_id"] = user_id

    metadata_filter = self._build_filter(conditions)

    result = await asyncio.to_thread(
      self._index.query,
      vector=self._zero_vector(),
      top_k=1,
      filter=metadata_filter,
      namespace="atoms",
      include_metadata=True,
    )

    matches = result.get("matches", [])
    if matches:
      return self._metadata_to_atom(matches[0]["metadata"])
    return None

  async def update_atom(self, atom_id: str, **fields: Any) -> None:
    await self._ensure_initialized()
    if not fields:
      return

    fetched = await asyncio.to_thread(
      self._index.fetch,
      ids=[atom_id],
      namespace="atoms",
    )
    vectors = fetched.get("vectors", {})
    if atom_id not in vectors:
      return

    current = vectors[atom_id]
    metadata = dict(current.get("metadata", {}))
    vector = list(current.get("values", self._zero_vector()))

    for key, value in fields.items():
      if key == "embedding":
        if value is not None:
          vector = value
          metadata["_has_embedding"] = True
        else:
          vector = self._zero_vector()
          metadata["_has_embedding"] = False
      elif key in ("topics", "source_episode_ids"):
        metadata[key] = value if value is not None else []
      elif key == "subject":
        metadata["subject"] = value
        metadata["subject_lower"] = value.lower() if isinstance(value, str) else value
      elif key == "predicate":
        metadata["predicate"] = value
        metadata["predicate_lower"] = value.lower() if isinstance(value, str) else value
      else:
        metadata[key] = value

    await asyncio.to_thread(
      self._index.upsert,
      vectors=[(atom_id, vector, metadata)],
      namespace="atoms",
    )

  async def prune_atoms(self, min_confidence: float) -> int:
    await self._ensure_initialized()

    metadata_filter = self._build_filter({
      "confidence": {"$lt": min_confidence},
    })

    result = await asyncio.to_thread(
      self._index.query,
      vector=self._zero_vector(),
      top_k=10000,
      filter=metadata_filter,
      namespace="atoms",
      include_metadata=True,
    )

    matches = result.get("matches", [])
    count = len(matches)

    if count > 0:
      ids_to_delete = [match["id"] for match in matches]
      await asyncio.to_thread(
        self._index.delete,
        ids=ids_to_delete,
        namespace="atoms",
      )

    return count

  # --- Procedures ---

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

    await asyncio.to_thread(
      self._index.upsert,
      vectors=[(procedure.id, self._zero_vector(), metadata)],
      namespace="procedures",
    )
    return procedure.id

  async def get_procedures(
    self,
    *,
    user_id: Optional[str] = None,
    min_confidence: float = 0.3,
  ) -> List[Procedure]:
    await self._ensure_initialized()

    conditions: dict = {
      "confidence": {"$gte": min_confidence},
    }
    if user_id is not None:
      conditions["user_id"] = user_id

    metadata_filter = self._build_filter(conditions)

    result = await asyncio.to_thread(
      self._index.query,
      vector=self._zero_vector(),
      top_k=10000,
      filter=metadata_filter,
      namespace="procedures",
      include_metadata=True,
    )

    procedures = [self._metadata_to_procedure(match["metadata"]) for match in result.get("matches", [])]
    procedures.sort(key=lambda p: p.confidence, reverse=True)
    return procedures

  async def find_similar_procedure(
    self,
    trigger: str,
    user_id: Optional[str] = None,
  ) -> Optional[Procedure]:
    await self._ensure_initialized()

    # Simple keyword matching -- fetch all procedures, find best overlap in Python
    trigger_lower = trigger.lower()
    trigger_words = set(trigger_lower.split())

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

    fetched = await asyncio.to_thread(
      self._index.fetch,
      ids=[procedure_id],
      namespace="procedures",
    )
    vectors = fetched.get("vectors", {})
    if procedure_id not in vectors:
      return

    current = vectors[procedure_id]
    metadata = dict(current.get("metadata", {}))

    for key, value in fields.items():
      metadata[key] = value

    await asyncio.to_thread(
      self._index.upsert,
      vectors=[(procedure_id, self._zero_vector(), metadata)],
      namespace="procedures",
    )

  # --- Topics ---

  async def store_topic_transition(
    self,
    from_topic: str,
    to_topic: str,
    user_id: Optional[str] = None,
  ) -> None:
    await self._ensure_initialized()

    # Check if transition already exists
    uid_value = user_id or ""
    conditions: dict = {
      "from_topic": from_topic,
      "to_topic": to_topic,
      "user_id": uid_value,
    }
    metadata_filter = self._build_filter(conditions)

    result = await asyncio.to_thread(
      self._index.query,
      vector=self._zero_vector(),
      top_k=1,
      filter=metadata_filter,
      namespace="transitions",
      include_metadata=True,
    )

    matches = result.get("matches", [])
    if matches:
      # Increment count
      existing = matches[0]
      existing_metadata = dict(existing["metadata"])
      existing_metadata["count"] = existing_metadata.get("count", 0) + 1
      await asyncio.to_thread(
        self._index.upsert,
        vectors=[(existing["id"], self._zero_vector(), existing_metadata)],
        namespace="transitions",
      )
    else:
      # Insert new transition
      point_id = str(uuid4())
      metadata = {
        "id": point_id,
        "user_id": uid_value,
        "from_topic": from_topic,
        "to_topic": to_topic,
        "count": 1,
        "_record_type": "transition",
      }
      await asyncio.to_thread(
        self._index.upsert,
        vectors=[(point_id, self._zero_vector(), metadata)],
        namespace="transitions",
      )

  async def get_topic_transitions(
    self,
    from_topic: str,
    user_id: Optional[str] = None,
    min_count: int = 3,
  ) -> List[TopicTransition]:
    await self._ensure_initialized()

    conditions: dict = {
      "from_topic": from_topic,
      "count": {"$gte": min_count},
    }
    if user_id is not None:
      conditions["user_id"] = user_id

    metadata_filter = self._build_filter(conditions)

    result = await asyncio.to_thread(
      self._index.query,
      vector=self._zero_vector(),
      top_k=10000,
      filter=metadata_filter,
      namespace="transitions",
      include_metadata=True,
    )

    raw = []
    for match in result.get("matches", []):
      md = match["metadata"]
      raw.append((md.get("from_topic", ""), md.get("to_topic", ""), int(md.get("count", 0))))

    # Sort by count descending
    raw.sort(key=lambda x: x[2], reverse=True)

    # Compute probabilities
    total = sum(r[2] for r in raw)
    transitions = []
    for from_t, to_t, count in raw:
      prob = count / total if total > 0 else 0.0
      transitions.append(TopicTransition(from_topic=from_t, to_topic=to_t, count=count, probability=prob))
    return transitions

  # --- Vector Search ---

  async def search_episodes_by_embedding(
    self,
    embedding: List[float],
    *,
    user_id: Optional[str] = None,
    top_k: int = 20,
  ) -> List[Episode]:
    await self._ensure_initialized()

    conditions: dict = {
      "_has_embedding": True,
    }
    if user_id is not None:
      conditions["user_id"] = user_id

    metadata_filter = self._build_filter(conditions)

    result = await asyncio.to_thread(
      self._index.query,
      vector=embedding,
      top_k=top_k,
      filter=metadata_filter,
      namespace="episodes",
      include_metadata=True,
      include_values=True,
    )

    episodes = []
    for match in result.get("matches", []):
      ep = self._metadata_to_episode(match["metadata"], vector=match.get("values"))
      episodes.append(ep)
    return episodes

  async def search_atoms_by_embedding(
    self,
    embedding: List[float],
    *,
    user_id: Optional[str] = None,
    top_k: int = 20,
  ) -> List[KnowledgeAtom]:
    await self._ensure_initialized()

    conditions: dict = {
      "_has_embedding": True,
    }
    if user_id is not None:
      conditions["user_id"] = user_id

    metadata_filter = self._build_filter(conditions)

    result = await asyncio.to_thread(
      self._index.query,
      vector=embedding,
      top_k=top_k,
      filter=metadata_filter,
      namespace="atoms",
      include_metadata=True,
      include_values=True,
    )

    atoms = []
    for match in result.get("matches", []):
      atom = self._metadata_to_atom(match["metadata"], vector=match.get("values"))
      atoms.append(atom)
    return atoms

  # --- Deletion ---

  async def delete_user_data(self, user_id: str) -> None:
    await self._ensure_initialized()
    metadata_filter = {"user_id": {"$eq": user_id}}

    for ns in ("episodes", "atoms", "procedures", "transitions"):
      await asyncio.to_thread(
        self._index.delete,
        filter=metadata_filter,
        namespace=ns,
      )

  async def delete_session_data(self, session_id: str) -> None:
    await self._ensure_initialized()
    metadata_filter = {"session_id": {"$eq": session_id}}

    await asyncio.to_thread(
      self._index.delete,
      filter=metadata_filter,
      namespace="episodes",
    )

  # --- Context manager ---

  async def __aenter__(self) -> "PineconeMemoryStore":
    await self.initialize()
    return self

  async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    await self.close()
