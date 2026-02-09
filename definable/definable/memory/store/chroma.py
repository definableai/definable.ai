"""ChromaDB-backed memory store."""

import asyncio
import json
import time
from typing import Any, List, Optional
from uuid import uuid4

from definable.memory.types import Episode, KnowledgeAtom, Procedure, TopicTransition
from definable.utils.log import log_debug


class ChromaMemoryStore:
  """Async ChromaDB memory store.

  Collections are auto-created on first `initialize()` call.
  Vector search uses ChromaDB's native cosine distance.
  All ChromaDB operations are run via `asyncio.to_thread` to avoid
  blocking the event loop (ChromaDB's Python client is synchronous).
  """

  def __init__(
    self,
    persist_directory: Optional[str] = None,
    collection_prefix: str = "memory_",
  ):
    self._persist_directory = persist_directory
    self._prefix = collection_prefix
    self._client: Any = None
    self._episodes: Any = None
    self._atoms: Any = None
    self._procedures: Any = None
    self._transitions: Any = None
    self._initialized = False

  async def initialize(self) -> None:
    if self._initialized:
      return

    try:
      import chromadb
    except ImportError as e:
      raise ImportError("chromadb is required for ChromaMemoryStore. Install it with: pip install definable-ai[chroma-memory]") from e

    if self._persist_directory is not None:
      self._client = await asyncio.to_thread(chromadb.PersistentClient, path=self._persist_directory)
    else:
      self._client = await asyncio.to_thread(chromadb.EphemeralClient)

    cosine_meta = {"hnsw:space": "cosine"}
    self._episodes = await asyncio.to_thread(
      self._client.get_or_create_collection,
      name=f"{self._prefix}episodes",
      metadata=cosine_meta,
    )
    self._atoms = await asyncio.to_thread(
      self._client.get_or_create_collection,
      name=f"{self._prefix}atoms",
      metadata=cosine_meta,
    )
    self._procedures = await asyncio.to_thread(
      self._client.get_or_create_collection,
      name=f"{self._prefix}procedures",
      metadata=cosine_meta,
    )
    self._transitions = await asyncio.to_thread(
      self._client.get_or_create_collection,
      name=f"{self._prefix}transitions",
      metadata=cosine_meta,
    )
    self._initialized = True
    log_debug("ChromaMemoryStore initialized", log_level=2)

  async def close(self) -> None:
    # ChromaDB client doesn't need explicit closing
    self._client = None
    self._episodes = None
    self._atoms = None
    self._procedures = None
    self._transitions = None
    self._initialized = False

  async def _ensure_initialized(self) -> None:
    if not self._initialized:
      await self.initialize()

  # --- Metadata helpers ---

  def _episode_to_metadata(self, episode: Episode) -> dict:
    """Flatten an Episode to a ChromaDB-compatible flat metadata dict."""
    meta: dict[str, Any] = {}
    if episode.user_id is not None:
      meta["user_id"] = episode.user_id
    meta["session_id"] = episode.session_id
    meta["role"] = episode.role
    meta["topics"] = json.dumps(episode.topics)
    meta["sentiment"] = episode.sentiment
    meta["token_count"] = episode.token_count
    meta["compression_stage"] = episode.compression_stage
    meta["created_at"] = episode.created_at
    meta["last_accessed_at"] = episode.last_accessed_at
    meta["access_count"] = episode.access_count
    return meta

  def _metadata_to_episode(
    self,
    id: str,
    document: Optional[str],
    metadata: dict,
    embedding: Optional[List[float]],
  ) -> Episode:
    """Reconstruct an Episode from ChromaDB record parts."""
    topics_raw = metadata.get("topics", "[]")
    topics = json.loads(topics_raw) if isinstance(topics_raw, str) else topics_raw
    return Episode(
      id=id,
      user_id=metadata.get("user_id"),
      session_id=metadata.get("session_id", ""),
      role=metadata.get("role", "user"),
      content=document or "",
      embedding=embedding,
      topics=topics,
      sentiment=metadata.get("sentiment", 0.0),
      token_count=metadata.get("token_count", 0),
      compression_stage=metadata.get("compression_stage", 0),
      created_at=metadata.get("created_at", 0.0),
      last_accessed_at=metadata.get("last_accessed_at", 0.0),
      access_count=metadata.get("access_count", 0),
    )

  def _atom_to_metadata(self, atom: KnowledgeAtom) -> dict:
    """Flatten a KnowledgeAtom to a ChromaDB-compatible flat metadata dict."""
    meta: dict[str, Any] = {}
    if atom.user_id is not None:
      meta["user_id"] = atom.user_id
    meta["subject"] = atom.subject
    meta["predicate"] = atom.predicate
    meta["object"] = atom.object
    meta["subject_lower"] = atom.subject.lower()
    meta["predicate_lower"] = atom.predicate.lower()
    meta["confidence"] = atom.confidence
    meta["reinforcement_count"] = atom.reinforcement_count
    meta["topics"] = json.dumps(atom.topics)
    meta["token_count"] = atom.token_count
    meta["source_episode_ids"] = json.dumps(atom.source_episode_ids)
    meta["created_at"] = atom.created_at
    meta["last_accessed_at"] = atom.last_accessed_at
    meta["last_reinforced_at"] = atom.last_reinforced_at
    meta["access_count"] = atom.access_count
    return meta

  def _metadata_to_atom(
    self,
    id: str,
    document: Optional[str],
    metadata: dict,
    embedding: Optional[List[float]],
  ) -> KnowledgeAtom:
    """Reconstruct a KnowledgeAtom from ChromaDB record parts."""
    topics_raw = metadata.get("topics", "[]")
    topics = json.loads(topics_raw) if isinstance(topics_raw, str) else topics_raw
    source_raw = metadata.get("source_episode_ids", "[]")
    source_ids = json.loads(source_raw) if isinstance(source_raw, str) else source_raw
    return KnowledgeAtom(
      id=id,
      user_id=metadata.get("user_id"),
      subject=metadata.get("subject", ""),
      predicate=metadata.get("predicate", ""),
      object=metadata.get("object", ""),
      content=document or "",
      embedding=embedding,
      confidence=metadata.get("confidence", 1.0),
      reinforcement_count=metadata.get("reinforcement_count", 0),
      topics=topics,
      token_count=metadata.get("token_count", 0),
      source_episode_ids=source_ids,
      created_at=metadata.get("created_at", 0.0),
      last_accessed_at=metadata.get("last_accessed_at", 0.0),
      last_reinforced_at=metadata.get("last_reinforced_at", 0.0),
      access_count=metadata.get("access_count", 0),
    )

  def _procedure_to_metadata(self, procedure: Procedure) -> dict:
    """Flatten a Procedure to a ChromaDB-compatible flat metadata dict."""
    meta: dict[str, Any] = {}
    if procedure.user_id is not None:
      meta["user_id"] = procedure.user_id
    meta["trigger_text"] = procedure.trigger
    meta["action"] = procedure.action
    meta["confidence"] = procedure.confidence
    meta["observation_count"] = procedure.observation_count
    meta["created_at"] = procedure.created_at
    meta["last_accessed_at"] = procedure.last_accessed_at
    return meta

  def _metadata_to_procedure(
    self,
    id: str,
    document: Optional[str],
    metadata: dict,
  ) -> Procedure:
    """Reconstruct a Procedure from ChromaDB record parts."""
    return Procedure(
      id=id,
      user_id=metadata.get("user_id"),
      trigger=metadata.get("trigger_text", document or ""),
      action=metadata.get("action", ""),
      confidence=metadata.get("confidence", 0.5),
      observation_count=metadata.get("observation_count", 1),
      created_at=metadata.get("created_at", 0.0),
      last_accessed_at=metadata.get("last_accessed_at", 0.0),
    )

  def _transition_to_metadata(
    self,
    from_topic: str,
    to_topic: str,
    count: int,
    user_id: Optional[str] = None,
  ) -> dict:
    """Flatten a topic transition to a ChromaDB-compatible flat metadata dict."""
    meta: dict[str, Any] = {
      "from_topic": from_topic,
      "to_topic": to_topic,
      "count": count,
    }
    if user_id is not None:
      meta["user_id"] = user_id
    return meta

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

    metadata = self._episode_to_metadata(episode)
    kwargs: dict[str, Any] = {
      "ids": [episode.id],
      "documents": [episode.content],
      "metadatas": [metadata],
    }
    if episode.embedding:
      kwargs["embeddings"] = [episode.embedding]

    await asyncio.to_thread(self._episodes.upsert, **kwargs)
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

    where_clauses: list[dict] = []
    if user_id is not None:
      where_clauses.append({"user_id": user_id})
    if session_id is not None:
      where_clauses.append({"session_id": session_id})
    if min_stage is not None:
      where_clauses.append({"compression_stage": {"$gte": min_stage}})
    if max_stage is not None:
      where_clauses.append({"compression_stage": {"$lte": max_stage}})

    get_kwargs: dict[str, Any] = {"include": ["documents", "metadatas", "embeddings"]}
    if len(where_clauses) == 1:
      get_kwargs["where"] = where_clauses[0]
    elif len(where_clauses) > 1:
      get_kwargs["where"] = {"$and": where_clauses}

    result = await asyncio.to_thread(self._episodes.get, **get_kwargs)

    episodes = []
    ids = result["ids"] or []
    documents = result["documents"] or []
    metadatas = result["metadatas"] or []
    embeddings = result["embeddings"] or [None] * len(ids)
    for i, eid in enumerate(ids):
      doc = documents[i] if i < len(documents) else None
      meta = metadatas[i] if i < len(metadatas) else {}
      emb = embeddings[i] if embeddings and i < len(embeddings) else None
      episodes.append(self._metadata_to_episode(eid, doc, meta, emb))

    # ChromaDB doesn't guarantee ordering — sort by created_at DESC in Python
    episodes.sort(key=lambda e: e.created_at, reverse=True)
    return episodes[:limit]

  async def update_episode(self, episode_id: str, **fields) -> None:
    await self._ensure_initialized()
    if not fields:
      return

    # Fetch existing record
    existing = await asyncio.to_thread(
      self._episodes.get,
      ids=[episode_id],
      include=["documents", "metadatas", "embeddings"],
    )
    if not existing["ids"]:
      return

    old_meta = existing["metadatas"][0] if existing["metadatas"] else {}
    old_doc = existing["documents"][0] if existing["documents"] else ""

    new_doc = old_doc
    new_embedding = None
    meta_updates: dict[str, Any] = {}

    for key, value in fields.items():
      if key == "content":
        new_doc = value
      elif key == "embedding":
        new_embedding = value
      elif key in ("topics", "source_episode_ids"):
        meta_updates[key] = json.dumps(value) if value is not None else "[]"
      else:
        meta_updates[key] = value

    merged_meta = {**old_meta, **meta_updates}

    update_kwargs: dict[str, Any] = {
      "ids": [episode_id],
      "documents": [new_doc],
      "metadatas": [merged_meta],
    }
    if new_embedding is not None:
      update_kwargs["embeddings"] = [new_embedding]

    await asyncio.to_thread(self._episodes.update, **update_kwargs)

  async def get_episodes_for_distillation(self, stage: int, older_than: float) -> List[Episode]:
    await self._ensure_initialized()

    where = {"$and": [{"compression_stage": stage}, {"created_at": {"$lt": older_than}}]}

    result = await asyncio.to_thread(
      self._episodes.get,
      where=where,
      include=["documents", "metadatas", "embeddings"],
    )

    episodes = []
    ids = result["ids"] or []
    documents = result["documents"] or []
    metadatas = result["metadatas"] or []
    embeddings = result["embeddings"] or [None] * len(ids)
    for i, eid in enumerate(ids):
      doc = documents[i] if i < len(documents) else None
      meta = metadatas[i] if i < len(metadatas) else {}
      emb = embeddings[i] if embeddings and i < len(embeddings) else None
      episodes.append(self._metadata_to_episode(eid, doc, meta, emb))

    # Sort ASC by created_at, limit to 50
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

    metadata = self._atom_to_metadata(atom)
    kwargs: dict[str, Any] = {
      "ids": [atom.id],
      "documents": [atom.content],
      "metadatas": [metadata],
    }
    if atom.embedding:
      kwargs["embeddings"] = [atom.embedding]

    await asyncio.to_thread(self._atoms.upsert, **kwargs)
    return atom.id

  async def get_atoms(
    self,
    *,
    user_id: Optional[str] = None,
    min_confidence: float = 0.1,
    limit: int = 50,
  ) -> List[KnowledgeAtom]:
    await self._ensure_initialized()

    where_clauses: list[dict] = [{"confidence": {"$gte": min_confidence}}]
    if user_id is not None:
      where_clauses.append({"user_id": user_id})

    where: dict
    if len(where_clauses) == 1:
      where = where_clauses[0]
    else:
      where = {"$and": where_clauses}

    result = await asyncio.to_thread(
      self._atoms.get,
      where=where,
      include=["documents", "metadatas", "embeddings"],
    )

    atoms = []
    ids = result["ids"] or []
    documents = result["documents"] or []
    metadatas = result["metadatas"] or []
    embeddings = result["embeddings"] or [None] * len(ids)
    for i, aid in enumerate(ids):
      doc = documents[i] if i < len(documents) else None
      meta = metadatas[i] if i < len(metadatas) else {}
      emb = embeddings[i] if embeddings and i < len(embeddings) else None
      atoms.append(self._metadata_to_atom(aid, doc, meta, emb))

    # Sort by last_accessed_at DESC
    atoms.sort(key=lambda a: a.last_accessed_at, reverse=True)
    return atoms[:limit]

  async def find_similar_atom(
    self,
    subject: str,
    predicate: str,
    user_id: Optional[str] = None,
  ) -> Optional[KnowledgeAtom]:
    await self._ensure_initialized()

    where_clauses: list[dict] = [
      {"subject_lower": subject.lower()},
      {"predicate_lower": predicate.lower()},
    ]
    if user_id is not None:
      where_clauses.append({"user_id": user_id})

    where = {"$and": where_clauses}

    result = await asyncio.to_thread(
      self._atoms.get,
      where=where,
      include=["documents", "metadatas", "embeddings"],
    )

    ids = result["ids"] or []
    if not ids:
      return None

    documents = result["documents"] or []
    metadatas = result["metadatas"] or []
    embeddings = result["embeddings"] or [None] * len(ids)
    doc = documents[0] if documents else None
    meta = metadatas[0] if metadatas else {}
    emb = embeddings[0] if embeddings else None
    return self._metadata_to_atom(ids[0], doc, meta, emb)

  async def update_atom(self, atom_id: str, **fields) -> None:
    await self._ensure_initialized()
    if not fields:
      return

    # Fetch existing record
    existing = await asyncio.to_thread(
      self._atoms.get,
      ids=[atom_id],
      include=["documents", "metadatas", "embeddings"],
    )
    if not existing["ids"]:
      return

    old_meta = existing["metadatas"][0] if existing["metadatas"] else {}
    old_doc = existing["documents"][0] if existing["documents"] else ""

    new_doc = old_doc
    new_embedding = None
    meta_updates: dict[str, Any] = {}

    for key, value in fields.items():
      if key == "content":
        new_doc = value
      elif key == "embedding":
        new_embedding = value
      elif key in ("topics", "source_episode_ids"):
        meta_updates[key] = json.dumps(value) if value is not None else "[]"
      elif key == "subject":
        meta_updates["subject"] = value
        meta_updates["subject_lower"] = value.lower()
      elif key == "predicate":
        meta_updates["predicate"] = value
        meta_updates["predicate_lower"] = value.lower()
      else:
        meta_updates[key] = value

    merged_meta = {**old_meta, **meta_updates}

    update_kwargs: dict[str, Any] = {
      "ids": [atom_id],
      "documents": [new_doc],
      "metadatas": [merged_meta],
    }
    if new_embedding is not None:
      update_kwargs["embeddings"] = [new_embedding]

    await asyncio.to_thread(self._atoms.update, **update_kwargs)

  async def prune_atoms(self, min_confidence: float) -> int:
    await self._ensure_initialized()

    # Find atoms below threshold
    result = await asyncio.to_thread(
      self._atoms.get,
      where={"confidence": {"$lt": min_confidence}},
      include=[],
    )

    ids = result["ids"] or []
    if not ids:
      return 0

    await asyncio.to_thread(self._atoms.delete, ids=ids)
    return len(ids)

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
      self._procedures.upsert,
      ids=[procedure.id],
      documents=[procedure.trigger],
      metadatas=[metadata],
    )
    return procedure.id

  async def get_procedures(
    self,
    *,
    user_id: Optional[str] = None,
    min_confidence: float = 0.3,
  ) -> List[Procedure]:
    await self._ensure_initialized()

    where_clauses: list[dict] = [{"confidence": {"$gte": min_confidence}}]
    if user_id is not None:
      where_clauses.append({"user_id": user_id})

    where: dict
    if len(where_clauses) == 1:
      where = where_clauses[0]
    else:
      where = {"$and": where_clauses}

    result = await asyncio.to_thread(
      self._procedures.get,
      where=where,
      include=["documents", "metadatas"],
    )

    procedures = []
    ids = result["ids"] or []
    documents = result["documents"] or []
    metadatas = result["metadatas"] or []
    for i, pid in enumerate(ids):
      doc = documents[i] if i < len(documents) else None
      meta = metadatas[i] if i < len(metadatas) else {}
      procedures.append(self._metadata_to_procedure(pid, doc, meta))

    # Sort by confidence DESC
    procedures.sort(key=lambda p: p.confidence, reverse=True)
    return procedures

  async def find_similar_procedure(
    self,
    trigger: str,
    user_id: Optional[str] = None,
  ) -> Optional[Procedure]:
    await self._ensure_initialized()
    # Simple keyword matching — look for procedures whose trigger overlaps with the query
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

  async def update_procedure(self, procedure_id: str, **fields) -> None:
    await self._ensure_initialized()
    if not fields:
      return

    # Fetch existing record
    existing = await asyncio.to_thread(
      self._procedures.get,
      ids=[procedure_id],
      include=["documents", "metadatas"],
    )
    if not existing["ids"]:
      return

    old_meta = existing["metadatas"][0] if existing["metadatas"] else {}
    old_doc = existing["documents"][0] if existing["documents"] else ""

    new_doc = old_doc
    meta_updates: dict[str, Any] = {}

    for key, value in fields.items():
      if key == "trigger":
        # Map 'trigger' field to 'trigger_text' in metadata and update document
        meta_updates["trigger_text"] = value
        new_doc = value
      else:
        meta_updates[key] = value

    merged_meta = {**old_meta, **meta_updates}

    await asyncio.to_thread(
      self._procedures.update,
      ids=[procedure_id],
      documents=[new_doc],
      metadatas=[merged_meta],
    )

  # --- Topics ---

  async def store_topic_transition(
    self,
    from_topic: str,
    to_topic: str,
    user_id: Optional[str] = None,
  ) -> None:
    await self._ensure_initialized()

    # Look for existing transition
    where_clauses: list[dict] = [
      {"from_topic": from_topic},
      {"to_topic": to_topic},
    ]
    if user_id is not None:
      where_clauses.append({"user_id": user_id})

    where = {"$and": where_clauses}

    result = await asyncio.to_thread(
      self._transitions.get,
      where=where,
      include=["metadatas", "documents"],
    )

    ids = result["ids"] or []

    if ids:
      # Increment count on existing
      old_meta = result["metadatas"][0] if result["metadatas"] else {}
      old_count = old_meta.get("count", 0)
      new_meta = {**old_meta, "count": old_count + 1}
      await asyncio.to_thread(
        self._transitions.update,
        ids=[ids[0]],
        metadatas=[new_meta],
      )
    else:
      # Insert new
      transition_id = str(uuid4())
      metadata = self._transition_to_metadata(from_topic, to_topic, 1, user_id)
      doc = json.dumps({"from_topic": from_topic, "to_topic": to_topic})
      await asyncio.to_thread(
        self._transitions.upsert,
        ids=[transition_id],
        documents=[doc],
        metadatas=[metadata],
      )

  async def get_topic_transitions(
    self,
    from_topic: str,
    user_id: Optional[str] = None,
    min_count: int = 3,
  ) -> List[TopicTransition]:
    await self._ensure_initialized()

    where_clauses: list[dict] = [{"from_topic": from_topic}]
    if user_id is not None:
      where_clauses.append({"user_id": user_id})

    where: dict
    if len(where_clauses) == 1:
      where = where_clauses[0]
    else:
      where = {"$and": where_clauses}

    result = await asyncio.to_thread(
      self._transitions.get,
      where=where,
      include=["metadatas"],
    )

    ids = result["ids"] or []
    metadatas = result["metadatas"] or []

    # Filter by min_count in Python (ChromaDB may not support compound where well for this)
    raw_transitions = []
    for i, _ in enumerate(ids):
      meta = metadatas[i] if i < len(metadatas) else {}
      count = meta.get("count", 0)
      if count >= min_count:
        raw_transitions.append((meta.get("from_topic", ""), meta.get("to_topic", ""), count))

    # Sort by count DESC
    raw_transitions.sort(key=lambda x: x[2], reverse=True)

    # Compute probabilities
    total = sum(t[2] for t in raw_transitions)
    transitions = []
    for ft, tt, cnt in raw_transitions:
      prob = cnt / total if total > 0 else 0.0
      transitions.append(TopicTransition(from_topic=ft, to_topic=tt, count=cnt, probability=prob))

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

    query_kwargs: dict[str, Any] = {
      "query_embeddings": [embedding],
      "n_results": top_k,
      "include": ["documents", "metadatas", "embeddings"],
    }
    if user_id is not None:
      query_kwargs["where"] = {"user_id": user_id}

    result = await asyncio.to_thread(self._episodes.query, **query_kwargs)

    episodes = []
    # query returns lists of lists (one per query)
    ids = result["ids"][0] if result["ids"] else []
    documents = result["documents"][0] if result["documents"] else []
    metadatas = result["metadatas"][0] if result["metadatas"] else []
    embeddings = result["embeddings"][0] if result["embeddings"] else [None] * len(ids)
    for i, eid in enumerate(ids):
      doc = documents[i] if i < len(documents) else None
      meta = metadatas[i] if i < len(metadatas) else {}
      emb = embeddings[i] if embeddings and i < len(embeddings) else None
      episodes.append(self._metadata_to_episode(eid, doc, meta, emb))

    return episodes

  async def search_atoms_by_embedding(
    self,
    embedding: List[float],
    *,
    user_id: Optional[str] = None,
    top_k: int = 20,
  ) -> List[KnowledgeAtom]:
    await self._ensure_initialized()

    query_kwargs: dict[str, Any] = {
      "query_embeddings": [embedding],
      "n_results": top_k,
      "include": ["documents", "metadatas", "embeddings"],
    }
    if user_id is not None:
      query_kwargs["where"] = {"user_id": user_id}

    result = await asyncio.to_thread(self._atoms.query, **query_kwargs)

    atoms = []
    # query returns lists of lists (one per query)
    ids = result["ids"][0] if result["ids"] else []
    documents = result["documents"][0] if result["documents"] else []
    metadatas = result["metadatas"][0] if result["metadatas"] else []
    embeddings = result["embeddings"][0] if result["embeddings"] else [None] * len(ids)
    for i, aid in enumerate(ids):
      doc = documents[i] if i < len(documents) else None
      meta = metadatas[i] if i < len(metadatas) else {}
      emb = embeddings[i] if embeddings and i < len(embeddings) else None
      atoms.append(self._metadata_to_atom(aid, doc, meta, emb))

    return atoms

  # --- Deletion ---

  async def delete_user_data(self, user_id: str) -> None:
    await self._ensure_initialized()
    for collection in (self._episodes, self._atoms, self._procedures, self._transitions):
      # ChromaDB delete with where filter
      try:
        await asyncio.to_thread(collection.delete, where={"user_id": user_id})
      except Exception:
        # Collection might be empty or no matching records — that's fine
        pass

  async def delete_session_data(self, session_id: str) -> None:
    await self._ensure_initialized()
    try:
      await asyncio.to_thread(self._episodes.delete, where={"session_id": session_id})
    except Exception:
      pass

  # --- Context manager ---

  async def __aenter__(self):
    await self.initialize()
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    await self.close()
