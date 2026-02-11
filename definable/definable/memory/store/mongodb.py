"""MongoDB-backed memory store."""

import re
import time
from typing import Any, List, Optional
from uuid import uuid4

from definable.memory.store._utils import cosine_similarity
from definable.memory.types import Episode, KnowledgeAtom, Procedure, TopicTransition
from definable.utils.log import log_debug


class MongoMemoryStore:
  """Async MongoDB memory store using motor.

  Collections are auto-created on first `initialize()` call.
  Vector search falls back to Python-side cosine similarity over recent entries.
  """

  def __init__(
    self,
    connection_string: str = "mongodb://localhost:27017",
    database: str = "memory",
    collection_prefix: str = "",
  ):
    self._connection_string = connection_string
    self._database_name = database
    self._prefix = collection_prefix
    self._client: Any = None
    self._db: Any = None
    self._episodes: Any = None
    self._atoms: Any = None
    self._procedures: Any = None
    self._topic_transitions: Any = None
    self._initialized = False

  async def initialize(self) -> None:
    if self._initialized:
      return

    try:
      import motor.motor_asyncio as motor
    except ImportError as e:
      raise ImportError("motor is required for MongoMemoryStore. Install it with: pip install definable[mongodb-memory]") from e

    self._client = motor.AsyncIOMotorClient(self._connection_string)
    self._db = self._client[self._database_name]
    self._episodes = self._db[f"{self._prefix}episodes"]
    self._atoms = self._db[f"{self._prefix}atoms"]
    self._procedures = self._db[f"{self._prefix}procedures"]
    self._topic_transitions = self._db[f"{self._prefix}topic_transitions"]
    await self._create_indexes()
    self._initialized = True
    log_debug("MongoMemoryStore initialized", log_level=2)

  async def close(self) -> None:
    if self._client:
      self._client.close()
      self._client = None
      self._db = None
      self._episodes = None
      self._atoms = None
      self._procedures = None
      self._topic_transitions = None
      self._initialized = False

  async def _ensure_initialized(self) -> None:
    if not self._initialized:
      await self.initialize()

  async def _create_indexes(self) -> None:
    # Episodes indexes
    await self._episodes.create_index([("user_id", 1), ("created_at", -1)])
    await self._episodes.create_index([("session_id", 1), ("created_at", -1)])
    await self._episodes.create_index([("compression_stage", 1), ("created_at", -1)])

    # Atoms indexes
    await self._atoms.create_index([("user_id", 1), ("last_accessed_at", -1)])
    await self._atoms.create_index([("subject", 1), ("predicate", 1)])
    await self._atoms.create_index([("confidence", 1)])

    # Procedures indexes
    await self._procedures.create_index([("user_id", 1), ("confidence", -1)])

    # Topic transitions indexes
    await self._topic_transitions.create_index(
      [("user_id", 1), ("from_topic", 1), ("to_topic", 1)],
      unique=True,
    )
    await self._topic_transitions.create_index([("from_topic", 1), ("user_id", 1)])

  # --- Document conversion helpers ---

  def _episode_to_doc(self, episode: Episode) -> dict:
    """Convert an Episode to a MongoDB document."""
    return {
      "_id": episode.id,
      "user_id": episode.user_id,
      "session_id": episode.session_id,
      "role": episode.role,
      "content": episode.content,
      "embedding": episode.embedding,
      "topics": episode.topics,
      "sentiment": episode.sentiment,
      "token_count": episode.token_count,
      "compression_stage": episode.compression_stage,
      "created_at": episode.created_at,
      "last_accessed_at": episode.last_accessed_at,
      "access_count": episode.access_count,
    }

  def _doc_to_episode(self, doc: dict) -> Episode:
    """Convert a MongoDB document to an Episode."""
    return Episode(
      id=doc["_id"],
      user_id=doc.get("user_id"),
      session_id=doc["session_id"],
      role=doc["role"],
      content=doc["content"],
      embedding=doc.get("embedding"),
      topics=doc.get("topics") or [],
      sentiment=doc.get("sentiment") or 0.0,
      token_count=doc.get("token_count") or 0,
      compression_stage=doc.get("compression_stage") or 0,
      created_at=doc.get("created_at") or 0.0,
      last_accessed_at=doc.get("last_accessed_at") or 0.0,
      access_count=doc.get("access_count") or 0,
    )

  def _atom_to_doc(self, atom: KnowledgeAtom) -> dict:
    """Convert a KnowledgeAtom to a MongoDB document."""
    return {
      "_id": atom.id,
      "user_id": atom.user_id,
      "subject": atom.subject,
      "predicate": atom.predicate,
      "object": atom.object,
      "content": atom.content,
      "embedding": atom.embedding,
      "confidence": atom.confidence,
      "reinforcement_count": atom.reinforcement_count,
      "topics": atom.topics,
      "token_count": atom.token_count,
      "source_episode_ids": atom.source_episode_ids,
      "created_at": atom.created_at,
      "last_accessed_at": atom.last_accessed_at,
      "last_reinforced_at": atom.last_reinforced_at,
      "access_count": atom.access_count,
    }

  def _doc_to_atom(self, doc: dict) -> KnowledgeAtom:
    """Convert a MongoDB document to a KnowledgeAtom."""
    return KnowledgeAtom(
      id=doc["_id"],
      user_id=doc.get("user_id"),
      subject=doc["subject"],
      predicate=doc["predicate"],
      object=doc["object"],
      content=doc["content"],
      embedding=doc.get("embedding"),
      confidence=doc.get("confidence") or 1.0,
      reinforcement_count=doc.get("reinforcement_count") or 0,
      topics=doc.get("topics") or [],
      token_count=doc.get("token_count") or 0,
      source_episode_ids=doc.get("source_episode_ids") or [],
      created_at=doc.get("created_at") or 0.0,
      last_accessed_at=doc.get("last_accessed_at") or 0.0,
      last_reinforced_at=doc.get("last_reinforced_at") or 0.0,
      access_count=doc.get("access_count") or 0,
    )

  def _procedure_to_doc(self, procedure: Procedure) -> dict:
    """Convert a Procedure to a MongoDB document."""
    return {
      "_id": procedure.id,
      "user_id": procedure.user_id,
      "trigger_text": procedure.trigger,
      "action": procedure.action,
      "confidence": procedure.confidence,
      "observation_count": procedure.observation_count,
      "created_at": procedure.created_at,
      "last_accessed_at": procedure.last_accessed_at,
    }

  def _doc_to_procedure(self, doc: dict) -> Procedure:
    """Convert a MongoDB document to a Procedure."""
    return Procedure(
      id=doc["_id"],
      user_id=doc.get("user_id"),
      trigger=doc["trigger_text"],
      action=doc["action"],
      confidence=doc.get("confidence") or 0.5,
      observation_count=doc.get("observation_count") or 1,
      created_at=doc.get("created_at") or 0.0,
      last_accessed_at=doc.get("last_accessed_at") or 0.0,
    )

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

    doc = self._episode_to_doc(episode)
    await self._episodes.insert_one(doc)
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
    query: dict[str, Any] = {}

    if user_id is not None:
      query["user_id"] = user_id
    if session_id is not None:
      query["session_id"] = session_id
    if min_stage is not None:
      query.setdefault("compression_stage", {})["$gte"] = min_stage
    if max_stage is not None:
      query.setdefault("compression_stage", {})["$lte"] = max_stage

    cursor = self._episodes.find(query).sort("created_at", -1).limit(limit)
    docs = await cursor.to_list(length=limit)
    return [self._doc_to_episode(doc) for doc in docs]

  async def update_episode(self, episode_id: str, **fields) -> None:
    await self._ensure_initialized()
    if not fields:
      return
    await self._episodes.update_one({"_id": episode_id}, {"$set": fields})

  async def get_episodes_for_distillation(self, stage: int, older_than: float) -> List[Episode]:
    await self._ensure_initialized()
    query = {
      "compression_stage": stage,
      "created_at": {"$lt": older_than},
    }
    cursor = self._episodes.find(query).sort("created_at", 1).limit(50)
    docs = await cursor.to_list(length=50)
    return [self._doc_to_episode(doc) for doc in docs]

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

    doc = self._atom_to_doc(atom)
    await self._atoms.insert_one(doc)
    return atom.id

  async def get_atoms(
    self,
    *,
    user_id: Optional[str] = None,
    min_confidence: float = 0.1,
    limit: int = 50,
  ) -> List[KnowledgeAtom]:
    await self._ensure_initialized()
    query: dict[str, Any] = {"confidence": {"$gte": min_confidence}}

    if user_id is not None:
      query["user_id"] = user_id

    cursor = self._atoms.find(query).sort("last_accessed_at", -1).limit(limit)
    docs = await cursor.to_list(length=limit)
    return [self._doc_to_atom(doc) for doc in docs]

  async def find_similar_atom(
    self,
    subject: str,
    predicate: str,
    user_id: Optional[str] = None,
  ) -> Optional[KnowledgeAtom]:
    await self._ensure_initialized()
    query: dict[str, Any] = {
      "subject": {"$regex": f"^{re.escape(subject)}$", "$options": "i"},
      "predicate": {"$regex": f"^{re.escape(predicate)}$", "$options": "i"},
    }
    if user_id is not None:
      query["user_id"] = user_id

    doc = await self._atoms.find_one(query)
    return self._doc_to_atom(doc) if doc else None

  async def update_atom(self, atom_id: str, **fields) -> None:
    await self._ensure_initialized()
    if not fields:
      return
    await self._atoms.update_one({"_id": atom_id}, {"$set": fields})

  async def prune_atoms(self, min_confidence: float) -> int:
    await self._ensure_initialized()
    result = await self._atoms.delete_many({"confidence": {"$lt": min_confidence}})
    return result.deleted_count

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

    doc = self._procedure_to_doc(procedure)
    await self._procedures.insert_one(doc)
    return procedure.id

  async def get_procedures(
    self,
    *,
    user_id: Optional[str] = None,
    min_confidence: float = 0.3,
  ) -> List[Procedure]:
    await self._ensure_initialized()
    query: dict[str, Any] = {"confidence": {"$gte": min_confidence}}

    if user_id is not None:
      query["user_id"] = user_id

    cursor = self._procedures.find(query).sort("confidence", -1)
    docs = await cursor.to_list(length=1000)
    return [self._doc_to_procedure(doc) for doc in docs]

  async def find_similar_procedure(
    self,
    trigger: str,
    user_id: Optional[str] = None,
  ) -> Optional[Procedure]:
    await self._ensure_initialized()
    # Simple keyword matching — fetch all procedures, find best overlap in Python
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
    # Map 'trigger' field to 'trigger_text' document key
    update_fields = {}
    for key, value in fields.items():
      doc_key = "trigger_text" if key == "trigger" else key
      update_fields[doc_key] = value
    await self._procedures.update_one({"_id": procedure_id}, {"$set": update_fields})

  # --- Topics ---

  async def store_topic_transition(
    self,
    from_topic: str,
    to_topic: str,
    user_id: Optional[str] = None,
  ) -> None:
    await self._ensure_initialized()
    await self._topic_transitions.update_one(
      {"user_id": user_id, "from_topic": from_topic, "to_topic": to_topic},
      {
        "$inc": {"count": 1},
        "$setOnInsert": {"_id": str(uuid4())},
      },
      upsert=True,
    )

  async def get_topic_transitions(
    self,
    from_topic: str,
    user_id: Optional[str] = None,
    min_count: int = 3,
  ) -> List[TopicTransition]:
    await self._ensure_initialized()
    query: dict[str, Any] = {
      "from_topic": from_topic,
      "count": {"$gte": min_count},
    }
    if user_id is not None:
      query["user_id"] = user_id

    cursor = self._topic_transitions.find(query).sort("count", -1)
    docs = await cursor.to_list(length=1000)

    # Compute probabilities
    total = sum(doc["count"] for doc in docs)
    transitions = []
    for doc in docs:
      prob = doc["count"] / total if total > 0 else 0.0
      transitions.append(TopicTransition(from_topic=doc["from_topic"], to_topic=doc["to_topic"], count=doc["count"], probability=prob))
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
    # Fetch recent episodes with embeddings, compute similarity in Python
    query: dict[str, Any] = {"embedding": {"$ne": None}}
    if user_id is not None:
      query["user_id"] = user_id

    cursor = self._episodes.find(query).sort("created_at", -1).limit(1000)
    docs = await cursor.to_list(length=1000)

    scored = []
    for doc in docs:
      episode = self._doc_to_episode(doc)
      if episode.embedding:
        sim = cosine_similarity(embedding, episode.embedding)
        scored.append((sim, episode))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [ep for _, ep in scored[:top_k]]

  async def search_atoms_by_embedding(
    self,
    embedding: List[float],
    *,
    user_id: Optional[str] = None,
    top_k: int = 20,
  ) -> List[KnowledgeAtom]:
    await self._ensure_initialized()
    query: dict[str, Any] = {"embedding": {"$ne": None}}
    if user_id is not None:
      query["user_id"] = user_id

    cursor = self._atoms.find(query).sort("last_accessed_at", -1).limit(1000)
    docs = await cursor.to_list(length=1000)

    scored = []
    for doc in docs:
      atom = self._doc_to_atom(doc)
      if atom.embedding:
        sim = cosine_similarity(embedding, atom.embedding)
        scored.append((sim, atom))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [atom for _, atom in scored[:top_k]]

  # --- Deletion ---

  async def delete_user_data(self, user_id: str) -> None:
    await self._ensure_initialized()
    for collection in (self._episodes, self._atoms, self._procedures, self._topic_transitions):
      await collection.delete_many({"user_id": user_id})

  async def delete_session_data(self, session_id: str) -> None:
    await self._ensure_initialized()
    await self._episodes.delete_many({"session_id": session_id})

  async def __aenter__(self):
    await self.initialize()
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    await self.close()
