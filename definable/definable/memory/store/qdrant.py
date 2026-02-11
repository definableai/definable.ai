"""Qdrant-backed memory store."""

import contextlib
import json
import time
from typing import Any, List, Optional
from uuid import uuid4

from definable.memory.types import Episode, KnowledgeAtom, Procedure, TopicTransition
from definable.utils.log import log_debug


class QdrantMemoryStore:
  """Async Qdrant memory store.

  Collections are auto-created on first `initialize()` call.
  Vector search is native Qdrant cosine similarity.
  """

  def __init__(
    self,
    url: str = "localhost",
    port: int = 6333,
    api_key: Optional[str] = None,
    prefix: str = "memory",
    vector_size: int = 1536,
  ):
    self._url = url
    self._port = port
    self._api_key = api_key
    self._prefix = prefix
    self._vector_size = vector_size
    self._client: Any = None
    self._initialized = False

  @property
  def _episodes_collection(self) -> str:
    return f"{self._prefix}_episodes"

  @property
  def _atoms_collection(self) -> str:
    return f"{self._prefix}_atoms"

  @property
  def _procedures_collection(self) -> str:
    return f"{self._prefix}_procedures"

  @property
  def _transitions_collection(self) -> str:
    return f"{self._prefix}_transitions"

  async def initialize(self) -> None:
    if self._initialized:
      return

    try:
      from qdrant_client import AsyncQdrantClient, models  # noqa: F401
    except ImportError as e:
      raise ImportError("qdrant-client is required for QdrantMemoryStore. Install it with: pip install definable[qdrant-memory]") from e

    self._client = AsyncQdrantClient(url=self._url, port=self._port, api_key=self._api_key)
    self._models = models
    await self._ensure_collections()
    self._initialized = True
    log_debug("QdrantMemoryStore initialized", log_level=2)

  async def _ensure_collections(self) -> None:
    models = self._models

    # Collections with real vectors: episodes, atoms
    for name in (self._episodes_collection, self._atoms_collection):
      if not await self._client.collection_exists(name):
        await self._client.create_collection(
          collection_name=name,
          vectors_config=models.VectorParams(size=self._vector_size, distance=models.Distance.COSINE),
        )
        await self._create_payload_indexes(name)

    # Collections without meaningful vectors: procedures, transitions
    for name in (self._procedures_collection, self._transitions_collection):
      if not await self._client.collection_exists(name):
        await self._client.create_collection(
          collection_name=name,
          vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
        )
        await self._create_payload_indexes(name)

  async def _create_payload_indexes(self, collection_name: str) -> None:
    models = self._models

    # Common indexes
    index_fields = {
      "user_id": models.PayloadSchemaType.KEYWORD,
      "session_id": models.PayloadSchemaType.KEYWORD,
      "compression_stage": models.PayloadSchemaType.INTEGER,
      "confidence": models.PayloadSchemaType.FLOAT,
      "subject": models.PayloadSchemaType.KEYWORD,
      "subject_lower": models.PayloadSchemaType.KEYWORD,
      "predicate": models.PayloadSchemaType.KEYWORD,
      "predicate_lower": models.PayloadSchemaType.KEYWORD,
      "from_topic": models.PayloadSchemaType.KEYWORD,
      "to_topic": models.PayloadSchemaType.KEYWORD,
      "created_at": models.PayloadSchemaType.FLOAT,
      "count": models.PayloadSchemaType.INTEGER,
    }

    for field_name, schema_type in index_fields.items():
      # Index may already exist or field may not be relevant to this collection
      with contextlib.suppress(Exception):
        await self._client.create_payload_index(
          collection_name=collection_name,
          field_name=field_name,
          field_schema=schema_type,
        )

  async def close(self) -> None:
    if self._client:
      await self._client.close()
      self._client = None
      self._initialized = False

  async def _ensure_initialized(self) -> None:
    if not self._initialized:
      await self.initialize()

  # --- Payload conversion helpers ---

  def _episode_to_payload(self, episode: Episode) -> dict:
    return {
      "id": episode.id,
      "user_id": episode.user_id,
      "session_id": episode.session_id,
      "role": episode.role,
      "content": episode.content,
      "embedding_json": json.dumps(episode.embedding) if episode.embedding else None,
      "topics": episode.topics,
      "sentiment": episode.sentiment,
      "token_count": episode.token_count,
      "compression_stage": episode.compression_stage,
      "created_at": episode.created_at,
      "last_accessed_at": episode.last_accessed_at,
      "access_count": episode.access_count,
    }

  def _payload_to_episode(self, payload: dict) -> Episode:
    embedding_json = payload.get("embedding_json")
    embedding = json.loads(embedding_json) if embedding_json else None
    return Episode(
      id=payload.get("id", ""),
      user_id=payload.get("user_id"),
      session_id=payload.get("session_id", ""),
      role=payload.get("role", ""),
      content=payload.get("content", ""),
      embedding=embedding,
      topics=payload.get("topics", []),
      sentiment=payload.get("sentiment", 0.0),
      token_count=payload.get("token_count", 0),
      compression_stage=payload.get("compression_stage", 0),
      created_at=payload.get("created_at", 0.0),
      last_accessed_at=payload.get("last_accessed_at", 0.0),
      access_count=payload.get("access_count", 0),
    )

  def _atom_to_payload(self, atom: KnowledgeAtom) -> dict:
    return {
      "id": atom.id,
      "user_id": atom.user_id,
      "subject": atom.subject,
      "subject_lower": atom.subject.lower(),
      "predicate": atom.predicate,
      "predicate_lower": atom.predicate.lower(),
      "object": atom.object,
      "content": atom.content,
      "embedding_json": json.dumps(atom.embedding) if atom.embedding else None,
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

  def _payload_to_atom(self, payload: dict) -> KnowledgeAtom:
    embedding_json = payload.get("embedding_json")
    embedding = json.loads(embedding_json) if embedding_json else None
    return KnowledgeAtom(
      id=payload.get("id", ""),
      user_id=payload.get("user_id"),
      subject=payload.get("subject", ""),
      predicate=payload.get("predicate", ""),
      object=payload.get("object", ""),
      content=payload.get("content", ""),
      embedding=embedding,
      confidence=payload.get("confidence", 1.0),
      reinforcement_count=payload.get("reinforcement_count", 0),
      topics=payload.get("topics", []),
      token_count=payload.get("token_count", 0),
      source_episode_ids=payload.get("source_episode_ids", []),
      created_at=payload.get("created_at", 0.0),
      last_accessed_at=payload.get("last_accessed_at", 0.0),
      last_reinforced_at=payload.get("last_reinforced_at", 0.0),
      access_count=payload.get("access_count", 0),
    )

  def _procedure_to_payload(self, procedure: Procedure) -> dict:
    return {
      "id": procedure.id,
      "user_id": procedure.user_id,
      "trigger": procedure.trigger,
      "action": procedure.action,
      "confidence": procedure.confidence,
      "observation_count": procedure.observation_count,
      "created_at": procedure.created_at,
      "last_accessed_at": procedure.last_accessed_at,
    }

  def _payload_to_procedure(self, payload: dict) -> Procedure:
    return Procedure(
      id=payload.get("id", ""),
      user_id=payload.get("user_id"),
      trigger=payload.get("trigger", ""),
      action=payload.get("action", ""),
      confidence=payload.get("confidence", 0.5),
      observation_count=payload.get("observation_count", 1),
      created_at=payload.get("created_at", 0.0),
      last_accessed_at=payload.get("last_accessed_at", 0.0),
    )

  def _zero_vector(self, size: Optional[int] = None) -> List[float]:
    return [0.0] * (size if size is not None else self._vector_size)

  # --- Episodes ---

  async def store_episode(self, episode: Episode) -> str:
    await self._ensure_initialized()
    models = self._models

    if not episode.id:
      episode.id = str(uuid4())
    now = time.time()
    if episode.created_at == 0.0:
      episode.created_at = now
    if episode.last_accessed_at == 0.0:
      episode.last_accessed_at = now

    vector = episode.embedding or self._zero_vector()
    payload = self._episode_to_payload(episode)

    await self._client.upsert(
      collection_name=self._episodes_collection,
      points=[
        models.PointStruct(
          id=episode.id,
          vector=vector,
          payload=payload,
        )
      ],
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
    models = self._models

    conditions: List[Any] = []
    if user_id is not None:
      conditions.append(models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)))
    if session_id is not None:
      conditions.append(models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id)))
    if min_stage is not None:
      conditions.append(models.FieldCondition(key="compression_stage", range=models.Range(gte=min_stage)))
    if max_stage is not None:
      conditions.append(models.FieldCondition(key="compression_stage", range=models.Range(lte=max_stage)))

    scroll_filter = models.Filter(must=conditions) if conditions else None

    points, _ = await self._client.scroll(
      collection_name=self._episodes_collection,
      scroll_filter=scroll_filter,
      limit=limit,
      order_by=models.OrderBy(key="created_at", direction=models.Direction.DESC),
      with_payload=True,
      with_vectors=False,
    )
    return [self._payload_to_episode(point.payload) for point in points]

  async def update_episode(self, episode_id: str, **fields) -> None:
    await self._ensure_initialized()
    if not fields:
      return

    payload_update: dict = {}
    for key, value in fields.items():
      if key == "embedding":
        payload_update["embedding_json"] = json.dumps(value) if value is not None else None
        # Also update the vector if embedding is provided
        if value is not None:
          models = self._models
          await self._client.update_vectors(
            collection_name=self._episodes_collection,
            points=[models.PointVectors(id=episode_id, vector=value)],
          )
      elif key in ("topics", "source_episode_ids"):
        payload_update[key] = value
      else:
        payload_update[key] = value

    if payload_update:
      await self._client.set_payload(
        collection_name=self._episodes_collection,
        payload=payload_update,
        points=[episode_id],
      )

  async def get_episodes_for_distillation(self, stage: int, older_than: float) -> List[Episode]:
    await self._ensure_initialized()
    models = self._models

    conditions = [
      models.FieldCondition(key="compression_stage", match=models.MatchValue(value=stage)),
      models.FieldCondition(key="created_at", range=models.Range(lt=older_than)),
    ]

    points, _ = await self._client.scroll(
      collection_name=self._episodes_collection,
      scroll_filter=models.Filter(must=conditions),
      limit=50,
      order_by=models.OrderBy(key="created_at", direction=models.Direction.ASC),
      with_payload=True,
      with_vectors=False,
    )
    return [self._payload_to_episode(point.payload) for point in points]

  # --- Knowledge Atoms ---

  async def store_atom(self, atom: KnowledgeAtom) -> str:
    await self._ensure_initialized()
    models = self._models

    if not atom.id:
      atom.id = str(uuid4())
    now = time.time()
    if atom.created_at == 0.0:
      atom.created_at = now
    if atom.last_accessed_at == 0.0:
      atom.last_accessed_at = now

    vector = atom.embedding or self._zero_vector()
    payload = self._atom_to_payload(atom)

    await self._client.upsert(
      collection_name=self._atoms_collection,
      points=[
        models.PointStruct(
          id=atom.id,
          vector=vector,
          payload=payload,
        )
      ],
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
    models = self._models

    conditions: List[Any] = [
      models.FieldCondition(key="confidence", range=models.Range(gte=min_confidence)),
    ]
    if user_id is not None:
      conditions.append(models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)))

    points, _ = await self._client.scroll(
      collection_name=self._atoms_collection,
      scroll_filter=models.Filter(must=conditions),
      limit=limit,
      order_by=models.OrderBy(key="last_accessed_at", direction=models.Direction.DESC),
      with_payload=True,
      with_vectors=False,
    )
    return [self._payload_to_atom(point.payload) for point in points]

  async def find_similar_atom(
    self,
    subject: str,
    predicate: str,
    user_id: Optional[str] = None,
  ) -> Optional[KnowledgeAtom]:
    await self._ensure_initialized()
    models = self._models

    conditions: List[Any] = [
      models.FieldCondition(key="subject_lower", match=models.MatchValue(value=subject.lower())),
      models.FieldCondition(key="predicate_lower", match=models.MatchValue(value=predicate.lower())),
    ]
    if user_id is not None:
      conditions.append(models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)))

    points, _ = await self._client.scroll(
      collection_name=self._atoms_collection,
      scroll_filter=models.Filter(must=conditions),
      limit=1,
      with_payload=True,
      with_vectors=False,
    )
    if points:
      return self._payload_to_atom(points[0].payload)
    return None

  async def update_atom(self, atom_id: str, **fields) -> None:
    await self._ensure_initialized()
    if not fields:
      return

    payload_update: dict = {}
    for key, value in fields.items():
      if key == "embedding":
        payload_update["embedding_json"] = json.dumps(value) if value is not None else None
        if value is not None:
          models = self._models
          await self._client.update_vectors(
            collection_name=self._atoms_collection,
            points=[models.PointVectors(id=atom_id, vector=value)],
          )
      elif key in ("topics", "source_episode_ids"):
        payload_update[key] = value
      elif key == "subject":
        payload_update["subject"] = value
        payload_update["subject_lower"] = value.lower() if isinstance(value, str) else value
      elif key == "predicate":
        payload_update["predicate"] = value
        payload_update["predicate_lower"] = value.lower() if isinstance(value, str) else value
      else:
        payload_update[key] = value

    if payload_update:
      await self._client.set_payload(
        collection_name=self._atoms_collection,
        payload=payload_update,
        points=[atom_id],
      )

  async def prune_atoms(self, min_confidence: float) -> int:
    await self._ensure_initialized()
    models = self._models

    # First, count how many atoms will be pruned
    conditions = [
      models.FieldCondition(key="confidence", range=models.Range(lt=min_confidence)),
    ]
    scroll_filter = models.Filter(must=conditions)

    points, _ = await self._client.scroll(
      collection_name=self._atoms_collection,
      scroll_filter=scroll_filter,
      limit=10000,
      with_payload=False,
      with_vectors=False,
    )
    count = len(points)

    if count > 0:
      point_ids = [point.id for point in points]
      await self._client.delete(
        collection_name=self._atoms_collection,
        points_selector=models.PointIdsList(points=point_ids),
      )

    return count

  # --- Procedures ---

  async def store_procedure(self, procedure: Procedure) -> str:
    await self._ensure_initialized()
    models = self._models

    if not procedure.id:
      procedure.id = str(uuid4())
    now = time.time()
    if procedure.created_at == 0.0:
      procedure.created_at = now
    if procedure.last_accessed_at == 0.0:
      procedure.last_accessed_at = now

    payload = self._procedure_to_payload(procedure)

    await self._client.upsert(
      collection_name=self._procedures_collection,
      points=[
        models.PointStruct(
          id=procedure.id,
          vector=self._zero_vector(size=1),
          payload=payload,
        )
      ],
    )
    return procedure.id

  async def get_procedures(
    self,
    *,
    user_id: Optional[str] = None,
    min_confidence: float = 0.3,
  ) -> List[Procedure]:
    await self._ensure_initialized()
    models = self._models

    conditions: List[Any] = [
      models.FieldCondition(key="confidence", range=models.Range(gte=min_confidence)),
    ]
    if user_id is not None:
      conditions.append(models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)))

    points, _ = await self._client.scroll(
      collection_name=self._procedures_collection,
      scroll_filter=models.Filter(must=conditions),
      limit=1000,
      with_payload=True,
      with_vectors=False,
    )

    procedures = [self._payload_to_procedure(point.payload) for point in points]
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

    payload_update: dict = {}
    for key, value in fields.items():
      payload_update[key] = value

    if payload_update:
      await self._client.set_payload(
        collection_name=self._procedures_collection,
        payload=payload_update,
        points=[procedure_id],
      )

  # --- Topics ---

  async def store_topic_transition(
    self,
    from_topic: str,
    to_topic: str,
    user_id: Optional[str] = None,
  ) -> None:
    await self._ensure_initialized()
    models = self._models

    # Check if transition already exists
    conditions: List[Any] = [
      models.FieldCondition(key="from_topic", match=models.MatchValue(value=from_topic)),
      models.FieldCondition(key="to_topic", match=models.MatchValue(value=to_topic)),
    ]
    if user_id is not None:
      conditions.append(models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)))
    else:
      conditions.append(models.FieldCondition(key="user_id_is_none", match=models.MatchValue(value=True)))

    points, _ = await self._client.scroll(
      collection_name=self._transitions_collection,
      scroll_filter=models.Filter(must=conditions),
      limit=1,
      with_payload=True,
      with_vectors=False,
    )

    if points:
      # Increment count
      existing = points[0]
      new_count = existing.payload.get("count", 0) + 1
      await self._client.set_payload(
        collection_name=self._transitions_collection,
        payload={"count": new_count},
        points=[existing.id],
      )
    else:
      # Insert new transition
      point_id = str(uuid4())
      payload = {
        "id": point_id,
        "user_id": user_id,
        "user_id_is_none": user_id is None,
        "from_topic": from_topic,
        "to_topic": to_topic,
        "count": 1,
      }
      await self._client.upsert(
        collection_name=self._transitions_collection,
        points=[
          models.PointStruct(
            id=point_id,
            vector=self._zero_vector(size=1),
            payload=payload,
          )
        ],
      )

  async def get_topic_transitions(
    self,
    from_topic: str,
    user_id: Optional[str] = None,
    min_count: int = 3,
  ) -> List[TopicTransition]:
    await self._ensure_initialized()
    models = self._models

    conditions: List[Any] = [
      models.FieldCondition(key="from_topic", match=models.MatchValue(value=from_topic)),
      models.FieldCondition(key="count", range=models.Range(gte=min_count)),
    ]
    if user_id is not None:
      conditions.append(models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)))

    points, _ = await self._client.scroll(
      collection_name=self._transitions_collection,
      scroll_filter=models.Filter(must=conditions),
      limit=1000,
      with_payload=True,
      with_vectors=False,
    )

    # Sort by count descending
    points_sorted = sorted(points, key=lambda p: p.payload.get("count", 0), reverse=True)

    # Compute probabilities
    total = sum(p.payload.get("count", 0) for p in points_sorted)
    transitions = []
    for point in points_sorted:
      count = point.payload.get("count", 0)
      prob = count / total if total > 0 else 0.0
      transitions.append(
        TopicTransition(
          from_topic=point.payload.get("from_topic", ""),
          to_topic=point.payload.get("to_topic", ""),
          count=count,
          probability=prob,
        )
      )
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
    models = self._models

    conditions: List[Any] = []
    if user_id is not None:
      conditions.append(models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)))

    query_filter = models.Filter(must=conditions) if conditions else None

    results = await self._client.search(
      collection_name=self._episodes_collection,
      query_vector=embedding,
      query_filter=query_filter,
      limit=top_k,
      with_payload=True,
    )

    episodes = []
    for scored_point in results:
      ep = self._payload_to_episode(scored_point.payload)
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
    models = self._models

    conditions: List[Any] = []
    if user_id is not None:
      conditions.append(models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)))

    query_filter = models.Filter(must=conditions) if conditions else None

    results = await self._client.search(
      collection_name=self._atoms_collection,
      query_vector=embedding,
      query_filter=query_filter,
      limit=top_k,
      with_payload=True,
    )

    atoms = []
    for scored_point in results:
      atom = self._payload_to_atom(scored_point.payload)
      atoms.append(atom)
    return atoms

  # --- Deletion ---

  async def delete_user_data(self, user_id: str) -> None:
    await self._ensure_initialized()
    models = self._models

    user_filter = models.Filter(must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))])

    for collection in (
      self._episodes_collection,
      self._atoms_collection,
      self._procedures_collection,
      self._transitions_collection,
    ):
      await self._client.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(filter=user_filter),
      )

  async def delete_session_data(self, session_id: str) -> None:
    await self._ensure_initialized()
    models = self._models

    session_filter = models.Filter(must=[models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id))])

    await self._client.delete(
      collection_name=self._episodes_collection,
      points_selector=models.FilterSelector(filter=session_filter),
    )

  async def __aenter__(self):
    await self.initialize()
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    await self.close()
