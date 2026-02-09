"""Redis-backed memory store."""

import json
import time
from typing import Any, List, Optional
from uuid import uuid4

from definable.memory.store._utils import cosine_similarity
from definable.memory.types import Episode, KnowledgeAtom, Procedure, TopicTransition
from definable.utils.log import log_debug


class RedisMemoryStore:
  """Async Redis memory store.

  Each record is stored as a Redis hash with secondary sorted sets for
  time-ordered retrieval.  Vector search falls back to Python-side
  cosine similarity (scan + compute) for maximum portability.
  """

  def __init__(self, redis_url: str = "redis://localhost:6379", prefix: str = "memory", db: int = 0):
    self.redis_url = redis_url
    self.prefix = prefix
    self.db = db
    self._client: Any = None
    self._initialized = False
    self._has_redisearch = False

  # ---------- Lifecycle ----------

  async def initialize(self) -> None:
    if self._initialized:
      return

    try:
      import redis.asyncio as aioredis
    except ImportError as e:
      raise ImportError("redis is required for RedisMemoryStore. Install it with: pip install definable-ai[redis-memory]") from e

    self._client = aioredis.from_url(self.redis_url, db=self.db, decode_responses=True)
    # Probe for RediSearch
    try:
      await self._client.execute_command("FT._LIST")
      self._has_redisearch = True
    except Exception:
      self._has_redisearch = False

    self._initialized = True
    log_debug("RedisMemoryStore initialized", log_level=2)

  async def close(self) -> None:
    if self._client:
      await self._client.aclose()
      self._client = None
      self._initialized = False

  async def _ensure_initialized(self) -> None:
    if not self._initialized:
      await self.initialize()

  # ---------- Key helpers ----------

  def _key(self, *parts: str) -> str:
    return ":".join([self.prefix, *parts])

  # ---------- Episode serialization ----------

  def _episode_to_hash(self, episode: Episode) -> dict:
    return {
      "id": episode.id,
      "user_id": episode.user_id or "",
      "session_id": episode.session_id,
      "role": episode.role,
      "content": episode.content,
      "embedding": json.dumps(episode.embedding) if episode.embedding else "",
      "topics": json.dumps(episode.topics),
      "sentiment": str(episode.sentiment),
      "token_count": str(episode.token_count),
      "compression_stage": str(episode.compression_stage),
      "created_at": str(episode.created_at),
      "last_accessed_at": str(episode.last_accessed_at),
      "access_count": str(episode.access_count),
    }

  def _hash_to_episode(self, data: dict) -> Episode:
    embedding_raw = data.get("embedding", "")
    topics_raw = data.get("topics", "[]")
    return Episode(
      id=data["id"],
      user_id=data.get("user_id") or None,
      session_id=data["session_id"],
      role=data["role"],
      content=data["content"],
      embedding=json.loads(embedding_raw) if embedding_raw else None,
      topics=json.loads(topics_raw) if topics_raw else [],
      sentiment=float(data.get("sentiment", 0.0)),
      token_count=int(data.get("token_count", 0)),
      compression_stage=int(data.get("compression_stage", 0)),
      created_at=float(data.get("created_at", 0.0)),
      last_accessed_at=float(data.get("last_accessed_at", 0.0)),
      access_count=int(data.get("access_count", 0)),
    )

  # ---------- Atom serialization ----------

  def _atom_to_hash(self, atom: KnowledgeAtom) -> dict:
    return {
      "id": atom.id,
      "user_id": atom.user_id or "",
      "subject": atom.subject,
      "predicate": atom.predicate,
      "object": atom.object,
      "content": atom.content,
      "embedding": json.dumps(atom.embedding) if atom.embedding else "",
      "confidence": str(atom.confidence),
      "reinforcement_count": str(atom.reinforcement_count),
      "topics": json.dumps(atom.topics),
      "token_count": str(atom.token_count),
      "source_episode_ids": json.dumps(atom.source_episode_ids),
      "created_at": str(atom.created_at),
      "last_accessed_at": str(atom.last_accessed_at),
      "last_reinforced_at": str(atom.last_reinforced_at),
      "access_count": str(atom.access_count),
    }

  def _hash_to_atom(self, data: dict) -> KnowledgeAtom:
    embedding_raw = data.get("embedding", "")
    topics_raw = data.get("topics", "[]")
    source_ids_raw = data.get("source_episode_ids", "[]")
    return KnowledgeAtom(
      id=data["id"],
      user_id=data.get("user_id") or None,
      subject=data["subject"],
      predicate=data["predicate"],
      object=data["object"],
      content=data["content"],
      embedding=json.loads(embedding_raw) if embedding_raw else None,
      confidence=float(data.get("confidence", 1.0)),
      reinforcement_count=int(data.get("reinforcement_count", 0)),
      topics=json.loads(topics_raw) if topics_raw else [],
      token_count=int(data.get("token_count", 0)),
      source_episode_ids=json.loads(source_ids_raw) if source_ids_raw else [],
      created_at=float(data.get("created_at", 0.0)),
      last_accessed_at=float(data.get("last_accessed_at", 0.0)),
      last_reinforced_at=float(data.get("last_reinforced_at", 0.0)),
      access_count=int(data.get("access_count", 0)),
    )

  # ---------- Procedure serialization ----------

  def _procedure_to_hash(self, procedure: Procedure) -> dict:
    return {
      "id": procedure.id,
      "user_id": procedure.user_id or "",
      "trigger": procedure.trigger,
      "action": procedure.action,
      "confidence": str(procedure.confidence),
      "observation_count": str(procedure.observation_count),
      "created_at": str(procedure.created_at),
      "last_accessed_at": str(procedure.last_accessed_at),
    }

  def _hash_to_procedure(self, data: dict) -> Procedure:
    return Procedure(
      id=data["id"],
      user_id=data.get("user_id") or None,
      trigger=data["trigger"],
      action=data["action"],
      confidence=float(data.get("confidence", 0.5)),
      observation_count=int(data.get("observation_count", 1)),
      created_at=float(data.get("created_at", 0.0)),
      last_accessed_at=float(data.get("last_accessed_at", 0.0)),
    )

  # ---------- Index helpers ----------

  async def _add_to_episode_indexes(self, episode: Episode) -> None:
    pipe = self._client.pipeline()
    pipe.zadd(self._key("episodes", "by_created"), {episode.id: episode.created_at})
    if episode.user_id:
      pipe.zadd(self._key("episodes", "user", episode.user_id), {episode.id: episode.created_at})
    pipe.zadd(self._key("episodes", "session", episode.session_id), {episode.id: episode.created_at})
    await pipe.execute()

  async def _remove_from_episode_indexes(self, episode_id: str, user_id: Optional[str], session_id: Optional[str]) -> None:
    pipe = self._client.pipeline()
    pipe.zrem(self._key("episodes", "by_created"), episode_id)
    if user_id:
      pipe.zrem(self._key("episodes", "user", user_id), episode_id)
    if session_id:
      pipe.zrem(self._key("episodes", "session", session_id), episode_id)
    await pipe.execute()

  async def _add_to_atom_indexes(self, atom: KnowledgeAtom) -> None:
    pipe = self._client.pipeline()
    pipe.zadd(self._key("atoms", "by_accessed"), {atom.id: atom.last_accessed_at})
    if atom.user_id:
      pipe.zadd(self._key("atoms", "user", atom.user_id), {atom.id: atom.last_accessed_at})
    await pipe.execute()

  async def _remove_from_atom_indexes(self, atom_id: str, user_id: Optional[str]) -> None:
    pipe = self._client.pipeline()
    pipe.zrem(self._key("atoms", "by_accessed"), atom_id)
    if user_id:
      pipe.zrem(self._key("atoms", "user", user_id), atom_id)
    await pipe.execute()

  async def _add_to_procedure_indexes(self, procedure: Procedure) -> None:
    pipe = self._client.pipeline()
    pipe.zadd(self._key("procedures", "by_confidence"), {procedure.id: procedure.confidence})
    if procedure.user_id:
      pipe.zadd(self._key("procedures", "user", procedure.user_id), {procedure.id: procedure.confidence})
    await pipe.execute()

  async def _remove_from_procedure_indexes(self, procedure_id: str, user_id: Optional[str]) -> None:
    pipe = self._client.pipeline()
    pipe.zrem(self._key("procedures", "by_confidence"), procedure_id)
    if user_id:
      pipe.zrem(self._key("procedures", "user", user_id), procedure_id)
    await pipe.execute()

  # ---------- Batch fetch helper ----------

  async def _fetch_hashes(self, keys: List[str]) -> List[dict]:
    """Pipeline HGETALL for multiple keys, returning list of dicts (skipping empty)."""
    if not keys:
      return []
    pipe = self._client.pipeline()
    for k in keys:
      pipe.hgetall(k)
    results = await pipe.execute()
    return [r for r in results if r]

  # ---------- Episodes ----------

  async def store_episode(self, episode: Episode) -> str:
    await self._ensure_initialized()
    if not episode.id:
      episode.id = str(uuid4())
    now = time.time()
    if episode.created_at == 0.0:
      episode.created_at = now
    if episode.last_accessed_at == 0.0:
      episode.last_accessed_at = now

    hash_key = self._key("episodes", episode.id)
    await self._client.hset(hash_key, mapping=self._episode_to_hash(episode))
    await self._add_to_episode_indexes(episode)
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

    # Pick the narrowest sorted set available
    if session_id is not None:
      sorted_key = self._key("episodes", "session", session_id)
    elif user_id is not None:
      sorted_key = self._key("episodes", "user", user_id)
    else:
      sorted_key = self._key("episodes", "by_created")

    # Get IDs ordered by created_at DESC (highest score first)
    ids = await self._client.zrevrange(sorted_key, 0, -1)
    if not ids:
      return []

    hash_keys = [self._key("episodes", eid) for eid in ids]
    raw_list = await self._fetch_hashes(hash_keys)

    episodes: List[Episode] = []
    for data in raw_list:
      ep = self._hash_to_episode(data)
      # Apply additional filters
      if user_id is not None and ep.user_id != user_id:
        continue
      if session_id is not None and ep.session_id != session_id:
        continue
      if min_stage is not None and ep.compression_stage < min_stage:
        continue
      if max_stage is not None and ep.compression_stage > max_stage:
        continue
      episodes.append(ep)

    episodes.sort(key=lambda e: e.created_at, reverse=True)
    return episodes[:limit]

  async def update_episode(self, episode_id: str, **fields: Any) -> None:
    await self._ensure_initialized()
    if not fields:
      return
    hash_key = self._key("episodes", episode_id)
    exists = await self._client.exists(hash_key)
    if not exists:
      return
    mapping: dict = {}
    for key, value in fields.items():
      if key in ("embedding", "topics", "source_episode_ids"):
        mapping[key] = json.dumps(value) if value is not None else ""
      elif isinstance(value, (int, float)):
        mapping[key] = str(value)
      else:
        mapping[key] = value if value is not None else ""
    await self._client.hset(hash_key, mapping=mapping)

  async def get_episodes_for_distillation(self, stage: int, older_than: float) -> List[Episode]:
    await self._ensure_initialized()
    # Fetch all episode IDs with created_at score up to older_than (exclusive)
    sorted_key = self._key("episodes", "by_created")
    # ZRANGEBYSCORE returns lowest scores first (ASC order), scores < older_than
    ids = await self._client.zrangebyscore(sorted_key, "-inf", f"({older_than}")
    if not ids:
      return []

    hash_keys = [self._key("episodes", eid) for eid in ids]
    raw_list = await self._fetch_hashes(hash_keys)

    episodes: List[Episode] = []
    for data in raw_list:
      ep = self._hash_to_episode(data)
      if ep.compression_stage == stage:
        episodes.append(ep)

    episodes.sort(key=lambda e: e.created_at)
    return episodes[:50]

  # ---------- Knowledge Atoms ----------

  async def store_atom(self, atom: KnowledgeAtom) -> str:
    await self._ensure_initialized()
    if not atom.id:
      atom.id = str(uuid4())
    now = time.time()
    if atom.created_at == 0.0:
      atom.created_at = now
    if atom.last_accessed_at == 0.0:
      atom.last_accessed_at = now

    hash_key = self._key("atoms", atom.id)
    await self._client.hset(hash_key, mapping=self._atom_to_hash(atom))
    await self._add_to_atom_indexes(atom)
    return atom.id

  async def get_atoms(
    self,
    *,
    user_id: Optional[str] = None,
    min_confidence: float = 0.1,
    limit: int = 50,
  ) -> List[KnowledgeAtom]:
    await self._ensure_initialized()

    if user_id is not None:
      sorted_key = self._key("atoms", "user", user_id)
    else:
      sorted_key = self._key("atoms", "by_accessed")

    ids = await self._client.zrevrange(sorted_key, 0, -1)
    if not ids:
      return []

    hash_keys = [self._key("atoms", aid) for aid in ids]
    raw_list = await self._fetch_hashes(hash_keys)

    atoms: List[KnowledgeAtom] = []
    for data in raw_list:
      atom = self._hash_to_atom(data)
      if atom.confidence < min_confidence:
        continue
      if user_id is not None and atom.user_id != user_id:
        continue
      atoms.append(atom)

    atoms.sort(key=lambda a: a.last_accessed_at, reverse=True)
    return atoms[:limit]

  async def find_similar_atom(
    self,
    subject: str,
    predicate: str,
    user_id: Optional[str] = None,
  ) -> Optional[KnowledgeAtom]:
    await self._ensure_initialized()
    subject_lower = subject.lower()
    predicate_lower = predicate.lower()

    if user_id is not None:
      sorted_key = self._key("atoms", "user", user_id)
    else:
      sorted_key = self._key("atoms", "by_accessed")

    ids = await self._client.zrevrange(sorted_key, 0, -1)
    if not ids:
      return None

    hash_keys = [self._key("atoms", aid) for aid in ids]
    raw_list = await self._fetch_hashes(hash_keys)

    for data in raw_list:
      atom = self._hash_to_atom(data)
      if user_id is not None and atom.user_id != user_id:
        continue
      if atom.subject.lower() == subject_lower and atom.predicate.lower() == predicate_lower:
        return atom
    return None

  async def update_atom(self, atom_id: str, **fields: Any) -> None:
    await self._ensure_initialized()
    if not fields:
      return
    hash_key = self._key("atoms", atom_id)
    exists = await self._client.exists(hash_key)
    if not exists:
      return
    mapping: dict = {}
    for key, value in fields.items():
      if key in ("embedding", "topics", "source_episode_ids"):
        mapping[key] = json.dumps(value) if value is not None else ""
      elif isinstance(value, (int, float)):
        mapping[key] = str(value)
      else:
        mapping[key] = value if value is not None else ""
    await self._client.hset(hash_key, mapping=mapping)

  async def prune_atoms(self, min_confidence: float) -> int:
    await self._ensure_initialized()
    sorted_key = self._key("atoms", "by_accessed")
    ids = await self._client.zrange(sorted_key, 0, -1)
    if not ids:
      return 0

    hash_keys = [self._key("atoms", aid) for aid in ids]
    raw_list = await self._fetch_hashes(hash_keys)

    deleted = 0
    for data in raw_list:
      atom = self._hash_to_atom(data)
      if atom.confidence < min_confidence:
        await self._client.delete(self._key("atoms", atom.id))
        await self._remove_from_atom_indexes(atom.id, atom.user_id)
        deleted += 1

    return deleted

  # ---------- Procedures ----------

  async def store_procedure(self, procedure: Procedure) -> str:
    await self._ensure_initialized()
    if not procedure.id:
      procedure.id = str(uuid4())
    now = time.time()
    if procedure.created_at == 0.0:
      procedure.created_at = now
    if procedure.last_accessed_at == 0.0:
      procedure.last_accessed_at = now

    hash_key = self._key("procedures", procedure.id)
    await self._client.hset(hash_key, mapping=self._procedure_to_hash(procedure))
    await self._add_to_procedure_indexes(procedure)
    return procedure.id

  async def get_procedures(
    self,
    *,
    user_id: Optional[str] = None,
    min_confidence: float = 0.3,
  ) -> List[Procedure]:
    await self._ensure_initialized()

    if user_id is not None:
      sorted_key = self._key("procedures", "user", user_id)
    else:
      sorted_key = self._key("procedures", "by_confidence")

    ids = await self._client.zrange(sorted_key, 0, -1)
    if not ids:
      return []

    hash_keys = [self._key("procedures", pid) for pid in ids]
    raw_list = await self._fetch_hashes(hash_keys)

    procedures: List[Procedure] = []
    for data in raw_list:
      proc = self._hash_to_procedure(data)
      if proc.confidence < min_confidence:
        continue
      if user_id is not None and proc.user_id != user_id:
        continue
      procedures.append(proc)

    procedures.sort(key=lambda p: p.confidence, reverse=True)
    return procedures

  async def find_similar_procedure(
    self,
    trigger: str,
    user_id: Optional[str] = None,
  ) -> Optional[Procedure]:
    await self._ensure_initialized()
    trigger_words = set(trigger.lower().split())

    # Get all procedures with a low confidence floor (same as SQLite impl)
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
    hash_key = self._key("procedures", procedure_id)
    exists = await self._client.exists(hash_key)
    if not exists:
      return
    mapping: dict = {}
    for key, value in fields.items():
      if isinstance(value, (int, float)):
        mapping[key] = str(value)
      else:
        mapping[key] = value if value is not None else ""
    await self._client.hset(hash_key, mapping=mapping)

  # ---------- Topics ----------

  async def store_topic_transition(
    self,
    from_topic: str,
    to_topic: str,
    user_id: Optional[str] = None,
  ) -> None:
    await self._ensure_initialized()
    uid_part = user_id or "_none_"
    transition_key = self._key("transitions", uid_part, from_topic, to_topic)
    index_key = self._key("transitions", "from", from_topic, "user", uid_part)

    exists = await self._client.exists(transition_key)
    if exists:
      await self._client.hincrby(transition_key, "count", 1)
    else:
      mapping = {
        "from_topic": from_topic,
        "to_topic": to_topic,
        "user_id": user_id or "",
        "count": "1",
      }
      await self._client.hset(transition_key, mapping=mapping)
      await self._client.sadd(index_key, transition_key)

  async def get_topic_transitions(
    self,
    from_topic: str,
    user_id: Optional[str] = None,
    min_count: int = 3,
  ) -> List[TopicTransition]:
    await self._ensure_initialized()
    uid_part = user_id or "_none_"
    index_key = self._key("transitions", "from", from_topic, "user", uid_part)

    members = await self._client.smembers(index_key)
    if not members:
      return []

    raw_list = await self._fetch_hashes(list(members))

    matching: List[TopicTransition] = []
    for data in raw_list:
      count = int(data.get("count", 0))
      if count < min_count:
        continue
      matching.append(
        TopicTransition(
          from_topic=data["from_topic"],
          to_topic=data["to_topic"],
          count=count,
          probability=0.0,  # computed below
        )
      )

    # Sort by count descending
    matching.sort(key=lambda t: t.count, reverse=True)

    # Compute probabilities
    total = sum(t.count for t in matching)
    for t in matching:
      t.probability = t.count / total if total > 0 else 0.0

    return matching

  # ---------- Vector Search ----------

  async def search_episodes_by_embedding(
    self,
    embedding: List[float],
    *,
    user_id: Optional[str] = None,
    top_k: int = 20,
  ) -> List[Episode]:
    await self._ensure_initialized()

    if user_id is not None:
      sorted_key = self._key("episodes", "user", user_id)
    else:
      sorted_key = self._key("episodes", "by_created")

    # Fetch up to 1000 recent IDs for scan
    ids = await self._client.zrevrange(sorted_key, 0, 999)
    if not ids:
      return []

    hash_keys = [self._key("episodes", eid) for eid in ids]
    raw_list = await self._fetch_hashes(hash_keys)

    scored: List[tuple] = []
    for data in raw_list:
      ep = self._hash_to_episode(data)
      if ep.embedding:
        if user_id is not None and ep.user_id != user_id:
          continue
        sim = cosine_similarity(embedding, ep.embedding)
        scored.append((sim, ep))

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

    if user_id is not None:
      sorted_key = self._key("atoms", "user", user_id)
    else:
      sorted_key = self._key("atoms", "by_accessed")

    ids = await self._client.zrevrange(sorted_key, 0, 999)
    if not ids:
      return []

    hash_keys = [self._key("atoms", aid) for aid in ids]
    raw_list = await self._fetch_hashes(hash_keys)

    scored: List[tuple] = []
    for data in raw_list:
      atom = self._hash_to_atom(data)
      if atom.embedding:
        if user_id is not None and atom.user_id != user_id:
          continue
        sim = cosine_similarity(embedding, atom.embedding)
        scored.append((sim, atom))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [atom for _, atom in scored[:top_k]]

  # ---------- Deletion ----------

  async def delete_user_data(self, user_id: str) -> None:
    await self._ensure_initialized()

    # Delete episodes for this user
    ep_index_key = self._key("episodes", "user", user_id)
    ep_ids = await self._client.zrange(ep_index_key, 0, -1)
    if ep_ids:
      # Get session_ids before deleting so we can clean session indexes
      hash_keys = [self._key("episodes", eid) for eid in ep_ids]
      raw_list = await self._fetch_hashes(hash_keys)
      pipe = self._client.pipeline()
      for data in raw_list:
        eid = data["id"]
        sid = data.get("session_id", "")
        pipe.delete(self._key("episodes", eid))
        pipe.zrem(self._key("episodes", "by_created"), eid)
        if sid:
          pipe.zrem(self._key("episodes", "session", sid), eid)
      pipe.delete(ep_index_key)
      await pipe.execute()

    # Delete atoms for this user
    atom_index_key = self._key("atoms", "user", user_id)
    atom_ids = await self._client.zrange(atom_index_key, 0, -1)
    if atom_ids:
      pipe = self._client.pipeline()
      for aid in atom_ids:
        pipe.delete(self._key("atoms", aid))
        pipe.zrem(self._key("atoms", "by_accessed"), aid)
      pipe.delete(atom_index_key)
      await pipe.execute()

    # Delete procedures for this user
    proc_index_key = self._key("procedures", "user", user_id)
    proc_ids = await self._client.zrange(proc_index_key, 0, -1)
    if proc_ids:
      pipe = self._client.pipeline()
      for pid in proc_ids:
        pipe.delete(self._key("procedures", pid))
        pipe.zrem(self._key("procedures", "by_confidence"), pid)
      pipe.delete(proc_index_key)
      await pipe.execute()

    # Delete topic transitions for this user
    uid_part = user_id
    pattern = self._key("transitions", uid_part, "*")
    keys_to_delete: List[str] = []
    async for key in self._client.scan_iter(match=pattern):
      keys_to_delete.append(key)
    if keys_to_delete:
      await self._client.delete(*keys_to_delete)

    # Clean up transition index sets
    index_pattern = self._key("transitions", "from", "*", "user", uid_part)
    index_keys: List[str] = []
    async for key in self._client.scan_iter(match=index_pattern):
      index_keys.append(key)
    if index_keys:
      await self._client.delete(*index_keys)

  async def delete_session_data(self, session_id: str) -> None:
    await self._ensure_initialized()
    session_key = self._key("episodes", "session", session_id)
    ep_ids = await self._client.zrange(session_key, 0, -1)
    if not ep_ids:
      return

    # Fetch episode data to get user_ids for index cleanup
    hash_keys = [self._key("episodes", eid) for eid in ep_ids]
    raw_list = await self._fetch_hashes(hash_keys)

    pipe = self._client.pipeline()
    for data in raw_list:
      eid = data["id"]
      uid = data.get("user_id") or None
      pipe.delete(self._key("episodes", eid))
      pipe.zrem(self._key("episodes", "by_created"), eid)
      if uid:
        pipe.zrem(self._key("episodes", "user", uid), eid)
    pipe.delete(session_key)
    await pipe.execute()

  # ---------- Context manager ----------

  async def __aenter__(self) -> "RedisMemoryStore":
    await self.initialize()
    return self

  async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    await self.close()
