"""PostgreSQL + pgvector backed memory store using asyncpg."""

import json
import os
import time
from typing import Any, List, Optional
from uuid import uuid4

from definable.memory.types import Episode, KnowledgeAtom, Procedure, TopicTransition
from definable.utils.log import log_debug


def _embedding_to_pgvector(embedding: Optional[List[float]]) -> Optional[str]:
  """Convert a list of floats to pgvector literal format '[0.1,0.2,...]'."""
  if embedding is None:
    return None
  return "[" + ",".join(str(v) for v in embedding) + "]"


def _pgvector_to_embedding(value: Any) -> Optional[List[float]]:
  """Parse pgvector string back to list of floats. Handles None and already-parsed values."""
  if value is None:
    return None
  if isinstance(value, (list, tuple)):
    return [float(v) for v in value]
  if isinstance(value, str):
    cleaned = value.strip("[]")
    if not cleaned:
      return None
    return [float(v) for v in cleaned.split(",")]
  return None


class PostgresMemoryStore:
  """Async PostgreSQL + pgvector memory store.

  Uses asyncpg connection pooling. Tables are auto-created on first
  `initialize()` call. Vector search uses native pgvector <=> operator
  for cosine distance.
  """

  def __init__(self, db_url: str = "", pool_size: int = 5, table_prefix: str = "memory_"):
    self._db_url = db_url or os.environ.get("MEMORY_POSTGRES_URL", "")
    self._pool_size = pool_size
    self._prefix = table_prefix
    self._pool: Any = None
    self._initialized = False

  async def initialize(self) -> None:
    if self._initialized:
      return

    try:
      import asyncpg
    except ImportError as e:
      raise ImportError("asyncpg is required for PostgresMemoryStore. Install it with: pip install definable[postgres-memory]") from e

    if not self._db_url:
      raise ValueError("PostgreSQL connection URL is required. Set db_url or MEMORY_POSTGRES_URL environment variable.")

    self._pool = await asyncpg.create_pool(self._db_url, min_size=1, max_size=self._pool_size)
    await self._create_tables()
    self._initialized = True
    log_debug("PostgresMemoryStore initialized", log_level=2)

  async def close(self) -> None:
    if self._pool:
      await self._pool.close()
      self._pool = None
      self._initialized = False

  async def _ensure_initialized(self) -> None:
    if not self._initialized:
      await self.initialize()

  async def _create_tables(self) -> None:
    p = self._prefix
    async with self._pool.acquire() as conn:
      await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

      await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {p}episodes (
          id TEXT PRIMARY KEY,
          user_id TEXT,
          session_id TEXT NOT NULL,
          role TEXT NOT NULL,
          content TEXT NOT NULL,
          embedding VECTOR,
          topics JSONB DEFAULT '[]'::jsonb,
          sentiment DOUBLE PRECISION DEFAULT 0.0,
          token_count INTEGER DEFAULT 0,
          compression_stage INTEGER DEFAULT 0,
          created_at DOUBLE PRECISION NOT NULL,
          last_accessed_at DOUBLE PRECISION NOT NULL,
          access_count INTEGER DEFAULT 0
        )
      """)
      await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{p}episodes_user_id ON {p}episodes(user_id)")
      await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{p}episodes_session_id ON {p}episodes(session_id)")
      await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{p}episodes_stage ON {p}episodes(compression_stage)")
      await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{p}episodes_created ON {p}episodes(created_at)")

      # HNSW index on embedding for fast cosine search
      await conn.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{p}episodes_embedding
        ON {p}episodes USING hnsw (embedding vector_cosine_ops)
      """)

      await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {p}atoms (
          id TEXT PRIMARY KEY,
          user_id TEXT,
          subject TEXT NOT NULL,
          predicate TEXT NOT NULL,
          object TEXT NOT NULL,
          content TEXT NOT NULL,
          embedding VECTOR,
          confidence DOUBLE PRECISION DEFAULT 1.0,
          reinforcement_count INTEGER DEFAULT 0,
          topics JSONB DEFAULT '[]'::jsonb,
          token_count INTEGER DEFAULT 0,
          source_episode_ids JSONB DEFAULT '[]'::jsonb,
          created_at DOUBLE PRECISION NOT NULL,
          last_accessed_at DOUBLE PRECISION NOT NULL,
          last_reinforced_at DOUBLE PRECISION DEFAULT 0.0,
          access_count INTEGER DEFAULT 0
        )
      """)
      await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{p}atoms_user_id ON {p}atoms(user_id)")
      await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{p}atoms_subject ON {p}atoms(subject)")
      await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{p}atoms_confidence ON {p}atoms(confidence)")

      # HNSW index on embedding for fast cosine search
      await conn.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{p}atoms_embedding
        ON {p}atoms USING hnsw (embedding vector_cosine_ops)
      """)

      await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {p}procedures (
          id TEXT PRIMARY KEY,
          user_id TEXT,
          trigger_text TEXT NOT NULL,
          action TEXT NOT NULL,
          confidence DOUBLE PRECISION DEFAULT 0.5,
          observation_count INTEGER DEFAULT 1,
          created_at DOUBLE PRECISION NOT NULL,
          last_accessed_at DOUBLE PRECISION NOT NULL
        )
      """)
      await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{p}procedures_user_id ON {p}procedures(user_id)")

      await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {p}topic_transitions (
          id TEXT PRIMARY KEY,
          user_id TEXT,
          from_topic TEXT NOT NULL,
          to_topic TEXT NOT NULL,
          count INTEGER DEFAULT 1,
          UNIQUE (COALESCE(user_id, ''), from_topic, to_topic)
        )
      """)
      await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{p}transitions_from ON {p}topic_transitions(from_topic)")
      await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{p}transitions_user ON {p}topic_transitions(user_id)")

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

    embedding_str = _embedding_to_pgvector(episode.embedding)
    topics_json = json.dumps(episode.topics)

    await self._pool.execute(
      f"""INSERT INTO {self._prefix}episodes
         (id, user_id, session_id, role, content, embedding, topics,
          sentiment, token_count, compression_stage, created_at, last_accessed_at, access_count)
         VALUES ($1, $2, $3, $4, $5, $6::vector, $7::jsonb, $8, $9, $10, $11, $12, $13)""",
      episode.id,
      episode.user_id,
      episode.session_id,
      episode.role,
      episode.content,
      embedding_str,
      topics_json,
      episode.sentiment,
      episode.token_count,
      episode.compression_stage,
      episode.created_at,
      episode.last_accessed_at,
      episode.access_count,
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
    query = f"SELECT * FROM {self._prefix}episodes WHERE TRUE"
    params: List[Any] = []
    idx = 0

    if user_id is not None:
      idx += 1
      query += f" AND user_id = ${idx}"
      params.append(user_id)
    if session_id is not None:
      idx += 1
      query += f" AND session_id = ${idx}"
      params.append(session_id)
    if min_stage is not None:
      idx += 1
      query += f" AND compression_stage >= ${idx}"
      params.append(min_stage)
    if max_stage is not None:
      idx += 1
      query += f" AND compression_stage <= ${idx}"
      params.append(max_stage)

    idx += 1
    query += f" ORDER BY created_at DESC LIMIT ${idx}"
    params.append(limit)

    rows = await self._pool.fetch(query, *params)
    return [self._row_to_episode(row) for row in rows]

  async def update_episode(self, episode_id: str, **fields) -> None:
    await self._ensure_initialized()
    if not fields:
      return
    set_clauses = []
    params: List[Any] = []
    idx = 0
    for key, value in fields.items():
      if key in ("embedding",):
        value = _embedding_to_pgvector(value)
        idx += 1
        set_clauses.append(f"embedding = ${idx}::vector")
        params.append(value)
      elif key in ("topics", "source_episode_ids"):
        value = json.dumps(value) if value is not None else None
        idx += 1
        set_clauses.append(f"{key} = ${idx}::jsonb")
        params.append(value)
      else:
        idx += 1
        set_clauses.append(f"{key} = ${idx}")
        params.append(value)
    idx += 1
    params.append(episode_id)
    await self._pool.execute(
      f"UPDATE {self._prefix}episodes SET {', '.join(set_clauses)} WHERE id = ${idx}",
      *params,
    )

  async def get_episodes_for_distillation(self, stage: int, older_than: float) -> List[Episode]:
    await self._ensure_initialized()
    rows = await self._pool.fetch(
      f"SELECT * FROM {self._prefix}episodes WHERE compression_stage = $1 AND created_at < $2 ORDER BY created_at ASC LIMIT 50",
      stage,
      older_than,
    )
    return [self._row_to_episode(row) for row in rows]

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

    embedding_str = _embedding_to_pgvector(atom.embedding)
    topics_json = json.dumps(atom.topics)
    source_ids_json = json.dumps(atom.source_episode_ids)

    await self._pool.execute(
      f"""INSERT INTO {self._prefix}atoms
         (id, user_id, subject, predicate, object, content, embedding,
          confidence, reinforcement_count, topics, token_count, source_episode_ids,
          created_at, last_accessed_at, last_reinforced_at, access_count)
         VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8, $9, $10::jsonb, $11, $12::jsonb, $13, $14, $15, $16)""",
      atom.id,
      atom.user_id,
      atom.subject,
      atom.predicate,
      atom.object,
      atom.content,
      embedding_str,
      atom.confidence,
      atom.reinforcement_count,
      topics_json,
      atom.token_count,
      source_ids_json,
      atom.created_at,
      atom.last_accessed_at,
      atom.last_reinforced_at,
      atom.access_count,
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
    query = f"SELECT * FROM {self._prefix}atoms WHERE confidence >= $1"
    params: List[Any] = [min_confidence]
    idx = 1

    if user_id is not None:
      idx += 1
      query += f" AND user_id = ${idx}"
      params.append(user_id)

    idx += 1
    query += f" ORDER BY last_accessed_at DESC LIMIT ${idx}"
    params.append(limit)

    rows = await self._pool.fetch(query, *params)
    return [self._row_to_atom(row) for row in rows]

  async def find_similar_atom(
    self,
    subject: str,
    predicate: str,
    user_id: Optional[str] = None,
  ) -> Optional[KnowledgeAtom]:
    await self._ensure_initialized()
    query = f"SELECT * FROM {self._prefix}atoms WHERE LOWER(subject) = LOWER($1) AND LOWER(predicate) = LOWER($2)"
    params: List[Any] = [subject, predicate]
    idx = 2
    if user_id is not None:
      idx += 1
      query += f" AND user_id = ${idx}"
      params.append(user_id)
    query += " LIMIT 1"

    row = await self._pool.fetchrow(query, *params)
    return self._row_to_atom(row) if row else None

  async def update_atom(self, atom_id: str, **fields) -> None:
    await self._ensure_initialized()
    if not fields:
      return
    set_clauses = []
    params: List[Any] = []
    idx = 0
    for key, value in fields.items():
      if key in ("embedding",):
        value = _embedding_to_pgvector(value)
        idx += 1
        set_clauses.append(f"embedding = ${idx}::vector")
        params.append(value)
      elif key in ("topics", "source_episode_ids"):
        value = json.dumps(value) if value is not None else None
        idx += 1
        set_clauses.append(f"{key} = ${idx}::jsonb")
        params.append(value)
      else:
        idx += 1
        set_clauses.append(f"{key} = ${idx}")
        params.append(value)
    idx += 1
    params.append(atom_id)
    await self._pool.execute(
      f"UPDATE {self._prefix}atoms SET {', '.join(set_clauses)} WHERE id = ${idx}",
      *params,
    )

  async def prune_atoms(self, min_confidence: float) -> int:
    await self._ensure_initialized()
    result = await self._pool.execute(
      f"DELETE FROM {self._prefix}atoms WHERE confidence < $1",
      min_confidence,
    )
    # asyncpg returns status string like "DELETE 3"
    return int(result.split()[-1])

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

    await self._pool.execute(
      f"""INSERT INTO {self._prefix}procedures
         (id, user_id, trigger_text, action, confidence, observation_count, created_at, last_accessed_at)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8)""",
      procedure.id,
      procedure.user_id,
      procedure.trigger,
      procedure.action,
      procedure.confidence,
      procedure.observation_count,
      procedure.created_at,
      procedure.last_accessed_at,
    )
    return procedure.id

  async def get_procedures(
    self,
    *,
    user_id: Optional[str] = None,
    min_confidence: float = 0.3,
  ) -> List[Procedure]:
    await self._ensure_initialized()
    query = f"SELECT * FROM {self._prefix}procedures WHERE confidence >= $1"
    params: List[Any] = [min_confidence]
    idx = 1

    if user_id is not None:
      idx += 1
      query += f" AND user_id = ${idx}"
      params.append(user_id)

    query += " ORDER BY confidence DESC"

    rows = await self._pool.fetch(query, *params)
    return [self._row_to_procedure(row) for row in rows]

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
    set_clauses = []
    params: List[Any] = []
    idx = 0
    # Map 'trigger' field to 'trigger_text' column
    for key, value in fields.items():
      col_name = "trigger_text" if key == "trigger" else key
      idx += 1
      set_clauses.append(f"{col_name} = ${idx}")
      params.append(value)
    idx += 1
    params.append(procedure_id)
    await self._pool.execute(
      f"UPDATE {self._prefix}procedures SET {', '.join(set_clauses)} WHERE id = ${idx}",
      *params,
    )

  # --- Topics ---

  async def store_topic_transition(
    self,
    from_topic: str,
    to_topic: str,
    user_id: Optional[str] = None,
  ) -> None:
    await self._ensure_initialized()
    # Upsert: increment count if exists, insert otherwise
    # Use COALESCE to handle NULL user_id in the unique constraint
    await self._pool.execute(
      f"""INSERT INTO {self._prefix}topic_transitions (id, user_id, from_topic, to_topic, count)
         VALUES ($1, $2, $3, $4, 1)
         ON CONFLICT (COALESCE(user_id, ''), from_topic, to_topic)
         DO UPDATE SET count = {self._prefix}topic_transitions.count + 1""",
      str(uuid4()),
      user_id,
      from_topic,
      to_topic,
    )

  async def get_topic_transitions(
    self,
    from_topic: str,
    user_id: Optional[str] = None,
    min_count: int = 3,
  ) -> List[TopicTransition]:
    await self._ensure_initialized()
    query = f"SELECT from_topic, to_topic, count FROM {self._prefix}topic_transitions WHERE from_topic = $1 AND count >= $2"
    params: List[Any] = [from_topic, min_count]
    idx = 2

    if user_id is not None:
      idx += 1
      query += f" AND user_id = ${idx}"
      params.append(user_id)

    query += " ORDER BY count DESC"

    rows = await self._pool.fetch(query, *params)

    # Compute probabilities
    total = sum(row["count"] for row in rows)
    transitions = []
    for row in rows:
      prob = row["count"] / total if total > 0 else 0.0
      transitions.append(TopicTransition(from_topic=row["from_topic"], to_topic=row["to_topic"], count=row["count"], probability=prob))
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
    embedding_str = _embedding_to_pgvector(embedding)

    query = f"SELECT * FROM {self._prefix}episodes WHERE embedding IS NOT NULL"
    params: List[Any] = []
    idx = 0

    if user_id is not None:
      idx += 1
      query += f" AND user_id = ${idx}"
      params.append(user_id)

    idx += 1
    query += f" ORDER BY embedding <=> ${idx}::vector"
    params.append(embedding_str)

    idx += 1
    query += f" LIMIT ${idx}"
    params.append(top_k)

    rows = await self._pool.fetch(query, *params)
    return [self._row_to_episode(row) for row in rows]

  async def search_atoms_by_embedding(
    self,
    embedding: List[float],
    *,
    user_id: Optional[str] = None,
    top_k: int = 20,
  ) -> List[KnowledgeAtom]:
    await self._ensure_initialized()
    embedding_str = _embedding_to_pgvector(embedding)

    query = f"SELECT * FROM {self._prefix}atoms WHERE embedding IS NOT NULL"
    params: List[Any] = []
    idx = 0

    if user_id is not None:
      idx += 1
      query += f" AND user_id = ${idx}"
      params.append(user_id)

    idx += 1
    query += f" ORDER BY embedding <=> ${idx}::vector"
    params.append(embedding_str)

    idx += 1
    query += f" LIMIT ${idx}"
    params.append(top_k)

    rows = await self._pool.fetch(query, *params)
    return [self._row_to_atom(row) for row in rows]

  # --- Deletion ---

  async def delete_user_data(self, user_id: str) -> None:
    await self._ensure_initialized()
    p = self._prefix
    for table in (f"{p}episodes", f"{p}atoms", f"{p}procedures", f"{p}topic_transitions"):
      await self._pool.execute(f"DELETE FROM {table} WHERE user_id = $1", user_id)

  async def delete_session_data(self, session_id: str) -> None:
    await self._ensure_initialized()
    await self._pool.execute(f"DELETE FROM {self._prefix}episodes WHERE session_id = $1", session_id)

  # --- Row conversion helpers ---

  def _row_to_episode(self, row) -> Episode:
    return Episode(
      id=row["id"],
      user_id=row["user_id"],
      session_id=row["session_id"],
      role=row["role"],
      content=row["content"],
      embedding=_pgvector_to_embedding(row["embedding"]),
      topics=row["topics"] or [],
      sentiment=row["sentiment"] or 0.0,
      token_count=row["token_count"] or 0,
      compression_stage=row["compression_stage"] or 0,
      created_at=row["created_at"] or 0.0,
      last_accessed_at=row["last_accessed_at"] or 0.0,
      access_count=row["access_count"] or 0,
    )

  def _row_to_atom(self, row) -> KnowledgeAtom:
    return KnowledgeAtom(
      id=row["id"],
      user_id=row["user_id"],
      subject=row["subject"],
      predicate=row["predicate"],
      object=row["object"],
      content=row["content"],
      embedding=_pgvector_to_embedding(row["embedding"]),
      confidence=row["confidence"] or 1.0,
      reinforcement_count=row["reinforcement_count"] or 0,
      topics=row["topics"] or [],
      token_count=row["token_count"] or 0,
      source_episode_ids=row["source_episode_ids"] or [],
      created_at=row["created_at"] or 0.0,
      last_accessed_at=row["last_accessed_at"] or 0.0,
      last_reinforced_at=row["last_reinforced_at"] or 0.0,
      access_count=row["access_count"] or 0,
    )

  def _row_to_procedure(self, row) -> Procedure:
    return Procedure(
      id=row["id"],
      user_id=row["user_id"],
      trigger=row["trigger_text"],
      action=row["action"],
      confidence=row["confidence"] or 0.5,
      observation_count=row["observation_count"] or 1,
      created_at=row["created_at"] or 0.0,
      last_accessed_at=row["last_accessed_at"] or 0.0,
    )

  async def __aenter__(self):
    await self.initialize()
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    await self.close()
