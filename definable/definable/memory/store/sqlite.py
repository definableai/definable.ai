"""SQLite-backed memory store using aiosqlite."""

import json
import math
import time
from typing import Any, List, Optional
from uuid import uuid4

from definable.memory.types import Episode, KnowledgeAtom, Procedure, TopicTransition
from definable.utils.log import log_debug


def _cosine_similarity(a: List[float], b: List[float]) -> float:
  """Compute cosine similarity between two vectors."""
  if len(a) != len(b) or not a:
    return 0.0
  dot = sum(x * y for x, y in zip(a, b))
  mag_a = math.sqrt(sum(x * x for x in a))
  mag_b = math.sqrt(sum(x * x for x in b))
  if mag_a == 0.0 or mag_b == 0.0:
    return 0.0
  return dot / (mag_a * mag_b)


class SQLiteMemoryStore:
  """Async SQLite memory store.

  Tables are auto-created on first `initialize()` call.
  Vector search is done in Python (cosine similarity over recent entries).
  """

  def __init__(self, db_path: str = "./memory.db"):
    self.db_path = db_path
    self._db: Any = None
    self._initialized = False

  async def initialize(self) -> None:
    if self._initialized:
      return

    try:
      import aiosqlite
    except ImportError as e:
      raise ImportError("aiosqlite is required for SQLiteMemoryStore. Install it with: pip install aiosqlite") from e

    self._db = await aiosqlite.connect(self.db_path)
    self._db.row_factory = None  # Use tuple rows for efficiency
    await self._create_tables()
    self._initialized = True
    log_debug("SQLiteMemoryStore initialized", log_level=2)

  async def close(self) -> None:
    if self._db:
      await self._db.close()
      self._db = None
      self._initialized = False

  async def _ensure_initialized(self) -> None:
    if not self._initialized:
      await self.initialize()

  async def _create_tables(self) -> None:
    await self._db.executescript("""
      CREATE TABLE IF NOT EXISTS episodes (
        id TEXT PRIMARY KEY,
        user_id TEXT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        embedding TEXT,
        topics TEXT DEFAULT '[]',
        sentiment REAL DEFAULT 0.0,
        token_count INTEGER DEFAULT 0,
        compression_stage INTEGER DEFAULT 0,
        created_at REAL NOT NULL,
        last_accessed_at REAL NOT NULL,
        access_count INTEGER DEFAULT 0
      );

      CREATE INDEX IF NOT EXISTS idx_episodes_user_id ON episodes(user_id);
      CREATE INDEX IF NOT EXISTS idx_episodes_session_id ON episodes(session_id);
      CREATE INDEX IF NOT EXISTS idx_episodes_stage ON episodes(compression_stage);
      CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes(created_at);

      CREATE TABLE IF NOT EXISTS atoms (
        id TEXT PRIMARY KEY,
        user_id TEXT,
        subject TEXT NOT NULL,
        predicate TEXT NOT NULL,
        object TEXT NOT NULL,
        content TEXT NOT NULL,
        embedding TEXT,
        confidence REAL DEFAULT 1.0,
        reinforcement_count INTEGER DEFAULT 0,
        topics TEXT DEFAULT '[]',
        token_count INTEGER DEFAULT 0,
        source_episode_ids TEXT DEFAULT '[]',
        created_at REAL NOT NULL,
        last_accessed_at REAL NOT NULL,
        last_reinforced_at REAL DEFAULT 0.0,
        access_count INTEGER DEFAULT 0
      );

      CREATE INDEX IF NOT EXISTS idx_atoms_user_id ON atoms(user_id);
      CREATE INDEX IF NOT EXISTS idx_atoms_subject ON atoms(subject);
      CREATE INDEX IF NOT EXISTS idx_atoms_confidence ON atoms(confidence);

      CREATE TABLE IF NOT EXISTS procedures (
        id TEXT PRIMARY KEY,
        user_id TEXT,
        trigger_text TEXT NOT NULL,
        action TEXT NOT NULL,
        confidence REAL DEFAULT 0.5,
        observation_count INTEGER DEFAULT 1,
        created_at REAL NOT NULL,
        last_accessed_at REAL NOT NULL
      );

      CREATE INDEX IF NOT EXISTS idx_procedures_user_id ON procedures(user_id);

      CREATE TABLE IF NOT EXISTS topic_transitions (
        id TEXT PRIMARY KEY,
        user_id TEXT,
        from_topic TEXT NOT NULL,
        to_topic TEXT NOT NULL,
        count INTEGER DEFAULT 1
      );

      CREATE INDEX IF NOT EXISTS idx_transitions_from ON topic_transitions(from_topic);
      CREATE INDEX IF NOT EXISTS idx_transitions_user ON topic_transitions(user_id);
      CREATE UNIQUE INDEX IF NOT EXISTS idx_transitions_unique
        ON topic_transitions(user_id, from_topic, to_topic);
    """)
    await self._db.commit()

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

    embedding_json = json.dumps(episode.embedding) if episode.embedding else None
    topics_json = json.dumps(episode.topics)

    await self._db.execute(
      """INSERT INTO episodes (id, user_id, session_id, role, content, embedding, topics,
         sentiment, token_count, compression_stage, created_at, last_accessed_at, access_count)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
      (
        episode.id,
        episode.user_id,
        episode.session_id,
        episode.role,
        episode.content,
        embedding_json,
        topics_json,
        episode.sentiment,
        episode.token_count,
        episode.compression_stage,
        episode.created_at,
        episode.last_accessed_at,
        episode.access_count,
      ),
    )
    await self._db.commit()
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
    query = "SELECT * FROM episodes WHERE 1=1"
    params: List[Any] = []

    if user_id is not None:
      query += " AND user_id = ?"
      params.append(user_id)
    if session_id is not None:
      query += " AND session_id = ?"
      params.append(session_id)
    if min_stage is not None:
      query += " AND compression_stage >= ?"
      params.append(min_stage)
    if max_stage is not None:
      query += " AND compression_stage <= ?"
      params.append(max_stage)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    cursor = await self._db.execute(query, params)
    rows = await cursor.fetchall()
    return [self._row_to_episode(row) for row in rows]

  async def update_episode(self, episode_id: str, **fields) -> None:
    await self._ensure_initialized()
    if not fields:
      return
    set_clauses = []
    params: List[Any] = []
    for key, value in fields.items():
      if key in ("embedding", "topics", "source_episode_ids"):
        value = json.dumps(value) if value is not None else None
      set_clauses.append(f"{key} = ?")
      params.append(value)
    params.append(episode_id)
    await self._db.execute(
      f"UPDATE episodes SET {', '.join(set_clauses)} WHERE id = ?",
      params,
    )
    await self._db.commit()

  async def get_episodes_for_distillation(self, stage: int, older_than: float) -> List[Episode]:
    await self._ensure_initialized()
    cursor = await self._db.execute(
      "SELECT * FROM episodes WHERE compression_stage = ? AND created_at < ? ORDER BY created_at ASC LIMIT 50",
      (stage, older_than),
    )
    rows = await cursor.fetchall()
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

    embedding_json = json.dumps(atom.embedding) if atom.embedding else None
    topics_json = json.dumps(atom.topics)
    source_ids_json = json.dumps(atom.source_episode_ids)

    await self._db.execute(
      """INSERT INTO atoms (id, user_id, subject, predicate, object, content, embedding,
         confidence, reinforcement_count, topics, token_count, source_episode_ids,
         created_at, last_accessed_at, last_reinforced_at, access_count)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
      (
        atom.id,
        atom.user_id,
        atom.subject,
        atom.predicate,
        atom.object,
        atom.content,
        embedding_json,
        atom.confidence,
        atom.reinforcement_count,
        topics_json,
        atom.token_count,
        source_ids_json,
        atom.created_at,
        atom.last_accessed_at,
        atom.last_reinforced_at,
        atom.access_count,
      ),
    )
    await self._db.commit()
    return atom.id

  async def get_atoms(
    self,
    *,
    user_id: Optional[str] = None,
    min_confidence: float = 0.1,
    limit: int = 50,
  ) -> List[KnowledgeAtom]:
    await self._ensure_initialized()
    query = "SELECT * FROM atoms WHERE confidence >= ?"
    params: List[Any] = [min_confidence]

    if user_id is not None:
      query += " AND user_id = ?"
      params.append(user_id)

    query += " ORDER BY last_accessed_at DESC LIMIT ?"
    params.append(limit)

    cursor = await self._db.execute(query, params)
    rows = await cursor.fetchall()
    return [self._row_to_atom(row) for row in rows]

  async def find_similar_atom(
    self,
    subject: str,
    predicate: str,
    user_id: Optional[str] = None,
  ) -> Optional[KnowledgeAtom]:
    await self._ensure_initialized()
    query = "SELECT * FROM atoms WHERE LOWER(subject) = LOWER(?) AND LOWER(predicate) = LOWER(?)"
    params: List[Any] = [subject, predicate]
    if user_id is not None:
      query += " AND user_id = ?"
      params.append(user_id)
    query += " LIMIT 1"

    cursor = await self._db.execute(query, params)
    row = await cursor.fetchone()
    return self._row_to_atom(row) if row else None

  async def update_atom(self, atom_id: str, **fields) -> None:
    await self._ensure_initialized()
    if not fields:
      return
    set_clauses = []
    params: List[Any] = []
    for key, value in fields.items():
      if key in ("embedding", "topics", "source_episode_ids"):
        value = json.dumps(value) if value is not None else None
      set_clauses.append(f"{key} = ?")
      params.append(value)
    params.append(atom_id)
    await self._db.execute(
      f"UPDATE atoms SET {', '.join(set_clauses)} WHERE id = ?",
      params,
    )
    await self._db.commit()

  async def prune_atoms(self, min_confidence: float) -> int:
    await self._ensure_initialized()
    cursor = await self._db.execute(
      "DELETE FROM atoms WHERE confidence < ?",
      (min_confidence,),
    )
    await self._db.commit()
    return cursor.rowcount

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

    await self._db.execute(
      """INSERT INTO procedures (id, user_id, trigger_text, action, confidence,
         observation_count, created_at, last_accessed_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
      (
        procedure.id,
        procedure.user_id,
        procedure.trigger,
        procedure.action,
        procedure.confidence,
        procedure.observation_count,
        procedure.created_at,
        procedure.last_accessed_at,
      ),
    )
    await self._db.commit()
    return procedure.id

  async def get_procedures(
    self,
    *,
    user_id: Optional[str] = None,
    min_confidence: float = 0.3,
  ) -> List[Procedure]:
    await self._ensure_initialized()
    query = "SELECT * FROM procedures WHERE confidence >= ?"
    params: List[Any] = [min_confidence]

    if user_id is not None:
      query += " AND user_id = ?"
      params.append(user_id)

    query += " ORDER BY confidence DESC"

    cursor = await self._db.execute(query, params)
    rows = await cursor.fetchall()
    return [self._row_to_procedure(row) for row in rows]

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
    set_clauses = []
    params: List[Any] = []
    # Map 'trigger' field to 'trigger_text' column
    for key, value in fields.items():
      col_name = "trigger_text" if key == "trigger" else key
      set_clauses.append(f"{col_name} = ?")
      params.append(value)
    params.append(procedure_id)
    await self._db.execute(
      f"UPDATE procedures SET {', '.join(set_clauses)} WHERE id = ?",
      params,
    )
    await self._db.commit()

  # --- Topics ---

  async def store_topic_transition(
    self,
    from_topic: str,
    to_topic: str,
    user_id: Optional[str] = None,
  ) -> None:
    await self._ensure_initialized()
    # Upsert: increment count if exists, insert otherwise
    cursor = await self._db.execute(
      "SELECT id, count FROM topic_transitions WHERE from_topic = ? AND to_topic = ? AND user_id IS ?",
      (from_topic, to_topic, user_id),
    )
    row = await cursor.fetchone()

    if row:
      await self._db.execute(
        "UPDATE topic_transitions SET count = count + 1 WHERE id = ?",
        (row[0],),
      )
    else:
      await self._db.execute(
        "INSERT INTO topic_transitions (id, user_id, from_topic, to_topic, count) VALUES (?, ?, ?, ?, 1)",
        (str(uuid4()), user_id, from_topic, to_topic),
      )
    await self._db.commit()

  async def get_topic_transitions(
    self,
    from_topic: str,
    user_id: Optional[str] = None,
    min_count: int = 3,
  ) -> List[TopicTransition]:
    await self._ensure_initialized()
    query = "SELECT from_topic, to_topic, count FROM topic_transitions WHERE from_topic = ? AND count >= ?"
    params: List[Any] = [from_topic, min_count]

    if user_id is not None:
      query += " AND user_id = ?"
      params.append(user_id)

    query += " ORDER BY count DESC"

    cursor = await self._db.execute(query, params)
    rows = await cursor.fetchall()

    # Compute probabilities
    total = sum(row[2] for row in rows)
    transitions = []
    for row in rows:
      prob = row[2] / total if total > 0 else 0.0
      transitions.append(TopicTransition(from_topic=row[0], to_topic=row[1], count=row[2], probability=prob))
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
    query = "SELECT * FROM episodes WHERE embedding IS NOT NULL"
    params: List[Any] = []
    if user_id is not None:
      query += " AND user_id = ?"
      params.append(user_id)
    query += " ORDER BY created_at DESC LIMIT 1000"

    cursor = await self._db.execute(query, params)
    rows = await cursor.fetchall()

    scored = []
    for row in rows:
      episode = self._row_to_episode(row)
      if episode.embedding:
        sim = _cosine_similarity(embedding, episode.embedding)
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
    query = "SELECT * FROM atoms WHERE embedding IS NOT NULL"
    params: List[Any] = []
    if user_id is not None:
      query += " AND user_id = ?"
      params.append(user_id)
    query += " ORDER BY last_accessed_at DESC LIMIT 1000"

    cursor = await self._db.execute(query, params)
    rows = await cursor.fetchall()

    scored = []
    for row in rows:
      atom = self._row_to_atom(row)
      if atom.embedding:
        sim = _cosine_similarity(embedding, atom.embedding)
        scored.append((sim, atom))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [atom for _, atom in scored[:top_k]]

  # --- Deletion ---

  async def delete_user_data(self, user_id: str) -> None:
    await self._ensure_initialized()
    for table in ("episodes", "atoms", "procedures", "topic_transitions"):
      await self._db.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
    await self._db.commit()

  async def delete_session_data(self, session_id: str) -> None:
    await self._ensure_initialized()
    await self._db.execute("DELETE FROM episodes WHERE session_id = ?", (session_id,))
    await self._db.commit()

  # --- Row conversion helpers ---

  def _row_to_episode(self, row) -> Episode:
    return Episode(
      id=row[0],
      user_id=row[1],
      session_id=row[2],
      role=row[3],
      content=row[4],
      embedding=json.loads(row[5]) if row[5] else None,
      topics=json.loads(row[6]) if row[6] else [],
      sentiment=row[7] or 0.0,
      token_count=row[8] or 0,
      compression_stage=row[9] or 0,
      created_at=row[10] or 0.0,
      last_accessed_at=row[11] or 0.0,
      access_count=row[12] or 0,
    )

  def _row_to_atom(self, row) -> KnowledgeAtom:
    return KnowledgeAtom(
      id=row[0],
      user_id=row[1],
      subject=row[2],
      predicate=row[3],
      object=row[4],
      content=row[5],
      embedding=json.loads(row[6]) if row[6] else None,
      confidence=row[7] or 1.0,
      reinforcement_count=row[8] or 0,
      topics=json.loads(row[9]) if row[9] else [],
      token_count=row[10] or 0,
      source_episode_ids=json.loads(row[11]) if row[11] else [],
      created_at=row[12] or 0.0,
      last_accessed_at=row[13] or 0.0,
      last_reinforced_at=row[14] or 0.0,
      access_count=row[15] or 0,
    )

  def _row_to_procedure(self, row) -> Procedure:
    return Procedure(
      id=row[0],
      user_id=row[1],
      trigger=row[2],
      action=row[3],
      confidence=row[4] or 0.5,
      observation_count=row[5] or 1,
      created_at=row[6] or 0.0,
      last_accessed_at=row[7] or 0.0,
    )

  async def __aenter__(self):
    await self.initialize()
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    await self.close()
