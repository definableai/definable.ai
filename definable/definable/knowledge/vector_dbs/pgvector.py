"""PostgreSQL + pgvector database implementation."""
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import uuid4

from definable.knowledge.document import Document
from definable.knowledge.vector_dbs.base import VectorDB


@dataclass
class PgVectorDB(VectorDB):
  """PostgreSQL + pgvector implementation."""

  connection_string: Optional[str] = None
  table_name: str = "documents"

  # Lazy loaded connection
  _conn: Any = None

  @property
  def conn(self):
    """Get or create database connection."""
    if self._conn is None:
      try:
        import psycopg
        from pgvector.psycopg import register_vector
      except ImportError:
        raise ImportError(
          "psycopg and pgvector required. Run: pip install 'psycopg[binary]' pgvector"
        )

      if not self.connection_string:
        raise ValueError("connection_string is required")

      self._conn = psycopg.connect(self.connection_string)
      register_vector(self._conn)
      self._ensure_table()
    return self._conn

  def _ensure_table(self) -> None:
    """Create table if not exists."""
    with self.conn.cursor() as cur:
      # Enable pgvector extension
      cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

      # Create table
      cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
          id TEXT PRIMARY KEY,
          content TEXT NOT NULL,
          name TEXT,
          embedding vector({self.dimensions}),
          metadata JSONB DEFAULT '{{}}'::jsonb,
          source TEXT,
          source_type TEXT,
          chunk_index INTEGER,
          chunk_total INTEGER,
          parent_id TEXT,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
      """)

      # Create index for vector similarity search
      cur.execute(f"""
        CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx
        ON {self.table_name}
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
      """)
    self.conn.commit()

  def add(self, documents: List[Document]) -> List[str]:
    """Add documents to the vector store."""
    ids: List[str] = []
    with self.conn.cursor() as cur:
      for doc in documents:
        doc_id = doc.id or str(uuid4())
        doc.id = doc_id
        ids.append(doc_id)

        cur.execute(
          f"""
          INSERT INTO {self.table_name}
          (id, content, name, embedding, metadata, source, source_type, chunk_index, chunk_total, parent_id)
          VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
          ON CONFLICT (id) DO UPDATE SET
            content = EXCLUDED.content,
            name = EXCLUDED.name,
            embedding = EXCLUDED.embedding,
            metadata = EXCLUDED.metadata,
            source = EXCLUDED.source,
            source_type = EXCLUDED.source_type,
            chunk_index = EXCLUDED.chunk_index,
            chunk_total = EXCLUDED.chunk_total,
            parent_id = EXCLUDED.parent_id
          """,
          (
            doc_id,
            doc.content,
            doc.name,
            doc.embedding,
            json.dumps(doc.meta_data) if doc.meta_data else "{}",
            doc.source,
            doc.source_type,
            doc.chunk_index,
            doc.chunk_total,
            doc.parent_id,
          ),
        )
    self.conn.commit()
    return ids

  async def aadd(self, documents: List[Document]) -> List[str]:
    """Async add documents to the vector store."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, self.add, documents)

  def search(
    self,
    query_embedding: List[float],
    top_k: int = 10,
    filter: Optional[Dict[str, Any]] = None,  # noqa: A002
  ) -> List[Document]:
    """Search for similar documents by embedding vector."""
    # Build filter clause
    filter_clause = ""
    filter_params: List[Any] = []
    if filter:
      conditions = []
      for key, value in filter.items():
        conditions.append(f"metadata->>'{key}' = %s")
        filter_params.append(str(value))
      if conditions:
        filter_clause = "WHERE " + " AND ".join(conditions)

    with self.conn.cursor() as cur:
      cur.execute(
        f"""
        SELECT id, content, name, metadata, source, source_type,
               chunk_index, chunk_total, parent_id,
               1 - (embedding <=> %s::vector) as similarity
        FROM {self.table_name}
        {filter_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        [query_embedding, *filter_params, query_embedding, top_k],
      )

      results: List[Document] = []
      for row in cur.fetchall():
        doc = Document(
          id=row[0],
          content=row[1],
          name=row[2],
          meta_data=row[3] or {},
          source=row[4],
          source_type=row[5],
          chunk_index=row[6],
          chunk_total=row[7],
          parent_id=row[8],
          reranking_score=float(row[9]),
        )
        results.append(doc)

    return results

  async def asearch(
    self,
    query_embedding: List[float],
    top_k: int = 10,
    filter: Optional[Dict[str, Any]] = None,  # noqa: A002
  ) -> List[Document]:
    """Async search for similar documents."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, self.search, query_embedding, top_k, filter)

  def delete(self, ids: List[str]) -> None:
    """Delete documents by IDs."""
    with self.conn.cursor() as cur:
      cur.execute(
        f"DELETE FROM {self.table_name} WHERE id = ANY(%s)",
        (ids,),
      )
    self.conn.commit()

  async def adelete(self, ids: List[str]) -> None:
    """Async delete documents by IDs."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, self.delete, ids)

  def clear(self) -> None:
    """Clear all documents from the collection."""
    with self.conn.cursor() as cur:
      cur.execute(f"TRUNCATE TABLE {self.table_name}")
    self.conn.commit()

  def count(self) -> int:
    """Return number of documents in collection."""
    with self.conn.cursor() as cur:
      cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
      result = cur.fetchone()
      return result[0] if result else 0

  def close(self) -> None:
    """Close the database connection."""
    if self._conn:
      self._conn.close()
      self._conn = None
