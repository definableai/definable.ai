"""FTS5 full-text search index backed by SQLite.

Provides keyword-based document search as a complement to vector search.
Used standalone or as part of the hybrid search pipeline.

Usage::

    from definable.knowledge.fts import FTSIndex

    fts = FTSIndex()
    await fts.initialize()
    await fts.add("hash123", documents)
    results = await fts.search("machine learning", limit=10)
    await fts.close()
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from definable.knowledge.document import Document
from definable.knowledge.fts.keywords import extract_keywords, keywords_to_fts5_query
from definable.utils.log import log_debug, log_warning


@dataclass
class FTSIndex:
  """SQLite FTS5-based full-text search index.

  Stores document content in an FTS5 virtual table for keyword search.
  Supports add, search, delete, and clear operations.

  Attributes:
    db_path: Path to the SQLite database. Defaults to ``:memory:``.
    table_name: Name of the FTS5 virtual table.
  """

  db_path: Optional[str] = None
  table_name: str = "fts_documents"
  _db: Optional[sqlite3.Connection] = field(default=None, repr=False)
  _initialized: bool = field(default=False, repr=False)

  async def initialize(self) -> None:
    """Create the FTS5 table if it doesn't exist."""
    if self._initialized:
      return

    path = self.db_path or ":memory:"
    if path != ":memory:":
      Path(path).parent.mkdir(parents=True, exist_ok=True)

    self._db = sqlite3.connect(path, check_same_thread=False)
    self._db.execute("PRAGMA journal_mode=WAL")

    # Check FTS5 support
    try:
      self._db.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {self.table_name} USING fts5(
          doc_id UNINDEXED,
          content_hash UNINDEXED,
          name UNINDEXED,
          source UNINDEXED,
          content,
          tokenize = 'unicode61'
        )
      """)
    except sqlite3.OperationalError as exc:
      if "fts5" in str(exc).lower():
        raise RuntimeError("SQLite FTS5 extension is not available. Please install a Python build with FTS5 support.") from exc
      raise

    self._db.commit()
    self._initialized = True
    log_debug(f"FTSIndex initialized at {path}")

  def _require_db(self) -> sqlite3.Connection:
    if self._db is None:
      raise RuntimeError("FTSIndex not initialized. Call await fts.initialize() first.")
    return self._db

  async def add(
    self,
    content_hash: str,
    documents: List[Document],
  ) -> int:
    """Add documents to the FTS index.

    Args:
      content_hash: Hash identifying this batch of documents.
      documents: Documents to index.

    Returns:
      Number of documents added.
    """
    db = self._require_db()
    count = 0
    for doc in documents:
      if not doc.content:
        continue
      db.execute(
        f"INSERT INTO {self.table_name} (doc_id, content_hash, name, source, content) VALUES (?, ?, ?, ?, ?)",
        (doc.id or "", content_hash, doc.name or "", doc.source or "", doc.content),
      )
      count += 1
    db.commit()
    return count

  async def search(
    self,
    query: str,
    limit: int = 20,
  ) -> List[Tuple[str, float, str]]:
    """Search the FTS index.

    Args:
      query: Natural language search query.
      limit: Maximum results to return.

    Returns:
      List of (doc_id, bm25_score, content) tuples, ordered by relevance.
    """
    db = self._require_db()

    # Extract keywords and build FTS5 query
    keywords = extract_keywords(query)
    if not keywords:
      # Fall back to raw query
      fts_query = query.replace('"', '""')
      fts_query = f'"{fts_query}"'
    else:
      fts_query = keywords_to_fts5_query(keywords)

    if not fts_query:
      return []

    try:
      cursor = db.execute(
        f"SELECT doc_id, rank, content FROM {self.table_name} WHERE {self.table_name} MATCH ? ORDER BY rank LIMIT ?",
        (fts_query, limit),
      )
      rows = cursor.fetchall()
    except sqlite3.OperationalError as exc:
      log_warning(f"FTS search error: {exc}")
      return []

    # Convert BM25 rank to score (rank is negative, lower = better)
    results = []
    for doc_id, rank, content in rows:
      score = 1.0 / (1.0 + abs(rank))
      results.append((doc_id, score, content))

    return results

  async def search_documents(
    self,
    query: str,
    limit: int = 20,
  ) -> List[Document]:
    """Search and return full Document objects.

    Args:
      query: Natural language search query.
      limit: Maximum results to return.

    Returns:
      List of Documents with reranking_score set to BM25 score.
    """
    raw_results = await self.search(query, limit=limit)
    documents = []
    for doc_id, score, content in raw_results:
      doc = Document(
        id=doc_id or None,
        content=content,
        reranking_score=score,
      )
      documents.append(doc)
    return documents

  async def delete(self, content_hash: str) -> int:
    """Delete documents by content hash.

    Returns:
      Number of documents deleted.
    """
    db = self._require_db()
    cursor = db.execute(
      f"DELETE FROM {self.table_name} WHERE content_hash = ?",
      (content_hash,),
    )
    db.commit()
    return cursor.rowcount

  async def clear(self) -> None:
    """Remove all documents from the index."""
    db = self._require_db()
    db.execute(f"DELETE FROM {self.table_name}")
    db.commit()

  async def count(self) -> int:
    """Return the total number of indexed documents."""
    db = self._require_db()
    cursor = db.execute(f"SELECT COUNT(*) FROM {self.table_name}")
    return cursor.fetchone()[0]

  async def close(self) -> None:
    """Close the database connection."""
    if self._db:
      self._db.close()
      self._db = None
      self._initialized = False
