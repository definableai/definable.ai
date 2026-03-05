"""Hierarchical tag index with DAG structure and prefix search.

Tags are path-like strings ("work/project/deadlines") that form a hierarchy.
The index supports prefix search (find all records tagged "work/*") and
exact tag lookup.
"""

from __future__ import annotations

from typing import Any, List, Optional, Set


class TagIndex:
  """SQLite-backed hierarchical tag index.

  Tables:
    cortex_tags: Maps record_id → tag (one row per tag per record).
  """

  def __init__(self, db: Any = None, separator: str = "/"):
    self._db = db
    self._separator = separator
    self._initialized = False

  async def initialize(self, db: Any) -> None:
    """Initialize with a shared aiosqlite connection."""
    self._db = db
    assert self._db is not None
    await self._db.executescript("""
      CREATE TABLE IF NOT EXISTS cortex_tags (
        record_id TEXT NOT NULL,
        tag TEXT NOT NULL,
        PRIMARY KEY (record_id, tag)
      );

      CREATE INDEX IF NOT EXISTS idx_cortex_tags_tag ON cortex_tags(tag);
      CREATE INDEX IF NOT EXISTS idx_cortex_tags_record ON cortex_tags(record_id);
    """)
    await self._db.commit()
    self._initialized = True

  async def add_tags(self, record_id: str, tags: List[str]) -> None:
    """Add tags for a record. Also adds parent tags in the hierarchy.

    For tag "work/project/deadlines", adds:
      - "work"
      - "work/project"
      - "work/project/deadlines"
    """
    assert self._db is not None
    all_tags = self._expand_hierarchy(tags)
    for tag in all_tags:
      await self._db.execute(
        "INSERT OR IGNORE INTO cortex_tags (record_id, tag) VALUES (?, ?)",
        (record_id, tag),
      )
    await self._db.commit()

  async def remove_tags(self, record_id: str, tags: Optional[List[str]] = None) -> None:
    """Remove tags for a record. None = remove all tags."""
    assert self._db is not None
    if tags is None:
      await self._db.execute("DELETE FROM cortex_tags WHERE record_id = ?", (record_id,))
    else:
      for tag in tags:
        await self._db.execute("DELETE FROM cortex_tags WHERE record_id = ? AND tag = ?", (record_id, tag))
    await self._db.commit()

  async def search_exact(self, tag: str) -> List[str]:
    """Find all record IDs with an exact tag match."""
    assert self._db is not None
    cursor = await self._db.execute("SELECT record_id FROM cortex_tags WHERE tag = ?", (tag,))
    rows = await cursor.fetchall()
    return [row[0] for row in rows]

  async def search_prefix(self, prefix: str) -> List[str]:
    """Find all record IDs with tags matching a prefix.

    search_prefix("work") matches "work", "work/project", "work/project/deadlines".
    """
    assert self._db is not None
    cursor = await self._db.execute(
      "SELECT DISTINCT record_id FROM cortex_tags WHERE tag = ? OR tag LIKE ?",
      (prefix, prefix + self._separator + "%"),
    )
    rows = await cursor.fetchall()
    return [row[0] for row in rows]

  async def get_tags(self, record_id: str) -> List[str]:
    """Get all tags for a record."""
    assert self._db is not None
    cursor = await self._db.execute("SELECT tag FROM cortex_tags WHERE record_id = ? ORDER BY tag", (record_id,))
    rows = await cursor.fetchall()
    return [row[0] for row in rows]

  async def get_all_tags(self) -> List[str]:
    """Get all unique tags in the index."""
    assert self._db is not None
    cursor = await self._db.execute("SELECT DISTINCT tag FROM cortex_tags ORDER BY tag")
    rows = await cursor.fetchall()
    return [row[0] for row in rows]

  async def count_by_tag(self, tag: str) -> int:
    """Count records with a given tag."""
    assert self._db is not None
    cursor = await self._db.execute("SELECT COUNT(*) FROM cortex_tags WHERE tag = ?", (tag,))
    row = await cursor.fetchone()
    return row[0] if row else 0

  def _expand_hierarchy(self, tags: List[str]) -> Set[str]:
    """Expand tags into full hierarchy.

    "work/project/deadlines" → {"work", "work/project", "work/project/deadlines"}
    """
    result: Set[str] = set()
    for tag in tags:
      parts = tag.split(self._separator)
      for i in range(1, len(parts) + 1):
        result.add(self._separator.join(parts[:i]))
    return result

  async def close(self) -> None:
    """No-op — db lifecycle managed by CortexStore."""
    pass
