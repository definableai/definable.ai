"""Cross-platform user identity resolution.

Maps (platform, platform_user_id) → canonical_user_id so that a single
user's memory is unified across platforms while sessions remain
platform-scoped.
"""

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, runtime_checkable

from definable.utils.log import log_debug


@dataclass
class PlatformIdentity:
  """A link between a platform-specific user ID and a canonical user ID.

  Attributes:
    platform: Platform name (e.g. "telegram", "discord").
    platform_user_id: User ID on the platform.
    canonical_user_id: Unified user ID used for memory.
    username: Optional display name.
    linked_at: Unix timestamp of when the link was created.
  """

  platform: str
  platform_user_id: str
  canonical_user_id: str
  username: Optional[str] = None
  linked_at: float = field(default_factory=time.time)


@runtime_checkable
class IdentityResolver(Protocol):
  """Protocol for resolving platform user IDs to canonical user IDs.

  Implementations must provide async methods for linking, resolving,
  and unlinking platform identities. Follows the same lifecycle
  pattern as MemoryStore (initialize/close, async context manager).
  """

  async def initialize(self) -> None: ...

  async def close(self) -> None: ...

  async def resolve(self, platform: str, platform_user_id: str) -> Optional[str]:
    """Resolve a platform user to a canonical user ID.

    Returns None if no link exists.
    """
    ...

  async def link(
    self,
    platform: str,
    platform_user_id: str,
    canonical_user_id: str,
    username: Optional[str] = None,
  ) -> None:
    """Create or update a link between a platform identity and a canonical user ID."""
    ...

  async def unlink(self, platform: str, platform_user_id: str) -> bool:
    """Remove a platform identity link. Returns True if a link was removed."""
    ...

  async def get_identities(self, canonical_user_id: str) -> List[PlatformIdentity]:
    """Get all platform identities linked to a canonical user ID."""
    ...


class SQLiteIdentityResolver:
  """SQLite-backed identity resolver using aiosqlite.

  Table is auto-created on first ``initialize()`` call.

  Args:
    db_path: Path to the SQLite database file.
  """

  def __init__(self, db_path: str = "./identity.db") -> None:
    self.db_path = db_path
    self._db: Any = None
    self._initialized = False

  async def initialize(self) -> None:
    if self._initialized:
      return

    try:
      import aiosqlite
    except ImportError as e:
      raise ImportError("aiosqlite is required for SQLiteIdentityResolver. Install it with: pip install aiosqlite") from e

    self._db = await aiosqlite.connect(self.db_path)
    self._db.row_factory = None
    await self._create_tables()
    self._initialized = True
    log_debug("SQLiteIdentityResolver initialized", log_level=2)

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
      CREATE TABLE IF NOT EXISTS identity_links (
        platform TEXT NOT NULL,
        platform_user_id TEXT NOT NULL,
        canonical_user_id TEXT NOT NULL,
        username TEXT,
        linked_at REAL NOT NULL,
        PRIMARY KEY (platform, platform_user_id)
      );

      CREATE INDEX IF NOT EXISTS idx_identity_canonical
        ON identity_links(canonical_user_id);
    """)
    await self._db.commit()

  async def resolve(self, platform: str, platform_user_id: str) -> Optional[str]:
    """Resolve a platform user to a canonical user ID.

    Returns None if no link exists.
    """
    await self._ensure_initialized()
    cursor = await self._db.execute(
      "SELECT canonical_user_id FROM identity_links WHERE platform = ? AND platform_user_id = ?",
      (platform, platform_user_id),
    )
    row = await cursor.fetchone()
    return row[0] if row else None

  async def link(
    self,
    platform: str,
    platform_user_id: str,
    canonical_user_id: str,
    username: Optional[str] = None,
  ) -> None:
    """Create or update a link between a platform identity and a canonical user ID."""
    await self._ensure_initialized()
    now = time.time()
    await self._db.execute(
      """INSERT INTO identity_links (platform, platform_user_id, canonical_user_id, username, linked_at)
         VALUES (?, ?, ?, ?, ?)
         ON CONFLICT (platform, platform_user_id)
         DO UPDATE SET canonical_user_id = excluded.canonical_user_id,
                       username = excluded.username,
                       linked_at = excluded.linked_at""",
      (platform, platform_user_id, canonical_user_id, username, now),
    )
    await self._db.commit()

  async def unlink(self, platform: str, platform_user_id: str) -> bool:
    """Remove a platform identity link. Returns True if a link was removed."""
    await self._ensure_initialized()
    cursor = await self._db.execute(
      "DELETE FROM identity_links WHERE platform = ? AND platform_user_id = ?",
      (platform, platform_user_id),
    )
    await self._db.commit()
    return cursor.rowcount > 0

  async def get_identities(self, canonical_user_id: str) -> List[PlatformIdentity]:
    """Get all platform identities linked to a canonical user ID."""
    await self._ensure_initialized()
    cursor = await self._db.execute(
      "SELECT platform, platform_user_id, canonical_user_id, username, linked_at "
      "FROM identity_links WHERE canonical_user_id = ? ORDER BY linked_at ASC",
      (canonical_user_id,),
    )
    rows = await cursor.fetchall()
    return [
      PlatformIdentity(
        platform=row[0],
        platform_user_id=row[1],
        canonical_user_id=row[2],
        username=row[3],
        linked_at=row[4],
      )
      for row in rows
    ]

  async def __aenter__(self) -> "SQLiteIdentityResolver":
    await self.initialize()
    return self

  async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    await self.close()
