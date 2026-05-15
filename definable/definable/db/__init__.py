"""Generic async SQLite layer for SDK modules that need persistence.

Tiny by design: one connection per namespace, numbered SQL migrations,
typed `Repo[T]` over a frozen dataclass. No ORM magic, no relations, no
session pattern — write raw SQL through `Repo` when something fancier is
needed.

Each namespace gets its own file at ``.definable/<namespace>.db`` so two
modules cannot collide on schema. The first call to :func:`connect`
creates the file lazily; subsequent calls in the same process return the
same connection (aiosqlite serializes access).

Example::

    from definable.db import connect, migrate, Repo

    conn = await connect("observability")
    await migrate("observability", conn)
    runs = Repo(conn, table="runs", model=RunRow)
    await runs.insert(RunRow(id="r1", ...))
    row = await runs.get("r1")
"""

from __future__ import annotations

from definable.db.connection import Connection, close_all, connect
from definable.db.migrate import migrate
from definable.db.repo import Repo

__all__ = ["Connection", "Repo", "close_all", "connect", "migrate"]
