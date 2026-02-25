"""Job stores — persistence for scheduled jobs."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Protocol, runtime_checkable

from definable.agent.scheduler.job import JobStatus, ScheduledJob


@runtime_checkable
class JobStore(Protocol):
  """Protocol for job persistence backends."""

  async def save(self, job: ScheduledJob) -> None:
    """Save or update a job."""
    ...

  async def get(self, job_id: str) -> Optional[ScheduledJob]:
    """Get a job by ID."""
    ...

  async def list_jobs(self, status: Optional[JobStatus] = None) -> List[ScheduledJob]:
    """List all jobs, optionally filtered by status."""
    ...

  async def delete(self, job_id: str) -> bool:
    """Delete a job by ID. Returns True if found and deleted."""
    ...


class InMemoryJobStore:
  """In-memory job store (non-persistent, good for testing).

  Example::

    store = InMemoryJobStore()
    await store.save(job)
    all_jobs = await store.list_jobs()
  """

  def __init__(self) -> None:
    self._jobs: Dict[str, ScheduledJob] = {}

  async def save(self, job: ScheduledJob) -> None:
    self._jobs[job.job_id] = job

  async def get(self, job_id: str) -> Optional[ScheduledJob]:
    return self._jobs.get(job_id)

  async def list_jobs(self, status: Optional[JobStatus] = None) -> List[ScheduledJob]:
    jobs = list(self._jobs.values())
    if status is not None:
      jobs = [j for j in jobs if j.status == status]
    return jobs

  async def delete(self, job_id: str) -> bool:
    return self._jobs.pop(job_id, None) is not None

  async def count(self, status: Optional[JobStatus] = None) -> int:
    if status is None:
      return len(self._jobs)
    return sum(1 for j in self._jobs.values() if j.status == status)


class SQLiteJobStore:
  """SQLite-backed job store for persistent scheduling.

  Stores job metadata (status, timing, run count, errors) in SQLite.
  Trigger objects are NOT serialized — they must be re-attached on load
  via :meth:`attach_trigger`.

  Args:
    db_path: Path to the SQLite database file.

  Example::

    store = SQLiteJobStore(".definable/scheduler.db")
    await store.initialize()
    await store.save(job)
  """

  def __init__(self, db_path: str = ".definable/scheduler.db") -> None:
    self._db_path = db_path
    self._initialized = False

  async def initialize(self) -> None:
    """Create the jobs table if it doesn't exist."""
    import aiosqlite

    async with aiosqlite.connect(self._db_path) as db:
      await db.execute("""
        CREATE TABLE IF NOT EXISTS scheduled_jobs (
          job_id TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'pending',
          max_runs INTEGER,
          metadata_json TEXT DEFAULT '{}',
          created_at REAL NOT NULL,
          next_run_at REAL NOT NULL,
          last_run_at REAL DEFAULT 0.0,
          run_count INTEGER DEFAULT 0,
          failure_count INTEGER DEFAULT 0,
          last_error TEXT,
          trigger_name TEXT NOT NULL
        )
      """)
      await db.commit()
    self._initialized = True

  async def _ensure_init(self) -> None:
    if not self._initialized:
      await self.initialize()

  async def save(self, job: ScheduledJob) -> None:
    import aiosqlite

    await self._ensure_init()
    async with aiosqlite.connect(self._db_path) as db:
      await db.execute(
        """
        INSERT OR REPLACE INTO scheduled_jobs
          (job_id, name, status, max_runs, metadata_json, created_at,
           next_run_at, last_run_at, run_count, failure_count, last_error, trigger_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
          job.job_id,
          job.name,
          job.status.value,
          job.max_runs,
          json.dumps(job.metadata),
          job.created_at,
          job.next_run_at,
          job.last_run_at,
          job.run_count,
          job.failure_count,
          job.last_error,
          job.trigger.name,
        ),
      )
      await db.commit()

  async def get(self, job_id: str) -> Optional[ScheduledJob]:
    """Get a job by ID.

    Note: Returns None if the job has no attached trigger.
    Use list_rows() + attach_trigger() for full reconstruction.
    """
    import aiosqlite

    await self._ensure_init()
    async with aiosqlite.connect(self._db_path) as db:
      db.row_factory = aiosqlite.Row
      cursor = await db.execute("SELECT * FROM scheduled_jobs WHERE job_id = ?", (job_id,))
      row = await cursor.fetchone()
      if row is None:
        return None
      return self._row_to_dict(row)  # type: ignore[return-value]

  async def list_jobs(self, status: Optional[JobStatus] = None) -> List[ScheduledJob]:
    """List jobs. Returns partial ScheduledJob-like dicts (no trigger)."""
    import aiosqlite

    await self._ensure_init()
    async with aiosqlite.connect(self._db_path) as db:
      db.row_factory = aiosqlite.Row
      if status is not None:
        cursor = await db.execute("SELECT * FROM scheduled_jobs WHERE status = ?", (status.value,))
      else:
        cursor = await db.execute("SELECT * FROM scheduled_jobs")
      rows = await cursor.fetchall()
      return [self._row_to_dict(r) for r in rows]  # type: ignore[misc]

  async def delete(self, job_id: str) -> bool:
    import aiosqlite

    await self._ensure_init()
    async with aiosqlite.connect(self._db_path) as db:
      cursor = await db.execute("DELETE FROM scheduled_jobs WHERE job_id = ?", (job_id,))
      await db.commit()
      return cursor.rowcount > 0  # type: ignore[return-value]

  async def list_rows(self) -> List[Dict]:
    """List raw row data (for reconstruction with triggers)."""
    import aiosqlite

    await self._ensure_init()
    async with aiosqlite.connect(self._db_path) as db:
      db.row_factory = aiosqlite.Row
      cursor = await db.execute("SELECT * FROM scheduled_jobs")
      rows = await cursor.fetchall()
      return [dict(r) for r in rows]

  @staticmethod
  def _row_to_dict(row) -> dict:
    """Convert a DB row to a dict with parsed fields."""
    return {
      "job_id": row["job_id"],
      "name": row["name"],
      "status": JobStatus(row["status"]),
      "max_runs": row["max_runs"],
      "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else {},
      "created_at": row["created_at"],
      "next_run_at": row["next_run_at"],
      "last_run_at": row["last_run_at"],
      "run_count": row["run_count"],
      "failure_count": row["failure_count"],
      "last_error": row["last_error"],
      "trigger_name": row["trigger_name"],
    }

  async def close(self) -> None:
    """No-op; connections are opened per-operation."""
    pass
