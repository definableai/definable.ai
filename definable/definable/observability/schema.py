"""Dataclass rows mapping 1:1 to the observability DB tables.

Used by :class:`definable.db.repo.Repo[T]` and the projection layer.
Field names and column names line up exactly with
``definable/db/migrations/observability/0001_init.sql``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, kw_only=True)
class AgentRow:
  id: str
  registered_at: float
  model: str
  instructions: str | None = None


@dataclass(frozen=True, kw_only=True)
class RunRow:
  id: str
  agent_id: str
  started_at: float
  ended_at: float | None = None
  status: str = "running"  # running | completed | errored
  turns: int = 0
  input: str | None = None
  output: str | None = None
  exit_reason: str | None = None
  error: str | None = None
  total_input_tokens: int = 0
  total_output_tokens: int = 0
  total_cached_tokens: int = 0
  total_cost_usd: float = 0.0


@dataclass(frozen=True, kw_only=True)
class EventRow:
  id: int | None = None  # populated by SQLite AUTOINCREMENT
  run_id: str
  timestamp: float
  type: str
  payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class SpanRow:
  id: int | None = None
  run_id: str
  parent_id: int | None = None
  kind: str  # llm | tool | memory
  name: str
  start_ts: float
  end_ts: float | None = None
  duration_ms: float | None = None
  status: str = "ok"  # ok | err
  metadata: dict[str, Any] = field(default_factory=dict)
