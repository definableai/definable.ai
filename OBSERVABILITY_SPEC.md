# Spec: Observability Dashboard + Generic DB Module

**Status**: draft, awaiting approval
**Branch**: `feat/harness-v2`
**Author**: hash
**Date**: 2026-05-13

> Channel rename SPEC (`/SPEC.md`) is paused. This spec is independent and lands first.

---

## 1. Objective

Build a live observability dashboard that ships with the SDK and is wired to every `Agent` instance when `observability=True`. The dashboard surfaces every event the harness already emits (tool calls, LLM calls, memory ops, run lifecycle) as a browsable UI: trace list, trace detail with waterfall, live metrics, and a fully functional **playground** to chat with the running agent and see tokens / cost / latency per turn.

The SDK gets a small, generic, durable DB layer at `definable.db` so observability — and any future module — can persist relational data without hand-rolling SQL. Goal: one SDK process opens one dashboard, all agents in that process register, all their runs/events stream live and persist for later replay.

**Users**

- SDK consumers debugging an agent locally: see what the agent did, why it failed, what it cost.
- Same consumers want a playground to iterate on prompts/tools/skills without leaving the browser.
- Future SDK module authors who need persistent local storage and pick `definable.db` instead of hand-rolling SQLite.

**Success looks like**

- `Agent(name="...", model="...", observability=True)` → terminal prints `[definable] observability live → http://127.0.0.1:7777`. Browser open → real dashboard.
- Every `agent.arun(...)` call shows up live in TRACES tab within ≤200 ms.
- Playground in browser invokes the **same Agent instance** the user constructed; events stream back via SSE; per-turn token/cost/duration shown.
- One `definable.db` API used by `observability` cleanly; future modules can claim a namespace with one call.

---

## 2. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Server | FastAPI + uvicorn | Already in `[serve]` extra. WebSocket+SSE+REST clean. No new dep. |
| DB driver | `aiosqlite>=0.20.0` | Already pinned. Async, file-backed, zero ops. |
| Schema layer | hand-rolled async repo + dataclass rows | No SQLAlchemy/SQLModel — keeps surface area small, matches harness-v2 "no magic" stance. Migrations via numbered SQL files. |
| Streaming | SSE for server→client event tail; REST POST for playground submit (returns SSE) | One-way is enough; SSE has auto-reconnect, survives proxies, simpler than WS in FastAPI (recall MCP+WS gotcha from harness-v2 smoke #163). |
| Frontend | React 18 via CDN + Babel-standalone (matches mock) | Zero build step. `index.html` + `.jsx` files load directly. Mock already in this shape. |
| Cost calc | `definable/model_pricing.json` (existing) | Already on disk; provider × model → `input_per_million` / `output_per_million` / `cached_input_per_million`. |

No new third-party dependencies. Everything either already pinned or stdlib.

---

## 3. Commands

```bash
# Quality gates (run before any commit)
.venv/bin/ruff check definable/definable/db definable/definable/observability
.venv/bin/ruff format --check definable/definable/db definable/definable/observability
.venv/bin/mypy definable/definable/db definable/definable/observability
.venv/bin/python -m pytest definable/tests/db definable/tests/observability -v

# Run a smoke agent with dashboard live
.venv/bin/python -m definable.observability.examples.basic_agent

# Manual: open dashboard
open http://127.0.0.1:7777
```

---

## 4. Project Structure

```
definable/definable/
├── db/                              # NEW — generic SDK DB layer
│   ├── __init__.py                  # public API: connect(), Repo, migrate()
│   ├── connection.py                # ConnectionManager — one aiosqlite conn per namespace, WAL mode, pooled
│   ├── repo.py                      # Repo[T] — typed CRUD over a dataclass + table name
│   ├── migrations.py                # Migration runner — scans `migrations/<namespace>/NNNN_*.sql`, applies in order, records in schema_migrations
│   ├── types.py                     # JSON column adapter (json.dumps/loads), datetime adapter
│   └── migrations/
│       └── observability/
│           ├── 0001_init.sql        # agents, runs, events, spans tables + indexes
│           └── (future numbered files)
│
├── observability/
│   ├── __init__.py                  # exports: Observability, attach_jsonl, ObservabilityServer
│   ├── subscriber.py                # EXISTING JSONL writer — kept, unchanged
│   ├── server.py                    # NEW — ObservabilityServer: FastAPI app, process-wide singleton, port discovery
│   ├── registry.py                  # NEW — AgentRegistry: tracks all Agent instances that opted in this process
│   ├── store.py                     # NEW — TraceStore: wraps db.Repo, writes events/runs/spans, queries for views
│   ├── pricing.py                   # NEW — Pricing.load() reads model_pricing.json, .cost(provider, model, in, out, cached) → USD
│   ├── projection.py                # NEW — projects Event stream into Run + Span rows (waterfall-ready)
│   ├── routes.py                    # NEW — REST + SSE handlers
│   ├── static/
│   │   ├── index.html               # was `Definable SDK Console.html`, retargeted to relative `/static/...`
│   │   └── console/
│   │       ├── styles.css           # KEPT from mock
│   │       ├── api.js               # NEW — real REST/SSE client, replaces data.js
│   │       ├── components.jsx       # KEPT
│   │       ├── app.jsx              # ADAPTED — drop FLEET+EVALS tabs; sidebar reads /api/agents
│   │       ├── views_playground.jsx # SPLIT from views_fleet_pg.jsx — wired to /api/playground/run
│   │       ├── views_traces.jsx     # SPLIT from views_obs.jsx — wired to /api/runs + /api/runs/{id}
│   │       └── views_metrics.jsx    # SPLIT from views_obs.jsx — wired to /api/metrics
│   └── examples/
│       └── basic_agent.py           # NEW — smoke runnable: opens dashboard, prints URL
│
└── (existing tree untouched)

definable/tests/
├── db/
│   ├── test_connection.py
│   ├── test_repo.py
│   └── test_migrations.py
└── observability/
    ├── test_pricing.py
    ├── test_projection.py
    ├── test_store.py
    ├── test_routes_rest.py
    └── test_routes_sse.py
```

**Deleted from mock**:
- `views_fleet_pg.jsx::FleetTab` (no fleet view in v1 — single-process scope; fleet page would lie)
- `views_obs.jsx::EvalsTab` (no eval module post harness-v2)
- `data.js` (all mock data — replaced by `api.js` REST/SSE client)

**Workspace layout**:
- `.definable/observability.db` — SQLite file, WAL mode, persists across runs
- `.definable/traces/<agent>.jsonl` — kept (existing JSONL writer untouched, runs alongside DB writer)

---

## 5. Code Style

Follows `framework-patterns` skill: 2-space indent, 150 char line, ruff+mypy clean, async-first, dataclasses for value types, no magic.

```python
# definable/db/repo.py — example of the style
from __future__ import annotations

import dataclasses
import json
import sqlite3
from typing import Any, Generic, Type, TypeVar

import aiosqlite

T = TypeVar("T")


class Repo(Generic[T]):
  """Typed CRUD over a dataclass `T` bound to `table`.

  No ORM magic. No relations. Insert / get / update / list / delete.
  Caller writes SQL for anything fancier.
  """

  def __init__(self, conn: aiosqlite.Connection, table: str, model: Type[T]) -> None:
    self._conn = conn
    self._table = table
    self._model = model
    self._fields = [f.name for f in dataclasses.fields(model)]  # type: ignore[arg-type]

  async def insert(self, row: T) -> None:
    cols = ", ".join(self._fields)
    placeholders = ", ".join("?" * len(self._fields))
    values = tuple(_encode(getattr(row, f)) for f in self._fields)
    await self._conn.execute(
      f"INSERT INTO {self._table} ({cols}) VALUES ({placeholders})", values
    )

  async def get(self, pk: Any, pk_col: str = "id") -> T | None:
    async with self._conn.execute(
      f"SELECT * FROM {self._table} WHERE {pk_col} = ?", (pk,)
    ) as cur:
      row = await cur.fetchone()
    return self._model(**_decode_row(row, cur.description)) if row else None


def _encode(v: Any) -> Any:
  if isinstance(v, (dict, list)):
    return json.dumps(v, default=str)
  return v
```

**Conventions**

- Module-level public surface in `__init__.py` only. No deep imports from consumers.
- All async. No sync sqlite calls in hot path.
- Dataclasses (`@dataclass(frozen=True, kw_only=True)`) for rows + DTOs.
- JSON columns serialized at insert, deserialized at fetch — single column adapter, not per-call.
- Comments: only when WHY is non-obvious. No "what" comments.
- Frontend `.jsx` files keep brutalist black/lime aesthetic from mock — design system stays, only data sources change.

---

## 6. Testing Strategy

| Layer | Where | Framework | Coverage target |
|---|---|---|---|
| `definable.db` unit | `definable/tests/db/` | pytest + pytest-asyncio | Every public API, in-memory `:memory:` DB |
| `definable.observability` unit | `definable/tests/observability/` | same | Pricing, projection, store, routes (TestClient) |
| Smoke | `smoke/observability/` | hand-rolled (matches harness-v2 smoke layout) | One end-to-end: spawn agent, hit `/api/runs`, drive playground via REST+SSE, verify event order |
| Frontend | none for v1 | — | Manual: open browser, click through all 3 tabs, run a playground turn, check live update on TRACES |

**MockModel hygiene** — reuse `definable/tests/_helpers/mock_model.py` (existing). DB tests use `:memory:` SQLite (no fs touch). Server tests use `httpx.AsyncClient(app=app)` (no real port bind).

**Pre-merge gates** (all four must be green):

1. `ruff check` on `definable/definable/db definable/definable/observability`
2. `ruff format --check` same scope
3. `mypy` same scope (clean = 0 errors)
4. `pytest definable/tests/db definable/tests/observability` — all green, no skips except known optional-dep marks

---

## 7. Boundaries

### Always do
- Run all 4 quality gates on every changed file before reporting done.
- Use `definable.utils.workspace.workspace_path("observability.db")` for DB location.
- Bind dashboard to `127.0.0.1` only — never `0.0.0.0`.
- Use prepared statements (no f-string SQL with user data).
- Persist every event via JSONL writer **and** DB writer in parallel — JSONL stays the durable wall in case DB layer is buggy in early phases.
- Use existing event types from `definable/agent/core/events.py` — don't invent new ones.
- Reuse the brutalist design system from `sample_ui/console/styles.css` verbatim.

### Ask first
- Adding a new third-party dep (current target: zero new deps).
- Changing the public `Agent(observability=...)` kwarg signature.
- Touching `definable/agent/agent.py` beyond the one-line swap from `Observability` (JSONL only) to `Observability` + `ObservabilityServer` attach.
- Frontend build step (still want zero-build CDN React).
- Pushing the listen port off `127.0.0.1`.

### Never do
- Auto-open browser. Print URL only.
- Block agent loop on DB writes — all DB writes are fire-and-forget via background task; if DB lags or crashes, agent runs continue.
- Hand-roll SQL inside `observability/` modules — go through `db.Repo` or named queries in `db/queries/`.
- Run any DB migration on import. Migrations run when the server starts (`await observability.aopen()`).
- Ship mock/sample data in production code paths. `data.js` is deleted, not preserved.

---

## 8. Architecture

### Wire diagram

```
Agent(observability=True) ──┐
Agent(observability=True) ──┼──→ AgentRegistry (process-wide singleton)
Agent(observability=True) ──┘             │
                                          │ registers each Agent's EventBus
                                          ▼
                                   ObservabilityServer (FastAPI, port :7777)
                                          │
       ┌──────────────────────────────────┼──────────────────────────────────┐
       │                                  │                                  │
       ▼                                  ▼                                  ▼
  TraceStore                       SSE event tail                     Playground POST
  (db.Repo over                    /api/stream                        /api/playground/run
   observability.db)               ↓                                  ↓
       │                           every Event from any              calls agent.arun()
       │                           registered EventBus               on the actual Agent
       ▼                           is fan-out broadcast              instance, streams
  Future:                          to all SSE subscribers            events back via SSE
  agent A's @bus.on()              (per-agent filter via             on the same socket
  + JSONL writer                   query param)
  + DB writer
  + SSE broadcast
```

### Lifecycle

1. First `Agent(observability=True)` constructed → `ObservabilityServer.singleton()` lazily binds free port (default 7777, increments on collision), prints URL, starts uvicorn task.
2. Subsequent agents attach to the same singleton — no new ports.
3. Each agent's `EventBus` gets a subscriber that: (a) inserts into `events` table, (b) appends to JSONL, (c) broadcasts to in-memory SSE fan-out.
4. Process exits → singleton sets a shutdown flag; uvicorn task gets cancelled in `atexit` hook; DB connection closed cleanly. WAL flushed.

### DB schema (migrations/observability/0001_init.sql)

```sql
CREATE TABLE schema_migrations (
  namespace TEXT, version INTEGER, applied_at REAL,
  PRIMARY KEY (namespace, version)
);

CREATE TABLE agents (
  id TEXT PRIMARY KEY,           -- agent.name (unique within process)
  registered_at REAL NOT NULL,
  model TEXT NOT NULL,
  instructions TEXT
);

CREATE TABLE runs (
  id TEXT PRIMARY KEY,            -- run_id (uuid)
  agent_id TEXT NOT NULL REFERENCES agents(id),
  started_at REAL NOT NULL,
  ended_at REAL,
  status TEXT NOT NULL,           -- running | completed | errored
  turns INTEGER DEFAULT 0,
  input TEXT,
  output TEXT,
  exit_reason TEXT,
  error TEXT,
  total_input_tokens INTEGER DEFAULT 0,
  total_output_tokens INTEGER DEFAULT 0,
  total_cached_tokens INTEGER DEFAULT 0,
  total_cost_usd REAL DEFAULT 0
);
CREATE INDEX idx_runs_agent_started ON runs (agent_id, started_at DESC);

CREATE TABLE events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL REFERENCES runs(id),
  timestamp REAL NOT NULL,
  type TEXT NOT NULL,             -- TurnStarted / ModelResponded / ToolCallStarted / ToolCallCompleted / ToolCallFailed / MemoryAccessed / StreamChunkEvent / RunCompleted / RunErrored
  payload TEXT NOT NULL           -- JSON of full event dataclass
);
CREATE INDEX idx_events_run_ts ON events (run_id, timestamp);

CREATE TABLE spans (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL REFERENCES runs(id),
  parent_id INTEGER,
  kind TEXT NOT NULL,             -- llm | tool | memory
  name TEXT NOT NULL,             -- model name / tool name / memory op
  start_ts REAL NOT NULL,
  end_ts REAL,
  duration_ms REAL,
  status TEXT NOT NULL,           -- ok | err
  metadata TEXT                   -- JSON (tokens, args, output preview)
);
CREATE INDEX idx_spans_run ON spans (run_id, start_ts);
```

### REST + SSE surface

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | serve `static/index.html` |
| GET | `/static/...` | static assets (jsx/css/js) |
| GET | `/api/agents` | list registered agents (name, model, status, run count) |
| GET | `/api/runs?agent=&limit=&offset=` | paginated run list w/ totals (tokens, cost, duration) |
| GET | `/api/runs/{run_id}` | run detail: row + spans + last 200 events |
| GET | `/api/runs/{run_id}/events?after=` | full event log paginated |
| GET | `/api/metrics?range=1h|6h|24h` | aggregated metrics for histogram + stat cards |
| GET | `/api/stream?agent=` | SSE — live event tail (all agents or filtered) |
| POST | `/api/playground/run` | body `{agent, input}` → returns SSE stream of events for that run |

### Cost calc

`pricing.py` loads `definable/model_pricing.json` once at import. Function:

```python
def cost(provider: str, model: str, in_tok: int, out_tok: int, cached_in_tok: int = 0) -> float:
  rates = _PRICING.get(provider, {}).get(model)
  if not rates:
    return 0.0  # graceful — unknown models render $0 (warned once)
  return (
    (in_tok - cached_in_tok) * rates["input_per_million"] / 1_000_000
    + cached_in_tok * rates.get("cached_input_per_million", rates["input_per_million"]) / 1_000_000
    + out_tok * rates["output_per_million"] / 1_000_000
  )
```

Token counts come from the model client's response metadata (already captured in `ModelResponded` event — confirm during plan phase).

### Tab scope decision

| Tab from mock | Action | Reason |
|---|---|---|
| FLEET | **drop** | Single-process scope. v1 is one consumer's process; fleet implies multi-process aggregator. |
| PLAYGROUND | **keep + wire** | Core ask. POST /api/playground/run → SSE → live render. |
| TRACES | **keep + wire** | Core ask. Real run rows + waterfall from `spans`. |
| EVALS | **drop** | Eval module was removed in harness-v2. Adding back is a separate spec. |
| METRICS | **keep + wire** | RPS/p50/p95/cost/tokens from aggregations over `events`+`runs`. |

Sidebar agent list reads `/api/agents` — only shows agents that registered this process.

---

## 9. Success Criteria

Concrete, testable conditions for "done":

- [ ] `from definable import Agent; Agent(name="t", model="claude-haiku-4-5", observability=True)` constructs without raising, prints `[definable] observability live → http://127.0.0.1:<port>`.
- [ ] Opening the printed URL renders the full dashboard (sidebar + topbar + 3 tabs).
- [ ] Calling `agent.arun("hi")` while the page is open: a new row appears in TRACES tab within 200 ms (verified via the SSE stream).
- [ ] Clicking a trace row opens the detail view with a waterfall containing one `llm` span and zero or more `tool` spans, durations populated.
- [ ] Playground: typing a message and pressing send invokes `agent.arun()` on the **same constructed instance**; assistant message streams token by token; per-turn footer shows input/output token counts and USD cost (non-zero for a known-pricing model).
- [ ] Metrics tab shows non-zero values for RPS, p95, tokens, cost after 3 runs.
- [ ] `definable.db.Repo` round-trips a dataclass (insert + get + list + update + delete) in a unit test.
- [ ] Migrations run idempotently — second app start does not re-apply 0001.
- [ ] All 4 quality gates green for `definable/db` and `definable/observability`.
- [ ] 30-run smoke test does not exceed 30 MB peak RSS attributable to dashboard subscribers (no event-buffer leak).

---

## 10. Open Questions

1. **Process exit semantics** — if the user's script ends right after `agent.arun()`, should the server hold the process open so they can keep browsing, or exit and dump the DB to disk? Current plan: exit cleanly; DB persists; user re-runs with `definable observability serve` CLI later (out of scope for v1?). **Decide in plan phase.**
2. **Token count extraction** — `ModelResponded` event currently carries `content` and `tool_calls` but I haven't confirmed it carries usage metadata. May need a small event change. **Verify in plan phase, propose minimal addition if missing.**
3. **Multiple Python processes** — if user runs `python a.py` and `python b.py` concurrently, both try port 7777. Plan: bump port and warn. **Confirm acceptable.**
4. **JSONL parity** — existing JSONL writer keeps running alongside DB writer. Fine for v1; eventually one becomes redundant. **Keep both for now; revisit after v1 ships.**
5. **CLI for re-opening past runs** — `definable observability serve --db .definable/observability.db` would be nice. **Stretch goal; not in v1 scope unless trivial.**

---

## 11. Out of Scope (v1)

- Multi-process / multi-host trace aggregation.
- Eval runs (no eval module).
- Persistent fleet view across machines.
- Authentication / multi-tenant.
- Production deployment story (this is a local dev tool).
- Replay (re-execute past trace with edited prompt).
- Diff between two runs.

These belong in later specs once the v1 surface lands.

---

## Next steps

1. **You review this spec.** Push back on anything that's wrong before I touch code.
2. On approval → I produce `/tasks/observability/plan.md` (Phase 2: PLAN) — dependency-ordered phases with verification checkpoints.
3. Then `/tasks/observability/todo.md` (Phase 3: TASKS) — discrete commits.
4. Then implement (Phase 4) one task at a time.

Open the spec. Mark anything wrong.
