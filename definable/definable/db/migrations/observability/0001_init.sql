CREATE TABLE agents (
  id TEXT PRIMARY KEY,
  registered_at REAL NOT NULL,
  model TEXT NOT NULL,
  instructions TEXT
);

CREATE TABLE runs (
  id TEXT PRIMARY KEY,
  agent_id TEXT NOT NULL REFERENCES agents(id),
  started_at REAL NOT NULL,
  ended_at REAL,
  status TEXT NOT NULL,
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
  type TEXT NOT NULL,
  payload TEXT NOT NULL
);

CREATE INDEX idx_events_run_ts ON events (run_id, timestamp);

CREATE TABLE spans (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL REFERENCES runs(id),
  parent_id INTEGER,
  kind TEXT NOT NULL,
  name TEXT NOT NULL,
  start_ts REAL NOT NULL,
  end_ts REAL,
  duration_ms REAL,
  status TEXT NOT NULL,
  metadata TEXT
);

CREATE INDEX idx_spans_run ON spans (run_id, start_ts);
