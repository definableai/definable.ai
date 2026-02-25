# agent/replay — Episodic Memory and Run Inspection

The replay module lets you treat any past agent run as a first-class object. Load a run from a `RunOutput` or from a JSONL trace file, inspect its timeline of model calls, tool dispatches, knowledge retrievals, and memory recalls, then diff it against a second run to see exactly what changed. This is the organism's episodic memory: the ability to re-experience what happened.

## Module Structure

```
replay/
├── __init__.py   # Public API
├── types.py      # ToolCallRecord, ReplayTokens, ReplayStep,
                  # KnowledgeRetrievalRecord, MemoryRecallRecord,
                  # ToolCallsDiff, ReplayComparison
├── replay.py     # Replay dataclass + from_run_output / from_events / from_trace_file
└── compare.py    # compare_runs()
```

## Quick Start

```python
from definable.agent import Agent

agent = Agent(model="openai/gpt-4o-mini", tools=[...])

# Run the agent, then build a Replay from the result
output = await agent.arun("Summarise the Q3 report.")

from definable.agent.replay import Replay
replay = Replay.from_run_output(output)

print(replay.content)
print(f"Tokens: {replay.tokens.total_tokens}  Cost: ${replay.cost:.6f}")
print(f"Tool calls: {[tc.tool_name for tc in replay.tool_calls]}")
```

Load from a JSONL trace file written by `JSONLExporter`:

```python
replay = Replay.from_trace_file("./traces/session_abc123.jsonl")
# or filter to a specific run within a multi-run session file:
replay = Replay.from_trace_file("./traces/session_abc123.jsonl", run_id="run_xyz")
```

## API Reference

### `Replay`

Dataclass. A complete, structured snapshot of one past agent run.

```python
@dataclass
class Replay:
  # Identity
  run_id: str
  session_id: str
  agent_id: str
  agent_name: str
  model: str
  model_provider: str

  # Input / output
  input: Any                            # RunInput — the original prompt/media
  content: Any                          # final answer (str or structured)
  messages: list                        # full conversation history

  # Tool executions
  tool_calls: list[ToolCallRecord]

  # Aggregated metrics
  tokens: ReplayTokens
  cost: float | None
  duration: float | None                # seconds

  # Step timeline
  steps: list[ReplayStep]

  # RAG / memory
  knowledge_retrievals: list[KnowledgeRetrievalRecord]
  memory_recalls: list[MemoryRecallRecord]

  # Status
  status: str                           # "completed" | "error" | "cancelled"
  error: str | None

  # Raw events
  events: list[BaseRunOutputEvent]

  # Source tag
  source: str                           # "run_output" | "trace_file"
```

**Construction class methods:**

| Method | Use when |
|--------|---------|
| `Replay.from_run_output(run_output)` | You have the `RunOutput` from `agent.arun()` in memory |
| `Replay.from_events(events, run_id=None)` | You already have a list of deserialized `BaseRunOutputEvent` objects |
| `Replay.from_trace_file(path, run_id=None)` | You want to load from a JSONL file on disk |

For `from_events` and `from_trace_file`, if `run_id=None` the method picks the first run found in the event stream.

### `ReplayTokens`

Aggregated token counts for the run.

```python
@dataclass
class ReplayTokens:
  input_tokens: int
  output_tokens: int
  total_tokens: int
  reasoning_tokens: int
  cache_read_tokens: int
  cache_write_tokens: int
```

### `ReplayStep`

One entry in the per-step timeline. Covers model calls, tool calls, knowledge retrievals, and memory recalls.

```python
@dataclass
class ReplayStep:
  step_type: str               # "model_call" | "tool_call" | "knowledge_retrieval" | "memory_recall"
  name: str | None             # tool name, "model", "knowledge", or "memory"
  started_at: int              # Unix timestamp (seconds)
  completed_at: int | None
  duration_ms: float | None
```

### `ToolCallRecord`

One tool execution extracted from the run.

```python
@dataclass
class ToolCallRecord:
  tool_name: str
  tool_args: dict[str, Any] | None
  result: str | None
  error: bool | None           # True if the tool call raised an error
  started_at: int
  completed_at: int | None
  duration_ms: float | None
```

### `KnowledgeRetrievalRecord`

One RAG retrieval during the run.

```python
@dataclass
class KnowledgeRetrievalRecord:
  query: str | None
  documents_found: int
  documents_used: int
  duration_ms: float | None
```

### `MemoryRecallRecord`

One memory recall during the run.

```python
@dataclass
class MemoryRecallRecord:
  query: str | None
  tokens_used: int
  chunks_included: int
  chunks_available: int
  duration_ms: float | None
```

### `ToolCallsDiff`

Tool call diff produced by `compare_runs`.

```python
@dataclass
class ToolCallsDiff:
  added: list[ToolCallRecord]    # calls present in b but not a
  removed: list[ToolCallRecord]  # calls present in a but not b
  common: int                    # positional matches by name
```

### `ReplayComparison`

Side-by-side diff of two runs, returned by `compare_runs`.

```python
@dataclass
class ReplayComparison:
  original: Replay | None
  replayed: Replay | None
  content_diff: str | None       # unified diff string; None when content is identical
  cost_diff: float | None        # b.cost - a.cost; None if either run has no cost
  token_diff: int                # b.total_tokens - a.total_tokens
  duration_diff: float | None    # b.duration - a.duration (seconds)
  tool_calls_diff: ToolCallsDiff
```

### `compare_runs(a, b) -> ReplayComparison`

Accepts either `Replay` or `RunOutput` objects. Converts `RunOutput` automatically.

```python
from definable.agent.replay import compare_runs

diff = compare_runs(output_a, output_b)

if diff.content_diff:
  print(diff.content_diff)     # unified diff
print(f"Token delta: {diff.token_diff:+d}")
print(f"Cost delta:  ${diff.cost_diff:+.6f}")
print(f"New tools:   {[t.tool_name for t in diff.tool_calls_diff.added]}")
```

## Integration with Agent

The most common pattern is building a `Replay` directly from a `RunOutput`, then optionally diffing against a later run to see how a model or prompt change affected behaviour.

```python
from definable.agent import Agent
from definable.agent.replay import Replay, compare_runs
from definable.agent.tracing import Tracing, JSONLExporter

agent = Agent(
  model="openai/gpt-4o-mini",
  tracing=Tracing(exporters=[JSONLExporter("./traces")]),
)

# First run
r1 = await agent.arun("What are the risks of index bloat in Postgres?")
replay1 = Replay.from_run_output(r1)

# Second run after a prompt change
r2 = await agent.arun("What are the risks of index bloat in Postgres?")
diff = compare_runs(r1, r2)

print(f"Token delta:  {diff.token_diff:+d}")
print(f"Tool changes: +{len(diff.tool_calls_diff.added)} -{len(diff.tool_calls_diff.removed)}")
```

Loading from a trace file written in a previous session:

```python
from pathlib import Path
from definable.agent.replay import Replay

replay = Replay.from_trace_file(
  Path("./traces/session_abc123.jsonl"),
  run_id="run_xyz789",   # optional — omit to get the first run
)

for step in replay.steps:
  label = f"[{step.step_type}] {step.name or ''}"
  duration = f"{step.duration_ms:.0f}ms" if step.duration_ms else "?"
  print(f"{label:40s} {duration}")
```

## Gotchas

- `Replay.from_run_output` sets `source="run_output"`. `Replay.from_trace_file` and `from_events` set `source="trace_file"`. The `source` field tells you how the Replay was built, not where the data came from originally.
- `replay.duration` is in **seconds** when populated from `RunOutput.metrics` (which uses the `Metrics` field). When computed as a fallback from event timestamps it is also in seconds. Be careful not to confuse it with `ReplayStep.duration_ms`, which is in **milliseconds**.
- `replay.events` is only populated when the `RunOutput` carried events (i.e., the agent was run with `return_events=True` or the trace file is complete). Building a `Replay` from a `RunOutput` that has no events still produces a valid `Replay` — `tool_calls`, `tokens`, `cost`, and `status` are always extracted from the metrics fields.
- `compare_runs` uses positional name matching for `tool_calls_diff.common`. It does not do deep argument comparison. Two calls to `search()` with different queries both count as "common".
- `Replay.from_events(events, run_id=None)` silently returns an empty `Replay` (with `source="trace_file"`) if no events carry a `run_id`. Check `replay.run_id != ""` before use.

## Related Modules

- `agent/tracing/` — `JSONLExporter` writes the JSONL files that `Replay.from_trace_file` reads; `read_trace_events` does the deserialization
- `agent/run/` — `RunOutput` is the primary source object for `Replay.from_run_output`; all event types used by `from_events` are defined in `run/agent.py`
