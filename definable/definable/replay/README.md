# replay

Inspect, compare, and re-execute past agent runs.

## Quick Start

```python
from definable.agents import Agent

agent = Agent(model=model, tools=[...])

# Run and inspect
output = agent.run("Summarize this document.")
replay = agent.replay(run_output=output)
print(replay.content, replay.tokens.total_tokens, replay.cost)

# Compare two runs
output2 = agent.run("Summarize this document.")
diff = agent.compare(output, output2)
print(diff.token_diff, diff.cost_diff, diff.content_diff)
```

## Module Structure

```
replay/
├── __init__.py    # Public API exports
├── types.py       # ToolCallRecord, ReplayTokens, ReplayStep, ReplayComparison, etc.
├── replay.py      # Replay dataclass and construction methods
└── compare.py     # compare_runs() function
```

## API Reference

### Replay

```python
from definable.replay import Replay
```

Structured view of a past agent run, built from trace events or `RunOutput`.

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | `str` | Run identifier |
| `session_id` | `str` | Session identifier |
| `agent_name` | `str` | Agent name |
| `model` | `str` | Model used |
| `input` | `Any` | Original input |
| `content` | `Any` | Final output content |
| `messages` | `List` | Conversation messages |
| `tool_calls` | `List[ToolCallRecord]` | Tool executions |
| `tokens` | `ReplayTokens` | Aggregated token usage |
| `cost` | `Optional[float]` | Total cost |
| `duration` | `Optional[float]` | Total duration (ms) |
| `steps` | `List[ReplayStep]` | Step-by-step timeline |
| `knowledge_retrievals` | `List[KnowledgeRetrievalRecord]` | RAG retrievals |
| `memory_recalls` | `List[MemoryRecallRecord]` | Memory recalls |
| `status` | `str` | `"completed"`, `"error"`, or `"cancelled"` |
| `error` | `Optional[str]` | Error message if failed |
| `source` | `str` | `"run_output"` or `"trace_file"` |

**Construction methods:**

| Method | Description |
|--------|-------------|
| `Replay.from_run_output(run_output)` | Build from a `RunOutput` object |
| `Replay.from_events(events, run_id=)` | Build from a list of trace events |
| `Replay.from_trace_file(path, run_id=)` | Build from a JSONL trace file |

### compare_runs

```python
from definable.replay import compare_runs

diff = compare_runs(a, b)  # Replay or RunOutput
```

Returns a `ReplayComparison`:

| Field | Type | Description |
|-------|------|-------------|
| `original` | `Replay` | First run |
| `replayed` | `Replay` | Second run |
| `content_diff` | `Optional[str]` | Unified diff of output content |
| `cost_diff` | `Optional[float]` | Cost difference (b - a) |
| `token_diff` | `int` | Token difference (b - a) |
| `duration_diff` | `Optional[float]` | Duration difference (b - a) |
| `tool_calls_diff` | `ToolCallsDiff` | Added/removed/common tool calls |

### Types

```python
from definable.replay import (
  ToolCallRecord,              # tool_name, tool_args, result, error, duration_ms
  ReplayTokens,                # input/output/total/reasoning/cache tokens
  ReplayStep,                  # step_type, name, started_at, duration_ms
  KnowledgeRetrievalRecord,    # query, documents_found, documents_used
  MemoryRecallRecord,          # query, tokens_used, chunks_included
  ToolCallsDiff,               # added, removed, common
)
```

### Agent Integration

```python
# Inspect a past run (returns Replay)
replay = agent.replay(run_output=output)
replay = agent.replay(trace_file="runs.jsonl")
replay = agent.replay(events=event_list)

# Re-execute with overrides (returns new RunOutput)
new_output = agent.replay(run_output=output, model=new_model)
new_output = agent.replay(
  trace_file="runs.jsonl",
  run_id="abc123",
  instructions="Updated instructions",
  tools=[new_tool],
)

# Compare runs
diff = agent.compare(output_a, output_b)
```

Async variants: `agent.areplay()`, `compare_runs()` (sync).

## See Also

- `agents/` — Agent integration via `replay()`, `areplay()`, `compare()`
- `tracing/` — JSONL trace files consumed by `Replay.from_trace_file()`
