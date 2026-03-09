# agent/tracing — Observability for Agent Runs

The tracing module is the nervous system of Definable. It captures every event emitted during an agent run — model calls, tool dispatches, knowledge retrievals, errors — and routes them to one or more backends (JSONL files, console output, or a custom sink you write). Tracing is entirely non-intrusive: failures in any exporter are suppressed so a broken trace backend never takes down the agent.

## Module Structure

```
tracing/
├── __init__.py   # Public API
├── base.py       # Tracing, TraceExporter, TraceWriter, NoOpExporter
├── debug.py      # DebugExporter (rich console output)
└── jsonl.py      # JSONLExporter, read_trace_file, read_trace_events
```

## Quick Start

```python
from definable.agent import Agent
from definable.agent.tracing import Tracing, JSONLExporter

# Write every event to ./traces/<session_id>.jsonl
agent = Agent(
  model="openai/gpt-4o-mini",
  tracing=Tracing(exporters=[JSONLExporter("./traces")]),
)

result = await agent.arun("Summarise the Q3 report.")
print(result.content)
```

Enable the colour-coded debug view instead:

```python
from definable.agent.tracing import Tracing, DebugExporter

agent = Agent(
  model="openai/gpt-4o-mini",
  tracing=Tracing(exporters=[DebugExporter()]),
)
```

Or use the shorthand — `Agent(debug=True)` automatically attaches `DebugExporter`:

```python
agent = Agent(model="openai/gpt-4o-mini", debug=True)
```

## API Reference

### `Tracing`

Dataclass. The block you pass to `Agent(tracing=...)`.

```python
@dataclass
class Tracing:
  enabled: bool = True
  exporters: list[TraceExporter] | None = None
  event_filter: Callable[[BaseRunOutputEvent], bool] | None = None
  batch_size: int = 1
  flush_interval_ms: int = 5000
```

| Field | Purpose |
|-------|---------|
| `enabled` | Master switch. Set `False` to disable all export without removing the block. |
| `exporters` | List of `TraceExporter` implementations. Multiple exporters fan-out in order. |
| `event_filter` | Optional predicate — return `False` to drop the event before export. |
| `batch_size` | Reserved for future batching. Currently `1` means emit immediately. |
| `flush_interval_ms` | Reserved for interval-based flushing. Ignored by current built-in exporters. |

### `TraceExporter` (Protocol)

Implement this protocol to write events to any backend.

```python
class TraceExporter(Protocol):
  def export(self, event: BaseRunOutputEvent) -> None: ...
  def flush(self) -> None: ...
  def shutdown(self) -> None: ...
```

Exceptions raised inside `export()`, `flush()`, or `shutdown()` are silently suppressed by `TraceWriter` — tracing must never break the main execution path.

### `TraceWriter`

Internal coordinator that holds the list of exporters and applies the filter. You do not normally instantiate this directly — `Agent` creates it from your `Tracing` block. It is documented here for custom exporter authors.

```python
writer = TraceWriter(tracing_config)
writer.write(event)  # fan-out to all exporters; respects event_filter
writer.flush()  # flush all exporters
writer.shutdown()  # flush then close all exporters
writer.add_exporter(e)  # add at runtime
writer.remove_exporter(e)  # remove at runtime; returns True if found
writer.exporter_count  # int — number of currently attached exporters
```

### `JSONLExporter`

Writes events to JSONL files, one file per session. Each line is a complete JSON document — the serialised event. The directory is created on first use.

```python
class JSONLExporter:
  def __init__(
    self,
    trace_dir: str | None = None,  # default: .definable/traces/
    flush_each: bool = True,  # flush after every write
    mirror_stdout: bool = True,  # also print each line to stdout
  ): ...
```

**File layout:**

```
./traces/
  session_abc123.jsonl    # one line per event
  session_xyz789.jsonl
  default.jsonl           # events with no session_id
```

**Reading traces back:**

```python
from pathlib import Path
from definable.agent.tracing import read_trace_file, read_trace_events

# Raw dicts — fast, no deserialization
events_raw = read_trace_file(Path("./traces/session_abc123.jsonl"))
for ev in events_raw:
  print(ev["event"], ev["run_id"])

# Typed event objects — full field access
from definable.agent.run.agent import RunCompletedEvent

events = read_trace_events(Path("./traces/session_abc123.jsonl"))
for ev in events:
  if isinstance(ev, RunCompletedEvent):
    print(f"Completed: {ev.content}")
```

`JSONLExporter` is also a context manager:

```python
with JSONLExporter("./traces") as exporter:
  agent = Agent(model=model, tracing=Tracing(exporters=[exporter]))
  await agent.arun("Hello")
# exporter.shutdown() called automatically
```

**Properties / helpers:**

| Member | Type | Notes |
|--------|------|-------|
| `open_sessions` | `int` | Number of currently open file handles |
| `get_trace_path(session_id)` | `Path` | Returns the path that would be used (may not exist yet) |

### `DebugExporter`

Rich-formatted console exporter used by `Agent(debug=True)`. Writes colour-coded output to **stderr** so it does not pollute captured stdout.

```python
class DebugExporter:
  def __init__(
    self,
    *,
    max_content_length: int = 500,  # truncate long content fields
    show_tools: bool = True,  # show tool definitions on ModelCallStarted
    show_metrics: bool = True,  # show token counts on ModelCallCompleted
  ): ...
```

Events it handles: `RunStarted`, `ModelCallStarted`, `ModelCallCompleted`, `ToolCallStarted`, `ToolCallCompleted`, `RunCompleted`, `DesktopAction`, `BridgeCall`. All other event types are silently ignored.

### `NoOpExporter`

Discards every event. Useful in tests to suppress any real I/O while still exercising code paths that write to a `TraceWriter`.

```python
from definable.agent.tracing import NoOpExporter

exporter = NoOpExporter()
exporter.export(event)  # no-op
exporter.flush()  # no-op
exporter.shutdown()  # no-op
```

### `read_trace_file(path: Path) -> list[dict]`

Reads a JSONL trace file and returns a list of raw dictionaries. Fast and allocation-light — use when you only need string fields (`event`, `run_id`, timestamps).

### `read_trace_events(path: Path) -> list[BaseRunOutputEvent]`

Reads a JSONL trace file and deserialises each line into the correct typed event dataclass via `run_output_event_from_dict`. Use when you need full typed access to event fields.

## Custom Exporter

```python
from definable.agent.tracing import TraceExporter
from definable.agent.events import BaseRunOutputEvent


class MyExporter:
  """Send events to a remote collector."""

  def __init__(self, endpoint: str):
    self._endpoint = endpoint

  def export(self, event: BaseRunOutputEvent) -> None:
    import httpx

    # Fire-and-forget; exceptions are suppressed by TraceWriter
    httpx.post(self._endpoint, json=event.to_dict())

  def flush(self) -> None:
    pass  # nothing to buffer

  def shutdown(self) -> None:
    pass  # no persistent resources
```

Attach it like any other exporter:

```python
agent = Agent(
  model="openai/gpt-4o-mini",
  tracing=Tracing(exporters=[MyExporter("https://collector.example.com/events")]),
)
```

## Filtering Events

Use `event_filter` to drop events you do not need. The filter runs before any exporter sees the event, so it reduces both I/O and deserialization cost downstream.

```python
from definable.agent.run.agent import RunContentEvent

# Drop streaming deltas — only keep structural events
agent = Agent(
  model="openai/gpt-4o-mini",
  tracing=Tracing(
    exporters=[JSONLExporter("./traces")],
    event_filter=lambda e: not isinstance(e, RunContentEvent),
  ),
)
```

## Combining Exporters

Exporters fan-out in declaration order. All exporters receive every event that passes the filter.

```python
agent = Agent(
  model="openai/gpt-4o-mini",
  tracing=Tracing(
    exporters=[
      JSONLExporter("./traces"),  # write to disk
      DebugExporter(),  # also print to stderr
    ],
  ),
)
```

## Gotchas

- `JSONLExporter` defaults `trace_dir` to `.definable/traces/` in the project workspace. Pass an explicit path in tests to avoid writing to your workspace.
- `mirror_stdout=True` on `JSONLExporter` prints raw JSONL lines. Disable it if you are running the agent inside a structured logging pipeline.
- `DebugExporter` writes to **stderr**, not stdout. Capturing stdout in tests does not suppress its output.
- Exporter exceptions are swallowed by `TraceWriter`. If your custom exporter appears silent, check it directly: instantiate and call `export()` manually to see any errors it raises.
- `TraceWriter` is created once when the agent initialises. Adding exporters to the `Tracing` block after agent creation has no effect — use `writer.add_exporter()` if you have access to the writer.

## Related Modules

- `agent/run/` — defines `BaseRunOutputEvent` and all concrete event types that flow through the writer
- `agent/replay/` — uses `read_trace_events()` to reconstruct past runs from JSONL files
- `agent/pipeline/` — emits `PhaseStartedEvent` / `PhaseCompletedEvent` and all other structured events
