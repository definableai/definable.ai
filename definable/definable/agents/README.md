# agents

The central entry point for building LLM-powered agents with tool calling, middleware, memory, knowledge retrieval, and tracing.

## Quick Start

```python
from definable.agents import Agent
from definable.models.openai import OpenAIChat
from definable.tools.decorator import tool

@tool
def get_weather(city: str) -> str:
  """Get the current weather for a city."""
  return f"The weather in {city} is sunny."

agent = Agent(
  model=OpenAIChat(id="gpt-4o"),
  tools=[get_weather],
  instructions="You are a helpful weather assistant.",
)

response = agent.run("What's the weather in Tokyo?")
print(response.content)
```

## Module Structure

```
agents/
├── __init__.py          # Public API exports
├── agent.py             # Agent class — run/arun/arun_stream/serve/aserve
├── config.py            # AgentConfig, CompressionConfig, KnowledgeConfig, ReadersConfig, TracingConfig
├── middleware.py         # Middleware protocol and built-in implementations
├── toolkit.py           # Toolkit base class for tool collections
├── testing.py           # MockModel, AgentTestCase, create_test_agent
├── toolkits/
│   └── knowledge.py     # KnowledgeToolkit — explicit RAG search tools
└── tracing/
    ├── base.py          # TraceExporter protocol, TraceWriter, NoOpExporter
    └── jsonl.py         # JSONLExporter, read_trace_file, read_trace_events
```

## API Reference

### Agent

The main orchestration class. Manages the model invocation loop, tool execution, middleware chain, memory, knowledge retrieval, and file reading.

```python
from definable.agents import Agent

agent = Agent(
  model=model,
  tools=[...],
  toolkits=[...],
  instructions="...",
  memory=memory,
  readers=True,
  name="my-agent",
  config=AgentConfig(...),
)
```

**Execution methods:**

| Method | Description |
|--------|-------------|
| `run(input)` | Synchronous multi-turn execution |
| `arun(input)` | Async execution with middleware chain |
| `run_stream(input)` | Sync streaming, yields `RunOutputEvent`s |
| `arun_stream(input)` | Async streaming with full agent loop |

**Lifecycle:**

| Method | Description |
|--------|-------------|
| `use(middleware)` | Add middleware to the chain |
| `before_request(fn)` | Register pre-execution hook |
| `after_response(fn)` | Register post-execution hook |
| `on(trigger)` | Register a trigger handler (decorator) |
| `emit(event_name, data)` | Fire EventTriggers (fire-and-forget) |
| `add_interface(interface)` | Register a messaging interface |
| `serve(...)` | Start sync runtime (server + interfaces + cron) |
| `aserve(...)` | Start async runtime |

### Configuration

```python
from definable.agents import (
  AgentConfig,
  CompressionConfig,
  KnowledgeConfig,
  ReadersConfig,
  TracingConfig,
)
```

| Class | Purpose |
|-------|---------|
| `AgentConfig` | Top-level frozen dataclass: identity, execution limits, retry, validation |
| `CompressionConfig` | Tool result compression settings (model, token/count limits) |
| `KnowledgeConfig` | RAG integration (Knowledge instance, top_k, rerank, context format) |
| `ReadersConfig` | File reader settings (registry, max content length) |
| `TracingConfig` | Trace exporters, event filtering, batching |

### Middleware

```python
from definable.agents import (
  Middleware,
  LoggingMiddleware,
  RetryMiddleware,
  MetricsMiddleware,
  KnowledgeMiddleware,
)
```

| Class | Description |
|-------|-------------|
| `Middleware` | Protocol: `async __call__(context, next_handler) -> RunOutput` |
| `LoggingMiddleware` | Logs run start, completion, and errors |
| `RetryMiddleware` | Exponential backoff retry on transient errors |
| `MetricsMiddleware` | Timing metrics collection (average latency, run/error counts) |
| `KnowledgeMiddleware` | RAG retrieval before model invocation |

### Toolkit

```python
from definable.agents import Toolkit, KnowledgeToolkit
```

- `Toolkit` — Base class for grouping related tools. Override the `tools` property or attach `Function` attributes.
- `KnowledgeToolkit` — Provides `search_knowledge(query)` and `get_document_count()` tools for explicit RAG.

### Tracing

```python
from definable.agents import (
  TraceExporter, TraceWriter, JSONLExporter, NoOpExporter,
)
from definable.agents.tracing import read_trace_file, read_trace_events
```

- `TraceExporter` — Protocol for pluggable trace backends (`export`, `flush`, `shutdown`).
- `JSONLExporter` — Writes per-session `.jsonl` trace files.
- `NoOpExporter` — Silent exporter for testing.
- `read_trace_file(path)` / `read_trace_events(path)` — Utilities to read trace files.

### Testing

```python
from definable.agents import MockModel, AgentTestCase, create_test_agent
```

- `MockModel(responses=[], tool_calls=[])` — Deterministic model mock with `assert_called()`, `assert_called_times(n)`.
- `AgentTestCase` — Base test class with `assert_tool_called()`, `assert_no_errors()`, `assert_content_contains()`.
- `create_test_agent(responses=[], tools=[])` — Convenience factory.

## See Also

- `models/` — LLM provider implementations
- `tools/` — `@tool` decorator and `Function` class
- `knowledge/` — RAG pipeline components
- `memory/` — Cognitive memory system
- `readers/` — File content extraction
- `triggers/` — Webhooks, cron, events
- `runtime/` — HTTP server and runtime orchestration
- `definable/examples/` — Runnable examples
