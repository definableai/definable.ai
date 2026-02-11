# run

Runtime context and status types for the agent execution pipeline.

## Overview

This module defines the core data structures that flow through the agent execution pipeline: `RunContext` carries configuration and state for a single run, `RunOutput` holds the complete result, and `RunEvent` types represent the streaming event protocol. This is an internal module used by `agents/` and is not typically imported directly by users.

## API Reference

### RunContext

```python
from definable.run import RunContext
```

Execution context passed through the middleware chain and into tool calls.

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | `str` | Unique run identifier |
| `session_id` | `str` | Session identifier |
| `user_id` | `Optional[str]` | User identifier |
| `dependencies` | `Dict` | Injected tool dependencies |
| `metadata` | `Dict` | Arbitrary metadata |
| `session_state` | `Dict` | Mutable session state |
| `output_schema` | `Optional` | Structured output schema |
| `knowledge_context` | `Optional[str]` | Injected RAG context |
| `knowledge_documents` | `Optional[List]` | Retrieved documents |
| `memory_context` | `Optional[str]` | Injected memory context |
| `readers_context` | `Optional[str]` | Extracted file content |

### RunStatus

```python
from definable.run import RunStatus
```

| Value | Description |
|-------|-------------|
| `PENDING` | Run has not started |
| `RUNNING` | Run is in progress |
| `COMPLETED` | Run finished successfully |
| `PAUSED` | Run paused for HITL (human-in-the-loop) |
| `CANCELLED` | Run was cancelled |
| `ERROR` | Run failed with an error |

## See Also

- `agents/` — Agent creates and manages `RunContext` during execution
- `run/agent.py` — `RunOutput`, `RunInput`, `RunEvent`, and all event dataclasses
- `run/requirement.py` — `RunRequirement` for HITL workflows
