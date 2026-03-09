# agent/run — Execution Context and Run Output

The run module is the heartbeat of agent execution. It defines the three core data structures that flow through every `arun()` call: `RunContext` carries configuration and in-flight state through the pipeline; `RunOutput` carries the completed result back to the caller; and the `RunOutputEvent` union type — backed by `RunEvent` and some 40+ event dataclasses — powers the streaming protocol. `RunStatus` tracks where a run is in its lifecycle.

## Module Structure

```
run/
├── __init__.py     # Public API: RunContext, RunStatus
├── base.py         # RunContext, RunStatus, BaseRunOutputEvent
├── agent.py        # RunOutput, RunInput, RunEvent, RunOutputEvent union,
                    # all event dataclasses, run_output_event_from_dict
└── requirement.py  # RunRequirement (HITL / human-in-the-loop)
```

## Quick Start

Most code never imports from `agent/run/` directly — the public surface lives on `RunOutput` returned by `agent.arun()` and on events yielded by `agent.arun_stream()`. The examples below show the most common direct uses.

```python
from definable.agent import Agent

agent = Agent(model="openai/gpt-4o-mini")
result = await agent.arun("What is the boiling point of tungsten?")

print(result.content)  # the answer
print(result.status)  # RunStatus.completed
print(result.metrics.cost)  # total cost in USD
print(result.run_id)  # unique run identifier
```

Checking run status programmatically:

```python
from definable.agent.run import RunStatus

if result.status == RunStatus.completed:
  print("OK:", result.content)
elif result.status == RunStatus.error:
  print("Error in run", result.run_id)
elif result.status == RunStatus.paused:
  # Human-in-the-loop — see RunRequirement
  for req in result.active_requirements:
    req.confirm()
```

## API Reference

### `RunStatus`

`str` enum — values are the wire strings stored in `RunOutput.status`.

```python
class RunStatus(str, Enum):
  pending = "PENDING"  # not yet started
  running = "RUNNING"  # in progress
  completed = "COMPLETED"  # finished successfully
  paused = "PAUSED"  # waiting for human input (HITL)
  cancelled = "CANCELLED"  # cancelled by the caller
  blocked = "BLOCKED"  # waiting on an external dependency
  error = "ERROR"  # failed with an error
```

Compare against enum members, not strings:

```python
from definable.agent.run import RunStatus

assert result.status == RunStatus.completed  # correct
assert result.status == "COMPLETED"  # also works (str enum), but prefer the member
```

### `RunContext`

Dataclass. Mutable execution context created at the start of each `arun()` call and threaded through the pipeline. Middleware, knowledge retrieval, memory, and tools can all read and write fields on this object.

```python
@dataclass
class RunContext:
  # Identifiers
  run_id: str
  session_id: str
  user_id: str | None = None

  # Caller-supplied configuration
  dependencies: dict[str, Any] | None = None
  knowledge_filters: dict[str, Any] | list[FilterExpr] | None = None
  metadata: dict[str, Any] | None = None
  session_state: dict[str, Any] | None = None
  output_schema: type[BaseModel] | dict[str, Any] | None = None

  # Pipeline-populated (read after the run)
  knowledge_context: str | None = None  # formatted RAG context injected into the prompt
  knowledge_documents: list[Document] | None = None  # the raw retrieved documents
  memory_context: str | None = None  # formatted memory payload
  research_context: str | None = None  # formatted deep research output
  research_result: object | None = None  # full ResearchResult object
  readers_context: str | None = None  # extracted file content

  # Which layers ran this turn
  active_layers: set[str]  # e.g. {"knowledge", "memory"}
```

**Supplying `RunContext` to a run:**

```python
from definable.agent.run import RunContext

ctx = RunContext(
  run_id="my-run-001",
  session_id="session-42",
  user_id="user_abc",
  dependencies={"db": my_db_connection},
  knowledge_filters={"source": "internal-docs"},
  metadata={"request_id": "req-999"},
)

result = await agent.arun("What changed in v3.2?", run_context=ctx)
```

Reading pipeline-populated fields after the run (via the context object passed in — the same object is mutated):

```python
print(ctx.knowledge_context)  # the text that was injected
print(ctx.active_layers)  # e.g. {"knowledge"}
```

### `RunOutput`

Dataclass. The complete result of `agent.arun()`. This is the primary return type for agent execution.

```python
@dataclass
class RunOutput:
  # Identity
  run_id: str | None
  agent_id: str | None
  agent_name: str | None
  session_id: str | None
  parent_run_id: str | None
  workflow_id: str | None
  user_id: str | None

  # Input
  input: RunInput | None  # original prompt + attached media

  # Output
  content: Any  # str or structured BaseModel
  parsed: Any  # populated when output_schema is used (see gotchas)
  content_type: str  # "str" by default
  reasoning_content: str | None  # raw reasoning text (thinking-enabled runs)
  reasoning_steps: list[ReasoningStep] | None
  reasoning_messages: list[Message] | None

  # Model info
  model: str | None
  model_provider: str | None
  model_provider_data: dict[str, Any] | None

  # Conversation
  messages: list[Message] | None  # full conversation history for multi-turn

  # Metrics
  metrics: Metrics | None  # tokens, cost, duration, cache hits

  # Tool executions
  tools: list[ToolExecution] | None

  # Media
  images: list[Image] | None
  videos: list[Video] | None
  audio: list[Audio] | None
  files: list[File] | None
  response_audio: Audio | None  # model-generated audio (voice runs)

  # Citations / references
  citations: Citations | None
  references: list[MessageReferences] | None

  # Metadata
  metadata: dict[str, Any] | None
  session_state: dict[str, Any] | None
  created_at: int  # Unix timestamp

  # Events (streaming history)
  events: list[RunOutputEvent] | None

  # Pipeline phase timing
  phase_metrics: list[PhaseMetric] | None

  # Status
  status: RunStatus

  # HITL
  requirements: list[RunRequirement] | None
  workflow_step_id: str | None  # FK to StepOutput.step_id in workflows
```

**Key properties:**

| Property | Returns |
|----------|---------|
| `is_paused` | `True` when `status == RunStatus.paused` |
| `is_cancelled` | `True` when `status == RunStatus.cancelled` |
| `active_requirements` | Unresolved `RunRequirement` items |
| `tools_requiring_confirmation` | Tools awaiting a `confirm()` / `reject()` call |
| `tools_requiring_user_input` | Tools awaiting field values from the user |
| `tools_awaiting_external_execution` | Tools that need external system execution |

**Serialisation:**

```python
json_str = result.to_json()  # pretty-printed JSON
json_str = result.to_json(indent=None)  # compact
d = result.to_dict()
restored = RunOutput.from_dict(d)
```

**Multi-turn conversation — passing history forward:**

```python
r1 = await agent.arun("What is async/await?")
r2 = await agent.arun("Give me a code example.", messages=r1.messages)
```

### `RunEvent` (enum)

All event type strings emitted by the streaming protocol. The full set:

```
RunStarted, RunContent, RunContentCompleted, RunIntermediateContent,
RunCompleted, RunError, RunCancelled, RunPaused, RunContinued,
PreHookStarted, PreHookCompleted, PostHookStarted, PostHookCompleted,
ToolCallStarted, ToolCallCompleted, ToolCallError,
ReasoningStarted, ReasoningStep, ReasoningContentDelta, ReasoningCompleted,
KnowledgeRetrievalStarted, KnowledgeRetrievalCompleted,
MemoryRecallStarted, MemoryRecallCompleted,
MemoryUpdateStarted, MemoryUpdateCompleted,
FileReadStarted, FileReadCompleted,
SessionSummaryStarted, SessionSummaryCompleted,
ParserModelResponseStarted, ParserModelResponseCompleted,
OutputModelResponseStarted, OutputModelResponseCompleted,
DeepResearchStarted, DeepResearchProgress, DeepResearchCompleted,
GuardrailChecked, GuardrailBlocked,
ModelCallStarted, ModelCallCompleted,
CompressionStarted, CompressionCompleted,
SubAgentSpawned, SubAgentCompleted, SubAgentFailed, SubAgentKilled,
PhaseStarted, PhaseCompleted,
InterfaceStarted, InterfaceStopped, InterfaceRestarted, InterfaceError,
BridgeCall, DesktopAction,
CustomEvent
```

### `RunOutputEvent` (union type)

`Union[RunStartedEvent, RunCompletedEvent, ToolCallStartedEvent, ...]` — the type of each item in `RunOutput.events` and each value yielded by `agent.arun_stream()`. Use `isinstance` checks to narrow:

```python
from definable.agent.run.agent import RunCompletedEvent, ToolCallCompletedEvent

async for event in agent.arun_stream("Analyse sales data."):
  if isinstance(event, ToolCallCompletedEvent) and event.tool:
    print(f"Tool {event.tool.tool_name} returned: {event.tool.result}")
  elif isinstance(event, RunCompletedEvent):
    print("Final answer:", event.content)
```

### `BaseRunOutputEvent`

Base dataclass for all event types. Provides `to_dict()`, `to_json()`, and `from_dict()`. Not instantiated directly.

```python
# Deserialise a dict back to the correct typed event
from definable.agent.run.agent import run_output_event_from_dict
event = run_output_event_from_dict({"event": "RunCompleted", "run_id": "...", ...})
```

### `RunInput`

Dataclass. Captures the exact input passed to `agent.arun()`, including attached media.

```python
@dataclass
class RunInput:
  input_content: str | list | dict | Message | list[Message] | BaseModel
  images: Sequence[Image] | None = None
  videos: Sequence[Video] | None = None
  audios: Sequence[Audio] | None = None
  files: Sequence[File] | None = None
```

Available on `RunOutput.input` and `RunStartedEvent.run_input`.

## Gotchas

- **`parsed` is rarely populated.** `RunOutput.parsed` is intended to hold the deserialized structured output when `output_schema=MyModel` is used. As of the current version this field is not reliably populated (open bug #6). Access structured output via `result.content` which is already the `BaseModel` instance when the schema matches.
- **`status` is a `RunStatus` enum on `RunOutput` but a plain string on `Replay`.** `RunOutput.status == RunStatus.completed` works; `Replay.status == "completed"` is correct on the replay side.
- **Multi-turn history requires explicit forwarding.** Passing `session_id` alone does nothing. You must pass `messages=r1.messages` to the next `arun()` call, or use `Memory` for automatic persistence.
- **`RunContext` is mutated by the pipeline.** After `arun()` returns, `ctx.knowledge_context`, `ctx.active_layers`, etc. reflect what happened during the run. Do not reuse the same `RunContext` across multiple calls.
- **`RunStatus` values are uppercase strings** (`"COMPLETED"`, not `"completed"`). This matters when comparing against strings from external sources or serialised JSON.
- **`blocked`** (`RunStatus.blocked`) is reserved for dependency-gated runs and is not yet emitted by the standard pipeline.
- **`workflow_step_id`** on `RunOutput` is a foreign key into `StepOutput.step_id` in the workflow module. It is populated only when the run was executed as part of a `Workflow`.

## Related Modules

- `agent/` — `Agent.arun()` creates `RunContext`, drives the pipeline, and returns `RunOutput`
- `agent/tracing/` — `BaseRunOutputEvent` is the base type for all events written to trace exporters
- `agent/replay/` — `RunOutput` is the source for `Replay.from_run_output()`
- `agent/workflow/` — `WorkflowOutput.step_outputs` contains one `RunOutput` per step; `workflow_step_id` links them
- `agent/run/requirement.py` — `RunRequirement` is used for HITL pause-and-resume flows
