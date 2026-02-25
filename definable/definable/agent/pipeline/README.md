# Pipeline

> Phase-based agent execution pipeline with hooks, events, debugging, and sub-agent spawning.

The Pipeline decomposes agent execution into discrete, hookable phases. Developers can add, remove, replace, and reorder phases, register before/after/instead hooks, subscribe to events, raise `ToolRetry` for model-feedback retries, and configure sub-agent spawning policies. The pipeline is built once at `Agent.__init__` and reused for all runs.

## Quick Start

```python
from definable.agent import Agent
from definable.agent.pipeline import Pipeline, ToolRetry, DebugConfig
from definable.tool.decorator import tool

# Tool retry: ask the model to fix its arguments
@tool
def search(query: str) -> str:
  """Search the web."""
  if len(query) < 3:
    raise ToolRetry("Query too short. Provide at least 3 characters.")
  return f"Results for: {query}"

agent = Agent(
  model="openai/gpt-4o-mini",
  tools=[search],
)

# Register a hook on the pipeline
@agent.pipeline.hook("before:invoke_loop")
async def log_messages(state):
  print(f"Messages going to model: {len(state.invoke_messages)}")
  return state

# Run with debug inspection
agent = Agent(
  model="openai/gpt-4o-mini",
  debug=DebugConfig(
    breakpoints={"invoke_loop"},
    step_mode=False,
    log_state_changes=True,
  ),
)
```

## Architecture

```
Pipeline (orchestrator)
  |
  +-- phases: List[Phase]              -- ordered execution steps
  |     +-- PreparePhase               -- set up identity, input, tools
  |     +-- RecallPhase                -- fetch knowledge, memory, research
  |     +-- ThinkPhase                 -- deliberate reasoning (optional)
  |     +-- GuardInputPhase            -- input guardrails
  |     +-- ComposePhase               -- build system prompt + invoke messages
  |     +-- InvokeLoopPhase            -- model calls + tool execution loop
  |     +-- GuardOutputPhase           -- output guardrails
  |     +-- StorePhase                 -- persist memory, emit final events
  |
  +-- hooks: Dict[spec, List[callback]] -- before/after/instead per phase
  +-- event_stream: EventStream         -- unified event dispatch
  +-- debug: DebugConfig                -- breakpoints, step mode, inspector
```

### Module Structure

```
agent/pipeline/
+-- __init__.py         # Public API: Pipeline, Phase, BasePhase, LoopState,
|                       #   LoopStatus, PhaseMetric, ToolRetry, DebugConfig,
|                       #   SubAgentPolicy, SubAgentConfig, ThreadControlBlock,
|                       #   EventStream
+-- pipeline.py         # Pipeline class (orchestration, hooks, execution)
+-- phase.py            # Phase protocol + BasePhase convenience class
+-- state.py            # LoopState (mutable state), LoopStatus, PhaseMetric
+-- tool_retry.py       # ToolRetry exception
+-- debug.py            # DebugConfig dataclass
+-- sub_agent.py        # SubAgentPolicy, SubAgentConfig, ThreadControlBlock
+-- event_stream.py     # EventStream (subscribe/emit)
+-- phases/
    +-- __init__.py     # All 8 default phase classes
    +-- prepare.py      # PreparePhase
    +-- recall.py       # RecallPhase
    +-- think.py        # ThinkPhase
    +-- guard.py        # GuardInputPhase, GuardOutputPhase
    +-- compose.py      # ComposePhase
    +-- invoke.py       # InvokeLoopPhase
    +-- store.py        # StorePhase
```

### Execution Flow

```
arun("prompt")
  |
  v
Pipeline.execute(initial_state)
  |
  +-- PreparePhase     -- raw_instruction -> new_messages, load tools, set identity
  +-- RecallPhase      -- knowledge search, memory fetch, deep research
  +-- ThinkPhase       -- reasoning/thinking (skipped if thinking=never)
  +-- GuardInputPhase  -- run input guardrails (skipped if none configured)
  +-- ComposePhase     -- build system_content, assemble invoke_messages
  +-- InvokeLoopPhase  -- model.ainvoke() + tool dispatch loop
  +-- GuardOutputPhase -- run output guardrails (skipped if none configured)
  +-- StorePhase       -- persist to memory, emit final events
  |
  v
RunOutput
```

## API Reference

### LoopStatus

Enum representing the execution status of the pipeline.

```python
from definable.agent.pipeline import LoopStatus

LoopStatus.pending     # "pending"   -- not yet started
LoopStatus.running     # "running"   -- currently executing phases
LoopStatus.completed   # "completed" -- all phases finished successfully
LoopStatus.paused      # "paused"    -- HITL pause (awaiting user input)
LoopStatus.cancelled   # "cancelled" -- cancelled via CancellationToken
LoopStatus.blocked     # "blocked"   -- blocked on a requirement
LoopStatus.error       # "error"     -- a phase raised an exception
```

### PhaseMetric

Timing and execution info for a single pipeline phase.

```python
from definable.agent.pipeline import PhaseMetric

metric = PhaseMetric(
  phase_name="invoke_loop",  # Phase that was executed
  duration_ms=234.5,         # Execution time in milliseconds
  skipped=False,             # True if phase was skipped (should_run=False)
)
```

### LoopState

Mutable state that flows through all pipeline phases. Each phase reads and writes fields relevant to its concern. This is the single source of truth for the entire run.

**Key fields:**

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | `str` | Unique run identifier |
| `session_id` | `str` | Session scope for memory |
| `user_id` | `str \| None` | User scope |
| `agent_id` | `str` | Agent identifier |
| `agent_name` | `str` | Agent display name |
| `raw_instruction` | `str \| Message \| list[Message] \| None` | Original input |
| `new_messages` | `list[Message]` | Messages from this turn |
| `all_messages` | `list[Message]` | Full conversation history |
| `invoke_messages` | `list[Message]` | Messages sent to the model |
| `system_content` | `str` | Assembled system prompt |
| `tools` | `dict[str, Function]` | Available tools |
| `content` | `str \| None` | Final output content |
| `status` | `LoopStatus` | Current execution status |
| `phase` | `str` | Currently executing phase name |
| `turn` | `int` | Current turn number |
| `knowledge_context` | `str \| None` | Retrieved knowledge |
| `memory_context` | `str \| None` | Retrieved memory |
| `thinking_output` | `ThinkingOutput \| None` | Reasoning results |
| `tool_executions` | `list[ToolExecution]` | Tool call history |
| `phase_metrics` | `list[PhaseMetric]` | Timing data per phase |
| `streaming` | `bool` | Whether streaming is active |
| `extra` | `dict[str, Any]` | Arbitrary extra state for custom phases |

### Pipeline

The core orchestrator. Manages phase ordering, hooks, and execution.

```python
from definable.agent.pipeline import Pipeline

pipeline = Pipeline(
  phases=None,   # List[Phase] -- defaults to 8 standard phases
  debug=None,    # DebugConfig for breakpoints and inspection
)
```

**Phase Manipulation:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_phase` | `pipeline.add_phase(phase, after=None, before=None) -> Pipeline` | Insert a phase at a specific position (chainable) |
| `remove_phase` | `pipeline.remove_phase(name) -> Pipeline` | Remove a phase by name (chainable) |
| `replace_phase` | `pipeline.replace_phase(name, new_phase) -> Pipeline` | Replace a phase by name (chainable) |

**Properties:**

| Property | Return | Description |
|----------|--------|-------------|
| `phase_names` | `list[str]` | Ordered list of phase names |
| `phases` | `list[Phase]` | Read-only copy of phase list |
| `event_stream` | `EventStream` | Unified event stream |

```python
# Manipulate the phase list
pipeline.add_phase(MyPhase(), after="recall")
pipeline.remove_phase("think")
pipeline.replace_phase("store", CustomStorePhase())

print(pipeline.phase_names)
# ["prepare", "recall", "my_phase", "guard_input", "compose",
#  "invoke_loop", "guard_output", "store"]
```

### Hook System

Hooks allow per-call customization without modifying the phase list.

**Hook spec format:** `"{timing}:{phase_name}"`

| Timing | Description |
|--------|-------------|
| `before` | Runs before the phase executes |
| `after` | Runs after the phase completes |
| `instead` | Replaces the phase entirely (must yield `(state, event)` tuples) |

**Phase names:** `prepare`, `recall`, `think`, `guard_input`, `compose`, `invoke_loop`, `guard_output`, `store`, or `*` (wildcard -- matches all phases).

| Method | Signature | Description |
|--------|-----------|-------------|
| `hook` | `pipeline.hook(spec, callback=None, priority=0)` | Register a hook (decorator or direct call). Lower priority runs first |
| `remove_hook` | `pipeline.remove_hook(spec, callback=None) -> bool` | Remove a hook. If callback is None, removes all hooks for that spec |
| `subscribe` | `pipeline.subscribe(handler)` | Register an event stream handler |

```python
# Decorator style
@pipeline.hook("before:invoke_loop")
async def log_before(state):
  print(f"About to invoke with {len(state.invoke_messages)} messages")
  return state

# Direct call
async def log_after(state):
  print(f"Invoke completed: {state.content[:50] if state.content else 'empty'}")
  return state
pipeline.hook("after:invoke_loop", log_after, priority=10)

# Wildcard -- runs on every phase
pipeline.hook("before:*", my_universal_hook, priority=100)

# Remove a specific hook
pipeline.remove_hook("after:invoke_loop", log_after)

# Remove ALL hooks for a spec
pipeline.remove_hook("before:invoke_loop")
```

**Hook priority:** Lower numbers run first. Default is 0. Use positive numbers for late-running hooks and negative numbers for early-running hooks.

```python
pipeline.hook("before:invoke_loop", critical_hook, priority=-10)   # runs first
pipeline.hook("before:invoke_loop", normal_hook, priority=0)       # runs second
pipeline.hook("before:invoke_loop", cleanup_hook, priority=100)    # runs last
```

### 8 Default Phases

| Phase | Name | Description |
|-------|------|-------------|
| `PreparePhase` | `prepare` | Parse raw instruction, load tools, set identity |
| `RecallPhase` | `recall` | Fetch knowledge, memory, deep research context |
| `ThinkPhase` | `think` | Deliberate reasoning (skipped when thinking=never) |
| `GuardInputPhase` | `guard_input` | Run input guardrails (skipped when none configured) |
| `ComposePhase` | `compose` | Build system prompt, assemble invoke messages |
| `InvokeLoopPhase` | `invoke_loop` | Model invocation + tool dispatch loop |
| `GuardOutputPhase` | `guard_output` | Run output guardrails (skipped when none configured) |
| `StorePhase` | `store` | Persist to memory, emit final events |

### Phase Protocol and BasePhase

```python
from definable.agent.pipeline import Phase, BasePhase

# Phase is a Protocol -- any class with name + execute works
class MyPhase:
  @property
  def name(self) -> str:
    return "my_phase"

  async def execute(self, state):
    state.extra["custom_data"] = "hello"
    yield state, None  # yield (state, optional_event) tuples

# BasePhase adds should_run, requires, provides
class ConditionalPhase(BasePhase):
  _name = "conditional"
  _requires = {"invoke_messages"}  # fields this phase reads
  _provides = {"custom_output"}    # fields this phase writes

  def should_run(self, state):
    return state.extra.get("enabled", True)

  async def execute(self, state):
    state.extra["custom_output"] = "computed"
    yield state, None
```

### ToolRetry

Exception raised inside `@tool` functions to ask the model to retry with better arguments. The message is sent back as the tool result prefixed with `[RETRY]`.

```python
from definable.agent.pipeline import ToolRetry

raise ToolRetry(
  message="Query too short. Provide at least 3 characters.",  # Feedback for the model
  max_retries=3,  # Max retry attempts for this tool call (default 3)
)
```

```python
from definable.agent.pipeline import ToolRetry
from definable.tool.decorator import tool

@tool
def search(query: str) -> str:
  """Search the web for information."""
  if len(query) < 3:
    raise ToolRetry("Query too short. Provide at least 3 characters.", max_retries=3)
  if not query.strip():
    raise ToolRetry("Query is empty. Provide a meaningful search term.")
  return f"Results for: {query}"
```

**Flow:**
1. Model calls `search(query="ab")`
2. Tool raises `ToolRetry("Query too short...")`
3. Pipeline sends `[RETRY] Query too short...` as tool result
4. Model adjusts arguments and retries: `search(query="artificial intelligence")`
5. If max retries exceeded, the retry message becomes the final tool result

### DebugConfig

Fine-grained pipeline inspection configuration.

```python
from definable.agent.pipeline import DebugConfig

config = DebugConfig(
  breakpoints=set(),      # Set of hook specs where execution pauses (e.g. {"invoke_loop"})
  step_mode=False,        # If True, pause after every phase (overrides breakpoints)
  inspector=None,         # Callback: (state, phase_name) -> None (sync or async)
  log_state_changes=False,  # If True, diff state between phases and log changes
  enable_trace=True,      # If True, attach DebugExporter (color-coded stderr)
)
```

```python
from definable.agent import Agent
from definable.agent.pipeline import DebugConfig

# Breakpoints on specific phases
agent = Agent(
  model="openai/gpt-4o-mini",
  debug=DebugConfig(
    breakpoints={"invoke_loop", "guard_output"},
    inspector=lambda state, phase: print(f"Inspecting {phase}: {state.status}"),
  ),
)

# Step through every phase
agent = Agent(
  model="openai/gpt-4o-mini",
  debug=DebugConfig(step_mode=True, log_state_changes=True),
)
```

### SubAgentPolicy

Controls for sub-agent spawning. When enabled, the agent gets a `spawn_agent` tool that creates child agents at runtime.

```python
from definable.agent.pipeline import SubAgentPolicy

policy = SubAgentPolicy(
  max_concurrent=5,          # Max simultaneous sub-agents (default 5)
  max_tool_rounds=15,        # Tool rounds limit for child agents (default 15)
  inherit_tools=True,        # Child inherits parent's tools (default True)
  inherit_knowledge=False,   # Child inherits parent's knowledge (default False)
  allowed_models=None,       # Restrict child model choices (None = use parent's)
  on_spawn=None,             # Callback: (ThreadControlBlock) -> None
  on_complete=None,          # Callback: (ThreadControlBlock) -> None
)
```

### ThreadControlBlock

Tracks a spawned sub-agent execution (inspired by the Self-Manager paper).

```python
from definable.agent.pipeline import ThreadControlBlock

tcb = ThreadControlBlock(
  id="uuid",                  # Unique thread identifier
  goal="Research quantum computing",  # The subtask
  state="running",            # "running" | "completed" | "failed" | "killed"
  agent_config=None,          # SubAgentConfig (optional)
  start_time=1234567890.0,    # Unix timestamp
  result=None,                # Final output string (on completion)
  error=None,                 # Error message (on failure)
  metrics=None,               # Model metrics
  run_output=None,            # Full RunOutput for inspection
)
```

### EventStream

Unified event dispatch for pipeline execution. All consumers (tracing, EventBus, debug) subscribe here.

```python
from definable.agent.pipeline import EventStream

stream = EventStream()

async def my_handler(event):
  print(f"Event: {type(event).__name__}")

stream.subscribe(my_handler)
print(stream.handler_count)  # 1

# Emit events (best-effort -- handler errors are logged, not propagated)
await stream.emit(some_event)

# Remove a specific handler
stream.unsubscribe(my_handler)

# Remove all handlers
stream.clear()
```

## Patterns

### Custom Phase Insertion

```python
from definable.agent.pipeline import BasePhase

class AuditPhase(BasePhase):
  _name = "audit"
  _requires = {"content"}

  async def execute(self, state):
    if state.content:
      print(f"[audit] Output length: {len(state.content)}")
    yield state, None

# Insert between guard_output and store
agent.pipeline.add_phase(AuditPhase(), after="guard_output")
```

### Replacing a Phase

```python
class FastComposePhase(BasePhase):
  _name = "compose"

  async def execute(self, state):
    # Custom system prompt logic
    state.system_content = "You are a fast assistant."
    state.invoke_messages = list(state.all_messages)
    yield state, None

agent.pipeline.replace_phase("compose", FastComposePhase())
```

### Instead Hook (Phase Replacement via Hook)

```python
async def custom_invoke(state):
  # Complete replacement of the invoke_loop phase
  state.content = "Hardcoded response for testing"
  yield state, None

agent.pipeline.hook("instead:invoke_loop", custom_invoke)
```

### Event-Driven Monitoring

```python
from definable.agent.pipeline import EventStream

async def monitor(event):
  event_name = type(event).__name__
  if "Completed" in event_name:
    print(f"Phase completed: {getattr(event, 'phase_name', '?')}")

agent.pipeline.subscribe(monitor)
```

### Sub-Agent Spawning

```python
from definable.agent import Agent
from definable.agent.pipeline import SubAgentPolicy

agent = Agent(
  model="openai/gpt-4o",
  sub_agents=SubAgentPolicy(
    max_concurrent=3,
    inherit_tools=True,
    inherit_knowledge=False,
    on_spawn=lambda tcb: print(f"Spawned: {tcb.goal}"),
    on_complete=lambda tcb: print(f"Done: {tcb.state}"),
  ),
)

# The agent now has a spawn_agent tool it can use autonomously
```

## Gotchas

| Issue | Solution |
|-------|----------|
| Hook spec must be `"timing:phase_name"` | Valid timings: `before`, `after`, `instead`, `on`. Phase can be `*` for wildcard |
| `add_phase(after=X, before=Y)` | Raises `ValueError` -- specify only one anchor |
| Phase name not found | `add_phase`, `remove_phase`, `replace_phase` raise `ValueError` with available names |
| `instead` hooks must yield `(state, event)` | They replace the phase entirely and must be async generators |
| Hooks can return `None` | Returning `None` from a hook means "no state change" -- previous state is kept |
| Hook errors are logged but not propagated | A broken hook does not crash the pipeline |
| EventStream errors are best-effort | Handler exceptions are logged, not propagated |
| `should_run` returning False skips entirely | No hooks fire, no events emitted, just a `PhaseMetric(skipped=True)` |
| `ToolRetry` max_retries exceeded | The retry message becomes the final tool result (no exception) |
| DebugConfig breakpoints use phase names | Not hook specs. Use `{"invoke_loop"}` not `{"before:invoke_loop"}` |
| `enable_trace=True` is the default | Setting `debug=DebugConfig()` also attaches the color-coded DebugExporter |
| Sub-agents do not inherit memory or middleware | Child agents are minimal: model + tools + optional knowledge only |
| Sub-agent failures do not crash the parent | Errors are caught and returned as tool result strings |

## Related Modules

- **[Agent](../README.md)** -- Agent owns the Pipeline and exposes `agent.pipeline`
- **[Plugin](../plugin/README.md)** -- Plugins register hooks on the pipeline via `on_load`
- **[Tool](../../tool/README.md)** -- Tools raise `ToolRetry` for model-feedback retries
- **[Guardrail](../guardrail/README.md)** -- GuardInputPhase and GuardOutputPhase execute guardrails
- **[Security](../security/README.md)** -- SecurityConfig auto-injects guardrails into the pipeline
