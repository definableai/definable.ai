# agent

The central module for building LLM-powered agents. Provides the `Agent` class plus 72 exports spanning orchestration, middleware, tracing, security, evaluation, multi-agent coordination, workflow, scheduling, and plugins.

## Quick Start

```python
from definable.agent import Agent
from definable.tool.decorator import tool

@tool
def get_weather(city: str) -> str:
  """Get the current weather for a city."""
  return f"The weather in {city} is sunny."

agent = Agent(
  model="openai/gpt-4o",
  tools=[get_weather],
  instructions="You are a helpful weather assistant.",
)

response = agent.run("What's the weather in Tokyo?")
print(response.content)
```

## Module Structure

```
agent/
├── __init__.py          # 72 public exports
├── agent.py             # Agent class
├── config.py            # AgentConfig, ReadersConfig
├── middleware.py         # Middleware protocol + built-in implementations
├── toolkit.py           # Toolkit base class
├── testing.py           # MockModel, AgentTestCase, create_test_agent
├── toolkits/            # KnowledgeToolkit
├── tracing/             # Tracing, JSONLExporter, DebugExporter, NoOpExporter
├── reasoning/           # Thinking
├── research/            # DeepResearch, DeepResearchConfig
├── guardrail/           # Guardrails, GuardrailResult
├── pipeline/            # Pipeline (8-phase execution backbone)
├── run/                 # RunOutput, RunContext
├── trigger/             # Webhook, Cron, EventTrigger, Interval, OneShot
├── interface/           # 9 platform interfaces (Telegram, Discord, Slack, Call, …)
├── auth/                # APIKeyAuth, JWTAuth, AllowlistAuth
├── compression/         # Tool result compression
├── replay/              # Trace replay + comparison
├── security/            # SecurityConfig, ToolPolicy, SSRF, rate limiting
├── eval/                # AccuracyEval, PerformanceEval, ReliabilityEval, AgentAsJudgeEval
├── team/                # Team multi-agent coordination
├── workflow/            # Workflow orchestration (Step, Parallel, Loop, …)
├── scheduler/           # Scheduler, ScheduledJob, JobStore
├── plugin/              # Plugin, PluginRegistry
└── runtime/             # AgentRuntime (HTTP server + lifecycle)
```

## API Reference

### Agent

The main orchestration class. Manages the model invocation loop, tool execution, middleware chain, memory, knowledge retrieval, security, and more.

```python
from definable.agent import Agent, AgentConfig

agent = Agent(
  # Identity
  name="my-agent",
  model="openai/gpt-4o",      # string shorthand or Model instance
  instructions="...",

  # Capabilities
  tools=[...],
  toolkits=[...],
  skills=[...],

  # Memory & knowledge
  memory=True,                 # InMemoryStore; or Memory(store=SQLiteStore(...))
  knowledge="./docs/",         # path shorthand; or Knowledge(vector_db=..., top_k=5)
  readers=True,                # file reading; or BaseReader instance

  # Reasoning
  thinking=True,               # inner monologue; or Thinking(...)
  deep_research=True,          # multi-wave research; or DeepResearchConfig(...)

  # Security
  security=True,               # default SecurityConfig; or SecurityConfig(...)

  # Observability
  tracing=Tracing(...),
  usage=True,                  # token + cost tracking; or UsageTracker(...)
  debug=False,                 # color-coded per-turn debug output

  # Audio
  audio_transcriber=True,      # Whisper transcription; or OpenAITranscriber(...)

  # Plugins
  plugins=[...],               # list[Plugin]

  # Guardrails & config
  guardrails=Guardrails(...),
  config=AgentConfig(...),
  session_id="...",
)
```

**Execution methods:**

| Method | Description |
|--------|-------------|
| `run(input)` | Synchronous multi-turn execution |
| `arun(input)` | Async execution with full middleware chain |
| `run_stream(input)` | Sync streaming, yields `RunOutputEvent`s |
| `arun_stream(input)` | Async streaming |

**Lifecycle methods:**

| Method | Description |
|--------|-------------|
| `use(middleware)` | Add middleware to the chain |
| `before_request(fn)` | Register a pre-execution hook |
| `after_response(fn)` | Register a post-execution hook |
| `on(trigger)` | Register a trigger handler (decorator) |
| `emit(event_name, data)` | Fire `EventTrigger`s (fire-and-forget) |
| `add_interface(interface)` | Register a messaging interface |
| `serve(...)` | Start sync runtime (server + interfaces + scheduler) |
| `aserve(...)` | Start async runtime |
| `security_audit()` | Run security checks, return `SecurityReport` |

### Configuration

```python
from definable.agent import AgentConfig, Compression, ReadersConfig
```

| Class | Purpose |
|-------|---------|
| `AgentConfig` | Frozen dataclass: identity, execution limits, retry, validation |
| `Compression` | Tool result compression (model, token/count limits) |
| `ReadersConfig` | File reader settings (registry, max content length) |

### Middleware

```python
from definable.agent import (
  Middleware,
  LoggingMiddleware,
  RetryMiddleware,
  MetricsMiddleware,
  KnowledgeMiddleware,
  StreamingMiddleware,
)
```

| Class | Description |
|-------|-------------|
| `Middleware` | Protocol: `async __call__(context, next_handler) -> RunOutput` |
| `LoggingMiddleware` | Logs run start, completion, and errors |
| `RetryMiddleware` | Exponential backoff retry on transient errors |
| `MetricsMiddleware` | Timing metrics (average latency, run/error counts) |
| `KnowledgeMiddleware` | RAG retrieval before model invocation |
| `StreamingMiddleware` | Streaming-aware wrapper (middleware chain is skipped in streaming; use this to participate) |

```python
agent = Agent(model="gpt-4o")
agent.use(LoggingMiddleware())
agent.use(RetryMiddleware(max_retries=3))
```

### Toolkit

```python
from definable.agent import Toolkit, KnowledgeToolkit
```

- `Toolkit` — Base class for grouping related tools. Override the `tools` property or attach `Function` attributes.
- `KnowledgeToolkit` — Provides `search_knowledge(query)` and `get_document_count()` tools for explicit RAG.

**Agent-managed lifecycle**: Async toolkits (like `MCPToolkit`) that implement `AsyncLifecycleToolkit` are automatically initialized and shut down by the agent — no manual `async with toolkit:` needed.

```python
from definable.agent import MCPToolkit

toolkit = MCPToolkit(config=config)
agent = Agent(model="gpt-4o", toolkits=[toolkit])
output = await agent.arun("List files")  # toolkit auto-initialized and shut down
```

### Tracing

```python
from definable.agent import (
  Tracing,
  TraceExporter,
  TraceWriter,
  JSONLExporter,
  NoOpExporter,
  DebugExporter,
)
from definable.agent.tracing import read_trace_file, read_trace_events
```

| Class | Description |
|-------|-------------|
| `TraceExporter` | Protocol: `export`, `flush`, `shutdown` |
| `JSONLExporter` | Writes per-session `.jsonl` trace files to a directory |
| `NoOpExporter` | Silent exporter for testing |
| `DebugExporter` | Prints color-coded turn-by-turn breakdowns to stdout |

```python
agent = Agent(
  model="gpt-4o",
  tracing=Tracing(exporters=[JSONLExporter("./traces")]),
)
# or enable per-turn debug output:
agent = Agent(model="gpt-4o", debug=True)
```

### Security

```python
from definable.agent import (
  SecurityConfig,
  ToolPolicy,
  SecurityReport,
  SecurityFinding,
  SecuritySeverity,
)
```

`security=True` (or a `SecurityConfig`) auto-injects guardrails — do not duplicate them manually.

| Class | Description |
|-------|-------------|
| `SecurityConfig` | Top-level security configuration |
| `ToolPolicy` | Tool access control: `deny` / `allowlist` / `full` modes |
| `SecurityReport` | Audit result: 0-100 score + list of `SecurityFinding`s |
| `SecurityFinding` | Individual finding with `SecuritySeverity` level |

```python
from definable.agent import Agent, SecurityConfig, ToolPolicy

agent = Agent(
  model="gpt-4o",
  security=SecurityConfig(
    tool_policy=ToolPolicy(mode="allowlist", allowed_tools={"search", "calculator"}),
  ),
)

# Run an audit
report = await agent.security_audit()
print(report.score)       # int 0-100
print(report.findings)    # list[SecurityFinding]
```

See [`security/README.md`](security/README.md) for full details (rate limiting, SSRF guard, prompt injection defense, env sanitization).

### Evaluation

```python
from definable.agent import (
  BaseEval,
  EvalCase,
  EvalSuite,
  AccuracyEval,
  PerformanceEval,
  ReliabilityEval,
  AgentAsJudgeEval,
  EvalResult,
  AccuracyResult,
  PerformanceResult,
  ReliabilityResult,
  JudgeResult,
)
```

| Class | Description |
|-------|-------------|
| `AccuracyEval` | LLM-judged scoring (1-10) against expected output, configurable threshold |
| `PerformanceEval` | Runtime + memory profiling via `tracemalloc`, p95 latency, warmup runs |
| `ReliabilityEval` | Tool call verification — checks required tools were called (strict/permissive) |
| `AgentAsJudgeEval` | Custom criteria evaluation, numeric or binary scoring modes |
| `EvalCase` | Single test case: `input`, `expected`, optional metadata overrides |
| `EvalSuite` | Batch result collection with `.pass_rate` |
| `BaseEval` | Abstract base — extend to create custom evaluators |

```python
from definable.agent import AccuracyEval, EvalCase

eval = AccuracyEval(judge_model="openai/gpt-4o-mini", threshold=7.0)

# Single case
result = await eval.arun(agent, EvalCase(input="What is 2+2?", expected="4"))
print(result.passed, result.score)

# Batch
suite = await eval.arun_batch(agent, [case1, case2, case3])
print(suite.pass_rate)
```

See [`eval/README.md`](eval/README.md) for full details.

### Team

```python
from definable.agent import Team, TeamMode
```

Multi-agent coordination — a leader model delegates to specialist members.

| Mode | Behavior |
|------|----------|
| `coordinate` | Leader picks the best member(s) per request |
| `route` | Leader routes to exactly one specialist |
| `collaborate` | All members run in parallel; leader synthesizes |
| `tasks` | Leader decomposes into a dependency-tracked `TaskList` |

```python
from definable.agent import Agent, Team, TeamMode

researcher = Agent(model="openai/gpt-4o", instructions="Research specialist.")
writer = Agent(model="openai/gpt-4o", instructions="Technical writer.")

team = Team(
  name="content-team",
  model="openai/gpt-4o",
  members=[researcher, writer],
  mode=TeamMode.coordinate,
  instructions="Produce well-researched technical content.",
)
result = await team.arun("Write about quantum computing")
print(result.content)
```

See [`team/README.md`](team/README.md) for full details.

### Workflow

```python
from definable.agent import (
  Workflow,
  Step,
  Steps,
  Parallel,
  Loop,
  Condition,
  Router,
)
```

Composable multi-step orchestration. Each `Step` wraps an agent, team, or callable.

| Type | Behavior |
|------|----------|
| `Step` | Single execution unit wrapping `agent=`, `team=`, or `executor=` |
| `Steps` | Sequential — each step receives the previous step's output |
| `Parallel` | Concurrent — all steps run simultaneously |
| `Loop` | Iterative — runs until `end_condition` returns `True` or `max_iterations` |
| `Condition` | Branching — routes to `true_steps` or `false_steps` based on a predicate |
| `Router` | N-way routing — `selector` function returns a route key |

```python
from definable.agent import Agent, Workflow, Step, Parallel, Loop

researcher = Agent(model="gpt-4o", instructions="Researcher.")
writer = Agent(model="gpt-4o", instructions="Writer.")
reviewer = Agent(model="gpt-4o", instructions="Reviewer.")

workflow = Workflow(
  name="research-pipeline",
  steps=[
    Step(name="research", agent=researcher),
    Step(name="write", agent=writer),
    Step(name="review", agent=reviewer),
  ],
)
result = await workflow.arun("Write about quantum computing")
print(result.content)
print(result.get_step_output("research"))
```

See [`workflow/README.md`](workflow/README.md) for full details.

### Scheduler

```python
from definable.agent import Scheduler, ScheduledJob, JobStatus, Interval, OneShot
```

Schedule agents to run on triggers without a live web server.

| Class | Description |
|-------|-------------|
| `Scheduler` | Tick loop with semaphore-based concurrency, pluggable `JobStore` |
| `ScheduledJob` | Lifecycle state machine: `pending → active → paused / completed / cancelled / failed` |
| `JobStatus` | Enum of job states |
| `Interval` | Trigger that fires repeatedly on a fixed interval |
| `OneShot` | Trigger that fires exactly once at a target time |

```python
from definable.agent import Agent, Interval, OneShot

# Attach triggers directly to the agent
agent = Agent(
  model="gpt-4o",
  instructions="Run the daily briefing.",
  triggers=[Interval(seconds=3600)],    # fires every hour
)
await agent.aserve()
```

See [`scheduler/README.md`](scheduler/README.md) and [`trigger/README.md`](trigger/README.md) for full details.

### Pipeline

```python
from definable.agent import (
  Pipeline,
  Phase,
  BasePhase,
  LoopState,
  LoopStatus,
  ToolRetry,
  DebugConfig,
  SubAgentPolicy,
)
```

The 8-phase execution backbone that powers every `agent.arun()` call. Most users do not need to touch this directly — it is exposed for advanced customization via hooks.

**Phases (in order):** `Prepare → Recall → Think → GuardInput → Compose → InvokeLoop → GuardOutput → Store`

| Class | Description |
|-------|-------------|
| `Pipeline` | Orchestrates all 8 phases; accepts `before_*/after_*/instead:*` hooks |
| `Phase` | Enum of phase names |
| `BasePhase` | Abstract base for custom phase implementations |
| `LoopState` | Typed state passed between phases |
| `LoopStatus` | Enum: `running`, `done`, `error`, `cancelled` |
| `ToolRetry` | Model-feedback retry logic for `@tool` functions |
| `DebugConfig` | Breakpoints and step-mode for the pipeline |
| `SubAgentPolicy` | Concurrency policy for parallel child agents |

See [`pipeline/README.md`](pipeline/README.md) for full details.

### Plugins

```python
from definable.agent import Plugin, PluginRegistry
```

The plugin system provides a structured extension point for adding capabilities to agents without modifying the core.

```python
from definable.agent import Plugin, PluginRegistry, Agent

class MyPlugin(Plugin):
  name = "my-plugin"

  def install(self, agent: Agent) -> None:
    # Attach tools, middleware, hooks, etc.
    ...

agent = Agent(model="gpt-4o", plugins=[MyPlugin()])
```

See [`plugin/README.md`](plugin/README.md) for full details.

### Usage Tracking

```python
from definable.agent import UsageTracker, UsageSnapshot
```

Token and cost accounting attached to an agent session.

```python
agent = Agent(model="gpt-4o", usage=True)
output = await agent.arun("Hello")

snap = agent.usage_tracker.session_total  # UsageSnapshot
print(snap.total_tokens, snap.total_cost_usd)
```

| Class | Description |
|-------|-------------|
| `UsageTracker` | Accumulates token counts and cost estimates across all model calls |
| `UsageSnapshot` | Immutable point-in-time snapshot: `prompt_tokens`, `completion_tokens`, `total_tokens`, `total_cost_usd` |

### Cancellation

```python
from definable.agent import AgentCancelled, CancellationToken
```

- `CancellationToken` — Pass into `arun(cancellation_token=token)` to cancel a running agent from outside.
- `AgentCancelled` — Exception raised when the token is triggered mid-run.

### Guardrails

```python
from definable.agent import Guardrails, GuardrailResult
```

Input, output, and tool-call validation. When `security=SecurityConfig(...)` is set, `ToolPolicy` and `ContentDefense` auto-inject the appropriate guardrails — do not add them again manually.

See [`guardrail/README.md`](guardrail/README.md) for full details.

### Reasoning

```python
from definable.agent import Thinking
```

- `Thinking` — Enables an inner-monologue reasoning phase before the main model call. Pass `thinking=True` for defaults or `thinking=Thinking(...)` for custom configuration.

See [`reasoning/README.md`](reasoning/README.md) for full details.

### Replay

```python
from definable.agent import Replay, ReplayComparison
```

Re-execute past runs from trace files and compare outputs.

See [`replay/README.md`](replay/README.md) for full details.

### Testing

```python
from definable.agent import MockModel, AgentTestCase, create_test_agent
```

| Class / Function | Description |
|------------------|-------------|
| `MockModel(responses=[], tool_calls=[])` | Deterministic model mock. Has `assert_called()`, `assert_called_times(n)`, `call_history`. |
| `AgentTestCase` | Base test class with `assert_tool_called()`, `assert_no_errors()`, `assert_content_contains()`. |
| `create_test_agent(responses=[], tools=[])` | Convenience factory that returns a fully wired test agent. |

**Note:** Use `len(mock.call_history)` rather than `mock.call_count` — `call_count` is not incremented when `side_effect` is set.

## See Also

| Sub-module | README |
|------------|--------|
| `auth/` | [`auth/README.md`](auth/README.md) — APIKeyAuth, JWTAuth, AllowlistAuth |
| `compression/` | [`compression/README.md`](compression/README.md) — Tool result compression |
| `eval/` | [`eval/README.md`](eval/README.md) — Evaluation framework |
| `guardrail/` | [`guardrail/README.md`](guardrail/README.md) — Input / output / tool guardrails |
| `interface/` | [`interface/README.md`](interface/README.md) — Telegram, Discord, Slack, Call, Desktop, CLI |
| `pipeline/` | [`pipeline/README.md`](pipeline/README.md) — 8-phase execution pipeline |
| `plugin/` | [`plugin/README.md`](plugin/README.md) — Plugin system |
| `reasoning/` | [`reasoning/README.md`](reasoning/README.md) — Thinking / inner monologue |
| `replay/` | [`replay/README.md`](replay/README.md) — Trace replay and comparison |
| `research/` | [`research/README.md`](research/README.md) — Deep research pipeline |
| `run/` | [`run/README.md`](run/README.md) — RunOutput, RunContext |
| `runtime/` | [`runtime/README.md`](runtime/README.md) — HTTP server and AgentRuntime |
| `scheduler/` | [`scheduler/README.md`](scheduler/README.md) — Scheduler, ScheduledJob, JobStore |
| `security/` | [`security/README.md`](security/README.md) — SecurityConfig, ToolPolicy, SSRF, rate limiting |
| `team/` | [`team/README.md`](team/README.md) — Multi-agent coordination |
| `trigger/` | [`trigger/README.md`](trigger/README.md) — Webhook, Cron, Interval, OneShot |
| `workflow/` | [`workflow/README.md`](workflow/README.md) — Workflow orchestration |
| `../model/` | LLM provider implementations (10 providers) |
| `../tool/` | `@tool` decorator and `Function` class |
| `../knowledge/` | RAG pipeline, hybrid search, scoring |
| `../memory/` | Conversation memory and stores |
| `../reader/` | File content extraction |
| `../mcp/` | MCP toolkit and client |
| `../examples/` | Runnable examples |
