# API Surface — Correct Signatures & Import Paths

> Load this doc when writing examples, tests, or any code that uses the public API.
> These are verified against eval run #5 (2026-02-20, 159/159 checks passed).

## Agent

```python
from definable.agent import Agent, AgentConfig

agent = Agent(
    model="openai/gpt-4o-mini",     # string shorthand OR OpenAIChat(id="gpt-4o-mini")
    tools=[my_tool],                 # List[Function]
    toolkits=[MCPToolkit(...)],      # List[Toolkit]
    skills=[Calculator()],           # List[Skill]
    instructions="...",              # str
    name="my-agent",                 # str
    memory=Memory(store=SQLiteStore("./memory.db")),  # or True
    knowledge=Knowledge(vector_db=InMemoryVectorDB(), top_k=5),  # NOT True
    thinking=True,                   # or Thinking(...)
    tracing=True,                    # or Tracing(exporters=[...])
    guardrails=Guardrails(input=[max_tokens(500)]),
    deep_research=True,              # or DeepResearchConfig(...)
    audio_transcriber=True,          # or OpenAITranscriber(language="en")
    security=SecurityConfig(...),    # or True for defaults
    usage=True,                      # or UsageTracker(...)
    config=AgentConfig(...),         # advanced settings
)

# Sync/async
result = agent.run("prompt", messages=[...], output_schema=MyModel)
result = await agent.arun("prompt", messages=[...], output_schema=MyModel)

# Multi-turn: pass messages, NOT session_id alone
out2 = agent.run("follow up", messages=out1.messages)

# Middleware
agent.use(LoggingMiddleware(logger))
agent.use(RetryMiddleware(max_retries=3))
```

## Models

```python
from definable.model import OpenAIChat, DeepSeekChat, MoonshotChat, xAI, OpenAILike
from definable.model import Message, Metrics, ModelResponse, ToolExecution

model = OpenAIChat(id="gpt-4o-mini")
response = model.invoke(
    messages=[Message(role="user", content="Hello")],
    assistant_message=Message(role="assistant", content="")  # REQUIRED
)
```

String shorthand providers (10): `openai`, `deepseek`, `moonshot`, `xai`, `anthropic`, `mistral`, `google`, `perplexity`, `ollama`, `openrouter`
Bare model names default to OpenAI: `Agent(model="gpt-4o-mini")` → `OpenAIChat(id="gpt-4o-mini")`

## Tools

```python
from definable.tool import tool, Function

@tool
def my_tool(arg: str) -> str:
    """Tool description used by the model."""
    return result
```

## Knowledge & RAG

```python
from definable.knowledge import Knowledge, Document, Reader, ReaderConfig
from definable.embedder import OpenAIEmbedder, VoyageAIEmbedder
from definable.vectordb import InMemoryVectorDB, PgVector, Qdrant, ChromaDb

# Document uses meta_data (NOT metadata)
doc = Document(content="...", meta_data={"source": "file.pdf"})

knowledge = Knowledge(vector_db=InMemoryVectorDB(), top_k=5)

# Path shorthand — auto-configures full RAG pipeline
agent = Agent(model=model, knowledge="./docs/")
```

## Memory

```python
from definable.memory import Memory, InMemoryStore, SQLiteStore, FileStore

memory = Memory(store=SQLiteStore("./memory.db"))
# memory=True → InMemoryStore (for quick testing)
```

## Guardrails

```python
from definable.agent.guardrail import Guardrails, GuardrailResult
from definable.agent.guardrail import InputGuardrail, OutputGuardrail, ToolGuardrail
from definable.agent.guardrail import max_tokens, block_topics, regex_filter
from definable.agent.guardrail import pii_filter, max_output_tokens  # pii_filter is OUTPUT
from definable.agent.guardrail import tool_allowlist, tool_blocklist
from definable.agent.guardrail import ALL, ANY, NOT, when
```

## MCP

```python
from definable.mcp import MCPToolkit, MCPConfig, MCPServerConfig, MCPClient
# Use config object, NOT individual params
toolkit = MCPToolkit(config=MCPConfig(...))
```

## Tracing

```python
from definable.agent.tracing import Tracing, JSONLExporter, read_trace_file
```

## Skills

```python
from definable.skill import Skill, Calculator, DateTime, FileOperations
from definable.skill import HTTPRequests, JSONOperations, Shell, TextProcessing
from definable.skill import WebSearch, MacOS, SkillRegistry
```

## Call Interface (Voice)

```python
from definable.agent.interface.call import CallInterface, CallConfig, CallSession

# Managed mode (Twilio only — simplest, ~500ms latency)
call = CallInterface(
    agent=agent,
    provider="twilio",          # or "plivo" (cascading/realtime only)
    phone_number="+15551234567",
    pipeline="managed",         # "managed" | "cascading" | "realtime"
    welcome_message="Hello!",
)

# Cascading mode (pluggable STT/TTS, works with Twilio and Plivo)
from definable.agent.interface.call.stt.deepgram import DeepgramSTT
from definable.agent.interface.call.tts.cartesia import CartesiaTTS
call = CallInterface(
    agent=agent, provider="twilio", phone_number="+1555",
    pipeline="cascading",
    stt=DeepgramSTT(model="nova-3"),
    tts=CartesiaTTS(model="sonic-2", voice_id="..."),
)

# Realtime mode (OpenAI speech-to-speech, ~200-300ms latency)
from definable.agent.interface.call import OpenAIRealtimeProvider
call = CallInterface(
    agent=agent, provider="twilio", phone_number="+1555",
    pipeline="realtime",
    realtime=OpenAIRealtimeProvider(model="gpt-4o-realtime-preview", voice="alloy"),
)

# Plivo does NOT support managed mode — only cascading or realtime
# CallInterface(provider="plivo", pipeline="managed") → ValueError
```

## Auth

```python
from definable.agent.auth import APIKeyAuth, JWTAuth, AllowlistAuth
auth = APIKeyAuth(keys={"key1", "key2"})    # NOT api_keys
auth = AllowlistAuth(user_ids={"user1"})     # NOT allowed_ids
```

## Testing

```python
from definable.agent import MockModel, create_test_agent, AgentTestCase
# MockModel gotcha: call_count NOT incremented with side_effect
# Use len(mock_model.call_history) instead
```

## Slack Interface

```python
from definable.agent.interface import SlackInterface, SlackConfig

# Socket Mode (development)
slack = SlackInterface(
    agent=agent,
    bot_token="xoxb-...",     # required
    app_token="xapp-...",     # required for socket mode
    mode="socket",            # default
)

# HTTP Events API (production)
slack = SlackInterface(
    agent=agent,
    bot_token="xoxb-...",
    signing_secret="...",     # required for http mode
    mode="http",
)

# Interactive callbacks (fluent API)
slack.on_command("/status", handler)
slack.on_action("button_clicked", handler)
slack.on_view("modal_submitted", handler)
slack.on_shortcut("quick_action", handler)
slack.on_reaction_added(handler)
slack.on_home_opened(handler)
slack.on_event("channel_created", handler)

# Block Kit builders
from definable.agent.interface.slack.formatter import (
    header_block, section_block, actions_block, button_element,
    modal_view, home_tab_view, markdown_to_mrkdwn,
)

# API methods
await slack.update_message(channel, ts, text="...")
await slack.send_ephemeral(channel, user, text)
await slack.send_blocks(channel, blocks)
await slack.open_modal(trigger_id, view)
await slack.publish_home(user_id, view)
await slack.schedule_message(channel, text, post_at)
```

## Audio Transcription

```python
from definable import Agent, OpenAITranscriber

# Auto-transcribe voice notes from interfaces (uses Whisper API)
agent = Agent(model="openai/gpt-4o-mini", audio_transcriber=True)

# Custom transcriber with language hint
agent = Agent(model="openai/gpt-4o-mini", audio_transcriber=OpenAITranscriber(language="en"))
```

Format normalization (Telegram OGA, Discord OGG → WAV/MP3):

```python
from definable.reader.audio import normalize_audio_format, OPENAI_INPUT_AUDIO_FORMATS

# Converts oga/ogg/opus → wav via ffmpeg if needed
out_bytes, out_fmt = normalize_audio_format(raw_bytes, "oga")
```

## Evaluation

```python
from definable.agent.eval import (
    BaseEval, EvalCase, EvalSuite,
    AccuracyEval, PerformanceEval, ReliabilityEval, AgentAsJudgeEval,
    EvalResult, AccuracyResult, PerformanceResult, ReliabilityResult, JudgeResult,
)

# Accuracy: LLM-judged output scoring (1-10)
eval = AccuracyEval(judge_model="openai/gpt-4o-mini", threshold=7.0)
result = await eval.arun(agent, EvalCase(input="What is 2+2?", expected="4"))

# Performance: latency + memory profiling
eval = PerformanceEval(duration_threshold_ms=5000, memory_threshold_mb=100, runs=3)

# Reliability: tool call verification
eval = ReliabilityEval(expected_tools=["search_web"], strict=False)

# Custom judge: numeric or binary mode
eval = AgentAsJudgeEval(criteria="Must be concise", mode="numeric", threshold=8.0)

# Batch evaluation
suite = await eval.arun_batch(agent, [EvalCase(...), EvalCase(...)])
# suite.pass_rate, suite.passed, suite.failed, suite.total

# Team evaluation
result = await eval.arun_team(team, EvalCase(...))
```

## Security

```python
from definable.agent.security import (
    SecurityConfig, ToolPolicy, ToolPolicyGuardrail, DEFAULT_DANGEROUS_TOOLS,
    RateLimitConfig, RateLimitHook, SlidingWindowRateLimiter,
    ContentDefenseConfig, ContentDefenseGuardrail, PromptInjectionDetector,
    InjectionScanResult, xml_wrap_content,
    SSRFGuard, SSRFGuardConfig, SSRFBlockedError, is_private_ip, resolve_and_check,
    EnvSanitizeConfig, DANGEROUS_ENV_VARS, sanitize_env, is_env_safe,
    SecurityReport, SecurityFinding, SecuritySeverity, security_audit,
)

# Unified config — all features optional
agent = Agent(model=model, security=SecurityConfig(
    tool_policy=ToolPolicy(mode="allowlist", allowed_tools={"search"}),
    rate_limit=RateLimitConfig(max_requests=10, window_seconds=60),
    content_defense=ContentDefenseConfig(injection_detection=True),
    ssrf_guard=SSRFGuardConfig(enabled=True),
    env_sanitize=EnvSanitizeConfig(),
))

# security=True → default SecurityConfig
agent = Agent(model=model, security=True)

# Security audit
report = await agent.security_audit()  # SecurityReport with score, findings
```

## Usage Tracking

```python
from definable.agent import UsageTracker, UsageSnapshot

# Enable via Agent constructor
agent = Agent(model=model, usage=True)
output = await agent.arun("Hello")

# Access tracking
tracker = agent.usage_tracker  # UsageTracker
print(tracker.session_total)   # UsageSnapshot (input_tokens, output_tokens, estimated_cost)
print(tracker.last_run)        # Most recent run snapshot
print(tracker.run_count)       # Number of recorded runs
```

## Knowledge — Hybrid Search & Scoring

```python
from definable.knowledge import (
    FTSIndex, HybridSearchConfig,  # Full-text + hybrid search
    TemporalDecay, MMRConfig,       # Scoring strategies
    FallbackEmbedder,               # Multi-provider embedder failover
)

# Hybrid search (vector + BM25)
fts = FTSIndex(db_path=":memory:")
await fts.initialize()
knowledge = Knowledge(
    vector_db=InMemoryVectorDB(),
    fts_index=fts,
    hybrid_config=HybridSearchConfig(vector_weight=0.6, text_weight=0.4),
)

# Temporal decay (exponential by age)
knowledge = Knowledge(vector_db=db, temporal_decay=TemporalDecay(half_life_days=30.0))

# MMR diversity (relevance vs diversity balance)
knowledge = Knowledge(vector_db=db, mmr=MMRConfig(lambda_param=0.7))

# Fallback embedder
from definable.embedder import OpenAIEmbedder, VoyageAIEmbedder
embedder = FallbackEmbedder(providers=[OpenAIEmbedder(), VoyageAIEmbedder()])
```

## Team

```python
from definable.agent.team import Team, TeamMode, Task, TaskList, TaskStatus

team = Team(
    name="my-team",
    model="openai/gpt-4o",                  # leader model (string shorthand or Model)
    members=[agent_a, agent_b],             # List[Agent | Team]
    mode=TeamMode.coordinate,               # coordinate | route | collaborate | tasks
    instructions="...",                      # leader instructions
    max_iterations=10,                       # tasks mode only
    share_member_interactions=False,         # share member outputs across delegates
    tools=[extra_tool],                      # additional leader tools
    output_schema=MyModel,                   # structured output
    debug=False,
)

result = await team.arun("instruction", session_id="...", user_id="...")
# Returns RunOutput (same as agent.arun)

# Events
team.events.on(MemberDelegatedEvent, handler)  # EventBus with .on() method

# Also from top-level
from definable.agent import Team, TeamMode
```

## Workflow

```python
from definable.agent.workflow import (
    Workflow, Step, Steps, Parallel, Loop, Condition, Router,
    StepInput, StepOutput, WorkflowOutput,
)

workflow = Workflow(
    name="pipeline",
    steps=[                                  # List[Step] or single BaseStep
        Step(name="s1", agent=agent_a),
        Step(name="s2", agent=agent_b),
    ],
    session_state={"key": "value"},         # shared state across steps
    debug=False,
)

result: WorkflowOutput = await workflow.arun("input")
# result.content, result.success, result.duration_ms
# result.get_step_output("s1") → StepOutput
# result.get_step_content("s1") → str

# Step types:
Step(name="x", agent=agent)                     # single agent/team/callable
Steps(steps=[...])                               # sequential (chaining context)
Parallel(steps=[...], max_concurrency=3)         # concurrent (same input)
Loop(steps=[...], end_condition=fn, max_iterations=5)  # iterative
Condition(condition=fn, true_steps=s1, false_steps=s2) # if/else branching
Router(selector=fn, routes={"a": s1, "b": s2})  # N-way routing

# Events
workflow.events.on(StepCompletedEvent, handler)

# Also from top-level
from definable.agent import Workflow, Step, Steps, Parallel, Loop, Condition, Router
```

## Desktop Events

```python
from definable.agent.interface.desktop import BridgeCallEvent, DesktopActionEvent

# Events emitted by BridgeClient when on_event callback is set
# BridgeCallEvent: endpoint, method, status_code, duration_ms, error
# DesktopActionEvent: category, action, target, value, result, error
```

## Known Gotchas
- `knowledge=True` → ValueError (unlike memory=True which works)
- `pii_filter()` is OUTPUT guardrail, not input
- `Document(meta_data={})` not `metadata`
- `output_schema` not `response_model` for structured output
- sync `run()` breaks after 2-3 sequential multi-turn calls
- `InMemoryVectorDB(dimensions=N)` — dimensions param deprecated/ignored
- `CallInterface(provider="plivo", pipeline="managed")` → ValueError (Plivo has no ConversationRelay)
- `FTSIndex` requires explicit `await fts.initialize()` before use
- `FallbackEmbedder(providers=[])` → ValueError (requires at least one provider)
- `ToolPolicy(mode="allowlist")` with no `allowed_tools` blocks all tools
