# Project Profile -- Definable v0.3.2

> Last updated: 2026-02-25 (post-docs-agent update)

## Package Info
- **Name**: definable
- **Version**: 0.3.2 (editable install)
- **Python**: >=3.12 (3.12.10 in .venv)
- **Source**: `definable/definable/` (400+ .py files)
- **Tests**: `definable/tests/` (3097+ tests, 122 test files)

## Key Correct Import Paths (verified eval run #6)

```python
# Top-level convenience
from definable import Agent, tool, Function, Message, RunOutput, Skill, Toolkit
from definable import OpenAIChat, DeepSeekChat, MoonshotChat, xAI
from definable import Perplexity, OpenRouter  # no extra deps
from definable import Memory, Knowledge, Document, Guardrails, Thinking, Tracing
from definable import Pipeline, ToolRetry, DebugConfig, SubAgentPolicy
from definable import AgentRunException, StopAgentRun, RetryAgentRun

# Security
from definable.agent.security import (
    SecurityConfig, ToolPolicy, ToolPolicyGuardrail, DEFAULT_DANGEROUS_TOOLS,
    RateLimitConfig, RateLimitHook, SlidingWindowRateLimiter,
    ContentDefenseConfig, ContentDefenseGuardrail, PromptInjectionDetector,
    SSRFGuard, SSRFGuardConfig, SSRFBlockedError, security_audit, SecurityReport,
    EnvSanitizeConfig, sanitize_env, is_env_safe,
)

# Evaluation
from definable.agent.eval import (
    BaseEval, EvalCase, EvalSuite,
    AccuracyEval, PerformanceEval, ReliabilityEval, AgentAsJudgeEval,
    EvalResult, AccuracyResult, PerformanceResult, ReliabilityResult, JudgeResult,
)

# Usage Tracking
from definable.agent import UsageTracker, UsageSnapshot

# Knowledge — Hybrid Search & Scoring
from definable.knowledge import FTSIndex, HybridSearchConfig, TemporalDecay, MMRConfig, FallbackEmbedder

# Team
from definable.agent.team import Team, TeamMode, Task, TaskList, TaskStatus
from definable.agent import Team, TeamMode  # also from top-level

# Workflow
from definable.agent.workflow import (
    Workflow, Step, Steps, Parallel, Loop, Condition, Router,
    StepInput, StepOutput, WorkflowOutput,
)
from definable.agent import Workflow, Step, Steps, Parallel, Loop, Condition, Router  # also from top-level

# Desktop Events
from definable.agent.interface.desktop import BridgeCallEvent, DesktopActionEvent

# WhatsApp Interface
from definable.agent.interface.whatsapp import WhatsAppInterface, WhatsAppPolicy
from definable.agent.interface.whatsapp.provider import (
    WhatsAppProvider, InboundMessage, OutboundMessage,
    PollMessage, ReactionMessage, SendResult,
    ConnectionStatus, QRLoginResult,
)
from definable.agent.interface.whatsapp.normalize import normalize_e164, redact_phone
from definable.agent.interface.whatsapp.formatting import markdown_to_whatsapp

# Agents
from definable.agent import Agent, AgentConfig, MockModel, create_test_agent, AgentTestCase
from definable.agent import Tracing, JSONLExporter, NoOpExporter, DebugExporter
from definable.agent import LoggingMiddleware, RetryMiddleware, MetricsMiddleware
from definable.agent import Pipeline, Phase, BasePhase, LoopState, LoopStatus

# Tools
from definable.tool import tool, Function

# Skills
from definable.skill import Skill, Calculator, DateTime, JSONOperations, TextProcessing
from definable.skill import FileOperations, HTTPRequests, Shell, WebSearch, MacOS

# Knowledge
from definable.knowledge import Knowledge, Document
from definable.embedder import Embedder, OpenAIEmbedder, VoyageAIEmbedder

# VectorDB
from definable.vectordb import InMemoryVectorDB, VectorDB

# Memory
from definable.memory import Memory, InMemoryStore, SQLiteStore, FileStore, MemoryEntry

# Guardrails
from definable.agent.guardrail import (
    Guardrails, GuardrailResult, InputGuardrail, OutputGuardrail, ToolGuardrail,
    max_tokens, block_topics, regex_filter, pii_filter, max_output_tokens,
    tool_allowlist, tool_blocklist, ALL, ANY, NOT, when,
)

# MCP
from definable.mcp import MCPToolkit, MCPConfig, MCPServerConfig, MCPClient

# Tracing
from definable.agent.tracing import Tracing, JSONLExporter, DebugExporter

# Pipeline
from definable.agent.pipeline import Pipeline, LoopState, ToolRetry, DebugConfig, SubAgentPolicy

# Events
from definable.agent.events import RunOutput, RunContext, RunStatus, RunInput

# Exceptions
from definable.exceptions import (
    AgentRunException, StopAgentRun, RetryAgentRun,
    DefinableError, ModelAuthenticationError, ModelProviderError,
    ModelRateLimitError, InputCheckError,
)
```

## Agent API (v0.3.2)

```python
agent = Agent(
    model="openai/gpt-4o-mini",         # string shorthand (provider/model)
    tools=[...],                          # List[Function] from @tool
    toolkits=[...],                       # List[Toolkit|MCPToolkit]
    skills=[...],                         # List[Skill]
    instructions="...",                   # str
    name="my-agent",                      # str -> config.agent_name
    memory=True,                          # or Memory(store=SQLiteStore("./memory.db"))
    knowledge=Knowledge(vector_db=InMemoryVectorDB(), top_k=5),  # knowledge=True raises ValueError!
    thinking=True,                        # or Thinking(...)
    tracing=True,                         # or Tracing(exporters=[JSONLExporter(...)])
    debug=True,                           # adds DebugExporter to tracing
    guardrails=Guardrails(input=[max_tokens(500)], output=[pii_filter()]),
    sub_agents=True,                      # or SubAgentPolicy(...)
    deep_research=True,                   # or DeepResearchConfig(...)
    observability=True,                   # or ObservabilityConfig(...)
    security=SecurityConfig(...),         # or True for defaults
    usage=True,                           # or UsageTracker(...)
    config=AgentConfig(...),              # Optional advanced settings
)

# Sync/async run
result = agent.run("prompt", messages=[...], output_schema=MyModel)
result = await agent.arun("prompt", messages=[...], output_schema=MyModel)

# Multi-turn
out2 = agent.run("follow up", messages=out1.messages)

# Middleware
agent.use(LoggingMiddleware(logger))
agent.use(RetryMiddleware(max_retries=3))
```

## Key Gotchas
- `knowledge=True` raises ValueError (unlike memory=True which works)
- `pii_filter()` is an OUTPUT guardrail, not input
- `Document(meta_data={})` -- note: meta_data NOT metadata
- Guardrail blocking raises `InputCheckError`, not RunOutput(status=blocked)
- Embedder abstract methods: `async_get_embedding` / `async_get_embedding_and_usage` (not aget_*)
- String model shorthand format: "provider/model-id" (e.g., "openai/gpt-4o-mini")
- `output_schema` not `response_model` for structured output
- System message sent as system-role Message in messages list (not separate kwarg)
- Multiple `asyncio.run()` calls can break HTTP connection pool -- use single async function
- `FTSIndex` requires explicit `await fts.initialize()` before use
- `FallbackEmbedder(providers=[])` raises ValueError (needs at least one provider)
- `ToolPolicy(mode="allowlist")` with no `allowed_tools` blocks all tools
- `security=True` auto-injects guardrails — don't duplicate ToolGuardrail/InputGuardrail
- `Team(model=None)` requires model on Team or at least one member, otherwise ValueError
- `Team.arun()` returns `RunOutput` (same as agent), NOT a TeamOutput
- `Workflow.arun()` returns `WorkflowOutput` (NOT RunOutput) — has `.step_outputs`, `.get_step_output(name)`
- `Step` needs exactly one of `agent=`, `team=`, or `executor=`
- `CLIInterface(mode="tui")` without textual raises ImportError — `pip install definable[cli]`
- `WhatsAppPolicy()` default `dm_policy="allowlist"` + empty `allow_from` blocks ALL senders
- `WhatsAppInterface(provider="baileys")` requires Node.js >= 18 + `pip install websockets`
- `WhatsAppInterface(provider="twilio")` requires `pip install httpx`

## Pipeline Architecture (8 phases)
Prepare -> Recall -> Think -> GuardInput -> Compose -> InvokeLoop -> GuardOutput -> Store

## Eval Run #6 Results (2026-02-25)
- 16 eval scripts, 305 checks: **302 passed, 0 failed, 3 skipped**
- Stability score: **10/10**
- All compositions tested: tools, skills, knowledge, memory, guardrails, tracing, MCP, full-stack
- Real API tests pass (OpenAI gpt-4o-mini, MCP filesystem)
