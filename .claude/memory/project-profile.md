# Project Profile -- Definable v0.2.8

> Last updated: 2026-02-19 (eval run #4, post-DX overhaul)

## Package Info
- **Name**: definable
- **Version**: 0.2.8
- **Python**: >=3.12 (3.12.10 in .venv)
- **Source**: `definable/definable/` (215+ .py files)
- **Tests**: `definable/tests/` (700+ tests)

## Key Correct Import Paths (post-DX overhaul)

These are the ACTUAL working imports verified by eval run #4:

```python
# Agents
from definable.agent import Agent, AgentConfig, MockModel, create_test_agent, AgentTestCase
from definable.agent import Tracing, JSONLExporter, NoOpExporter, TraceExporter, TraceWriter
from definable.agent import LoggingMiddleware, RetryMiddleware, MetricsMiddleware, Middleware
from definable.agent import StreamingMiddleware, KnowledgeMiddleware
from definable.agent import Toolkit, KnowledgeToolkit, MCPToolkit
from definable.agent import AgentCancelled, CancellationToken, EventBus
from definable.agent import CompressionConfig, ReadersConfig, DeepResearchConfig
from definable.agent import Memory, MemoryManager, Thinking, Guardrails, GuardrailResult
from definable.agent import Replay, ReplayComparison

# Models
from definable.model import OpenAIChat, DeepSeekChat, MoonshotChat, xAI, OpenAILike
from definable.model import Message, Metrics, ModelResponse, ToolExecution

# Tools
from definable.tool import tool, Function
from definable.tool.decorator import tool  # same as above, explicit path

# Skills
from definable.skill import Skill, Calculator, DateTime, FileOperations, HTTPRequests
from definable.skill import JSONOperations, MacOS, Shell, TextProcessing, WebSearch
from definable.skill import SkillRegistry  # lazy import

# Knowledge
from definable.knowledge import Knowledge, Document, Reader, ReaderConfig
from definable.embedder import Embedder, OpenAIEmbedder, VoyageAIEmbedder

# VectorDB
from definable.vectordb import InMemoryVectorDB, VectorDB, Distance, SearchType

# Memory
from definable.memory import Memory, MemoryManager, InMemoryStore, SQLiteStore, MemoryStore

# Guardrails
from definable.agent.guardrail import Guardrails, GuardrailResult
from definable.agent.guardrail import InputGuardrail, OutputGuardrail, ToolGuardrail
from definable.agent.guardrail import max_tokens, block_topics, regex_filter
from definable.agent.guardrail import pii_filter, max_output_tokens
from definable.agent.guardrail import tool_allowlist, tool_blocklist
from definable.agent.guardrail import ALL, ANY, NOT, when

# MCP
from definable.mcp import MCPToolkit, MCPConfig, MCPServerConfig, MCPClient

# Tracing
from definable.agent.tracing import Tracing, JSONLExporter, read_trace_file, read_trace_events

# Events
from definable.agent.events import RunOutput, RunContext, RunStatus, RunInput
from definable.agent.events import RunStartedEvent, RunCompletedEvent, RunContentEvent

# Exceptions
from definable.exceptions import AgentRunException, StopAgentRun, RetryAgentRun
from definable.exceptions import DefinableError, ModelAuthenticationError, ModelProviderError
```

## Agent API (post-DX)

```python
agent = Agent(
    model="gpt-4o-mini",            # string shorthand OR OpenAIChat(id="gpt-4o-mini")
    tools=[...],                     # List[Function] from @tool
    toolkits=[...],                  # List[Toolkit|MCPToolkit]
    skills=[...],                    # List[Skill]
    instructions="...",              # str
    name="my-agent",                 # str -> config.agent_name
    memory=Memory(store=SQLiteStore("./memory.db")),  # or memory=True for InMemoryStore
    knowledge=Knowledge(vector_db=InMemoryVectorDB(), top_k=5),  # knowledge=True raises ValueError!
    thinking=True,                   # or Thinking(...)
    tracing=True,                    # or Tracing(exporters=[JSONLExporter(...)])
    guardrails=Guardrails(input=[max_tokens(500)], output=[pii_filter()]),
    deep_research=True,              # or DeepResearchConfig(...)
    config=AgentConfig(...),         # Optional advanced settings
)

# Run (sync or async)
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
- `InMemoryVectorDB(dimensions=N)` is deprecated, dimensions param ignored
- `Document(meta_data={})` -- note: meta_data NOT metadata
- sync `run()` breaks after 2-3 sequential multi-turn calls (#19)
- `Agent(model=None)` silently accepts None (#18)
- `output_schema` not `response_model` for structured output

## Static Analysis
- mypy: 0 errors
- ruff check: 0 warnings
- ruff format: 0 issues
