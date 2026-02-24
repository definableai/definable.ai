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

String shorthand providers: `openai`, `deepseek`, `moonshot`, `xai`

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

## Known Gotchas
- `knowledge=True` → ValueError (unlike memory=True which works)
- `pii_filter()` is OUTPUT guardrail, not input
- `Document(meta_data={})` not `metadata`
- `output_schema` not `response_model` for structured output
- sync `run()` breaks after 2-3 sequential multi-turn calls
- `InMemoryVectorDB(dimensions=N)` — dimensions param deprecated/ignored
