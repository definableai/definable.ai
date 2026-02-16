<div align="center">

# Definable

**Build LLM agents that work in production.**

[![PyPI](https://img.shields.io/pypi/v/definable)](https://pypi.org/project/definable-ai/)
[![Python](https://img.shields.io/pypi/pyversions/definable)](https://pypi.org/project/definable-ai/)
[![License](https://img.shields.io/github/license/definable-ai/definable)](https://github.com/definable-ai/definable/blob/main/LICENSE)

[Documentation](https://definable.ai/docs) · [Examples](https://github.com/definable-ai/definable/tree/main/definable/examples) · [PyPI](https://pypi.org/project/definable-ai/)

</div>

A Python framework for building agent applications with tools, RAG, persistent memory, guardrails, skills, file readers, messaging platform integrations, and the Model Context Protocol. Switch providers without rewriting agent code.

---

## Install

```bash
pip install definable
```

## Quick Start

```python
from definable.agents import Agent
from definable.models.openai import OpenAIChat

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions="You are a helpful assistant.",
)

output = agent.run("What is the capital of Japan?")
print(output.content)  # "The capital of Japan is Tokyo."
```

## Add Tools

```python
from definable.tools.decorator import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 72°F in {city}"

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[get_weather],
    instructions="Help users check the weather.",
)

output = agent.run("What's the weather in Tokyo?")
```

The agent calls tools automatically. No manual function routing.

## Structured Output

```python
from pydantic import BaseModel

class WeatherReport(BaseModel):
    city: str
    temperature: float
    conditions: str

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[get_weather],
    instructions="Report weather data.",
)

output = agent.run("Weather in Tokyo?", output_schema=WeatherReport)
report = output.parsed  # WeatherReport(city="Tokyo", temperature=72.0, ...)
```

Pass any Pydantic model to `output_schema` and get validated, typed results back.

## Streaming

```python
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions="You are a helpful assistant.",
)

for event in agent.run_stream("Write a haiku about Python."):
    if event.content:
        print(event.content, end="", flush=True)
```

`run_stream()` yields events as they arrive — content chunks, tool calls, and completion signals.

## Multi-Turn Conversations

```python
output1 = agent.run("My name is Alice.")
output2 = agent.run("What's my name?", messages=output1.messages)
print(output2.content)  # "Your name is Alice."
```

Pass `messages` from a previous run to continue the conversation.

## Persistent Memory

```python
from definable.memory import CognitiveMemory, SQLiteMemoryStore

memory = CognitiveMemory(store=SQLiteMemoryStore("memory.db"))

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    memory=memory,
    instructions="You are a personal assistant.",
)

agent.run("My name is Alice and I prefer dark mode.", user_id="alice")
# Later, even in a new session...
agent.run("What's my name?", user_id="alice")  # Recalls "Alice"
```

Memory is automatic: the agent stores interactions and recalls relevant context on each turn. Eight store backends available (SQLite, PostgreSQL, Redis, Qdrant, Chroma, Pinecone, MongoDB, in-memory).

## Knowledge Base (RAG)

```python
from definable.knowledge import Knowledge, InMemoryVectorDB, Document
from definable.knowledge.embedders.openai import OpenAIEmbedder
from definable.agents import AgentConfig, KnowledgeConfig

kb = Knowledge(
    vector_db=InMemoryVectorDB(dimensions=1536),
    embedder=OpenAIEmbedder(),
)
kb.add(Document(content="Company vacation policy: 20 days PTO per year."))

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions="You are an HR assistant.",
    config=AgentConfig(knowledge=KnowledgeConfig(knowledge=kb, top_k=3)),
)

output = agent.run("How many vacation days do I get?")
```

The agent retrieves relevant documents before responding. Supports embedders (OpenAI, Voyage), vector DBs (in-memory, PostgreSQL), rerankers (Cohere), and chunkers.

## Guardrails

```python
from definable.guardrails import Guardrails, max_tokens, pii_filter, tool_blocklist

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions="You are a support agent.",
    tools=[get_weather],
    guardrails=Guardrails(
        input=[max_tokens(500)],
        output=[pii_filter()],
        tool=[tool_blocklist(["dangerous_tool"])],
    ),
)

output = agent.run("What's the weather?")
```

Guardrails check, modify, or block content at input, output, and tool-call checkpoints. Built-ins include token limits, PII redaction, topic blocking, and regex filters. Compose rules with `ALL`, `ANY`, `NOT`, and `when()`.

## Skills

```python
from definable.skills import Calculator, WebSearch, DateTime

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    skills=[Calculator(), WebSearch(), DateTime()],
    instructions="You are a helpful assistant.",
)

output = agent.run("What is 15% of 230?")
```

Skills bundle domain expertise (instructions) with tools. Built-in skills include Calculator, WebSearch, DateTime, HTTPRequests, JSONOperations, TextProcessing, Shell, and FileOperations. Create custom skills by subclassing `Skill`.

## File Readers

```python
from definable.media import File

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    readers=True,
    instructions="Summarize the uploaded document.",
)

output = agent.run("Summarize this.", files=[File(filepath="report.pdf")])
```

Pass `readers=True` to enable automatic parsing. Supports PDF, DOCX, PPTX, XLSX, ODS, RTF, HTML, images, and audio. AI-powered OCR available via Mistral, OpenAI, Anthropic, and Google providers.

## Deploy It

```python
from definable.triggers import Webhook, Cron
from definable.auth import APIKeyAuth

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions="You are a support agent.",
)

agent.on(Webhook(path="/support", method="POST"))
agent.on(Cron(schedule="0 9 * * *", instruction="Send the daily summary."))
agent.auth = APIKeyAuth(keys=["sk-my-secret-key"])
agent.serve(host="0.0.0.0", port=8000, dev=True)
```

`agent.serve()` starts an HTTP server with registered webhooks, cron triggers, and interfaces in a single process. Add `dev=True` for hot-reload during development.

## Connect to Platforms

```python
from definable.interfaces.telegram import TelegramInterface, TelegramConfig

telegram = TelegramInterface(
    config=TelegramConfig(bot_token="BOT_TOKEN"),
)

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions="You are a Telegram bot.",
)

agent.serve(telegram)
```

One agent, multiple platforms. Discord and Signal interfaces also available.

## MCP

```python
from definable.mcp import MCPConfig, MCPServerConfig, MCPToolkit

config = MCPConfig(
    servers=[
        MCPServerConfig(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        )
    ]
)

async with MCPToolkit(config) as toolkit:
    agent = Agent(model=OpenAIChat(id="gpt-4o-mini"), toolkits=[toolkit])
    await agent.arun("List files in /tmp")
```

Connect to any MCP server. Use the same tools as Claude Desktop.

## Replay & Compare

```python
from definable.agents import MockModel

# Inspect a past run
output = agent.run("Explain quantum computing.")
replay = agent.replay(run_output=output)
print(replay.steps)       # Each model call and tool invocation
print(replay.tokens)      # Token usage breakdown

# Re-run with a different model and compare
new_output = agent.replay(run_output=output, model=OpenAIChat(id="gpt-4o"))
comparison = agent.compare(output, new_output)
print(comparison.cost_diff)   # Cost difference between runs
print(comparison.token_diff)  # Token usage difference
```

Replay lets you inspect past runs, re-execute them with different models or instructions, and compare results side by side.

## Testing

```python
from definable.agents import Agent, MockModel

agent = Agent(
    model=MockModel(responses=["The capital of France is Paris."]),
    instructions="You are a geography expert.",
)

output = agent.run("What is the capital of France?")
assert "Paris" in output.content
```

`MockModel` returns canned responses — no API keys needed. Use it in unit tests to verify agent behavior deterministically.

---

## Features

| Category | Details |
|---|---|
| **Models** | OpenAI, DeepSeek, Moonshot, xAI, any OpenAI-compatible provider |
| **Agents** | Multi-turn conversations, structured output, configurable retries, max iterations |
| **Tools** | `@tool` decorator with automatic parameter extraction from type hints and docstrings |
| **Toolkits** | Composable tool groups, `KnowledgeToolkit` for explicit RAG search |
| **Skills** | Domain expertise + tools in one package; 8 built-in skills, custom `Skill` subclass |
| **Knowledge / RAG** | Embedders, vector DBs, rerankers (Cohere), chunkers, automatic retrieval |
| **Memory** | `CognitiveMemory` with multi-tier recall, distillation, topic prediction |
| **Memory Stores** | SQLite, PostgreSQL, Redis, Qdrant, Chroma, Pinecone, MongoDB, in-memory |
| **Readers** | PDF, DOCX, PPTX, XLSX, ODS, RTF, HTML, images, audio |
| **Reader Providers** | Mistral OCR, OpenAI, Anthropic, Google (AI-powered document parsing) |
| **Guardrails** | Input/output/tool checkpoints, PII redaction, token limits, topic blocking, regex filters |
| **Guardrails Composition** | `ALL`, `ANY`, `NOT`, `when()` combinators for complex policy rules |
| **Interfaces** | Telegram, Discord, Signal, session management, identity resolution |
| **Runtime** | `agent.serve()`, webhooks, cron triggers, event triggers, `dev=True` hot-reload |
| **Auth** | `APIKeyAuth`, `JWTAuth`, `AllowlistAuth`, `CompositeAuth`, pluggable `AuthProvider` protocol |
| **Streaming** | Real-time response and tool call streaming |
| **Replay** | Inspect past runs, re-execute with overrides, `agent.compare()` for side-by-side diffs |
| **Middleware** | Request/response transforms, logging, retry, metrics |
| **Tracing** | JSONL trace export for debugging and analysis |
| **Compression** | Automatic context window management for long conversations |
| **Testing** | `MockModel`, `AgentTestCase`, `create_test_agent` utilities |
| **MCP** | Model Context Protocol client for external tool servers |
| **Types** | Full Pydantic models, mypy verified |

## Supported Models

```python
from definable.models.openai import OpenAIChat      # GPT-4o, GPT-4o-mini, o1, o3, ...
from definable.models.deepseek import DeepSeekChat   # deepseek-chat, deepseek-reasoner
from definable.models.moonshot import MoonshotChat   # moonshot-v1-8k, moonshot-v1-128k
from definable.models.xai import xAIChat             # grok-2-latest
```

Any OpenAI-compatible API works with `OpenAIChat(base_url=..., api_key=...)`.

## Optional Extras

Install only what you need:

```bash
pip install definable[readers]          # PDF, DOCX, PPTX, XLSX, ODS, RTF parsers
pip install definable[serve]            # FastAPI + Uvicorn for agent.serve()
pip install definable[cron]             # Cron trigger support
pip install definable[jwt]              # JWT authentication
pip install definable[runtime]          # serve + cron combined
pip install definable[discord]          # Discord interface
pip install definable[signal]           # Signal interface
pip install definable[interfaces]       # All interface dependencies
pip install definable[postgres-memory]  # PostgreSQL memory store
pip install definable[redis-memory]     # Redis memory store
pip install definable[qdrant-memory]    # Qdrant memory store
pip install definable[chroma-memory]    # Chroma memory store
pip install definable[mongodb-memory]   # MongoDB memory store
pip install definable[pinecone-memory]  # Pinecone memory store
pip install definable[mistral-ocr]      # Mistral AI document parsing
pip install definable[mistral-ocr-images]  # Mistral OCR with image support
```

## Documentation

Full documentation: [definable.ai/docs](https://definable.ai/docs)

## Project Structure

```
definable/definable/
├── agents/        # Agent orchestration, config, middleware, tracing, testing
├── auth/          # APIKeyAuth, JWTAuth, AllowlistAuth, CompositeAuth
├── compression/   # Context window compression
├── guardrails/    # Input/output/tool policy, PII, token limits, composable rules
├── interfaces/    # Telegram, Discord, Signal integrations
├── knowledge/     # RAG: embedders, vector DBs, rerankers, chunkers
├── mcp/           # Model Context Protocol client
├── media.py       # Image, Audio, Video, File types
├── memory/        # CognitiveMemory + 8 store backends
├── models/        # OpenAI, DeepSeek, Moonshot, xAI providers
├── readers/       # File parsers + AI reader providers
├── reasoning/     # Reasoning capabilities
├── replay/        # Run inspection, re-execution, comparison
├── run/           # RunOutput, RunEvent types
├── runtime/       # AgentRuntime, AgentServer, dev mode
├── skills/        # Built-in + custom skills, skill registry
├── tokens.py      # Token counting utilities
├── tools/         # @tool decorator, tool wrappers
├── triggers/      # Webhook, Cron, EventTrigger
├── utils/         # Logging, supervisor, shared utilities
└── vectordbs/     # Vector database interfaces
```

## Contributing

Contributions welcome! To get started:

1. Fork the repo and clone it locally
2. Install for development: `pip install -e .`
3. Make your changes — follow existing code patterns
4. Add tests in `definable/tests_e2e/` for new features
5. Run `ruff check` and `ruff format` for linting
6. Run `mypy` for type checking
7. Open a pull request

See `definable/examples/` for usage patterns.

## License

MIT
