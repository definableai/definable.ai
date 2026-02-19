<div align="center">

<h1>Definable</h1>

<p><strong>Build LLM agents that work in production.</strong></p>

<p>
  <a href="https://pypi.org/project/definable/"><img src="https://img.shields.io/pypi/v/definable?color=%2334D058&label=pypi" alt="PyPI"></a>
  <a href="https://pypi.org/project/definable/"><img src="https://img.shields.io/pypi/pyversions/definable?color=%2334D058" alt="Python"></a>
  <a href="https://github.com/definableai/definable.ai/blob/main/LICENSE"><img src="https://img.shields.io/github/license/definableai/definable.ai?color=%2334D058" alt="License"></a>
  <a href="https://pypi.org/project/definable/"><img src="https://img.shields.io/pypi/dm/definable?color=%2334D058&label=downloads" alt="Downloads"></a>
  <a href="https://github.com/definableai/definable.ai/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/definableai/definable.ai/ci.yml?label=CI" alt="CI"></a>
</p>

<p>
  <a href="https://docs.definable.ai">Documentation</a> &nbsp;·&nbsp;
  <a href="https://github.com/definableai/definable.ai/tree/main/definable/examples">Examples</a> &nbsp;·&nbsp;
  <a href="https://pypi.org/project/definable/">PyPI</a>
</p>

</div>

<br>

A Python framework for building agent applications with tools, RAG, persistent memory, guardrails, skills, file readers, browser automation, messaging platform integrations, and the Model Context Protocol. Switch providers without rewriting agent code.

---

## Install

```bash
pip install definable
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv pip install definable
```

## Quick Start

```python
from definable.agent import Agent
from definable.model.openai import OpenAIChat

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions="You are a helpful assistant.",
)

output = agent.run("What is the capital of Japan?")
print(output.content)  # The capital of Japan is Tokyo.
```

Or use **string model shorthand** — no explicit import needed:

```python
from definable.agent import Agent

agent = Agent(model="gpt-4o-mini", instructions="You are a helpful assistant.")
output = agent.run("What is the capital of Japan?")
```

## Add Tools

```python
from definable.tool.decorator import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 72°F in {city}"

agent = Agent(
    model="gpt-4o-mini",
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

agent = Agent(model="gpt-4o-mini", tools=[get_weather])

output = agent.run("Weather in Tokyo?", output_schema=WeatherReport)
print(output.content)  # JSON string matching WeatherReport schema
```

Pass any Pydantic model to `output_schema` and get validated, typed results back.

## Streaming

```python
agent = Agent(model="gpt-4o-mini", instructions="You are a helpful assistant.")

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
from definable.memory import Memory, SQLiteStore

agent = Agent(
    model="gpt-4o-mini",
    memory=Memory(store=SQLiteStore("memory.db")),
    instructions="You are a personal assistant.",
)

await agent.arun("My name is Alice and I prefer dark mode.", user_id="alice")
# Later, even in a new session...
await agent.arun("What's my name?", user_id="alice")  # Recalls "Alice"
```

Memory is LLM-driven: the model decides what to remember via tool calls (add/update/delete). For quick testing, use `memory=True` for an in-memory store. Three backends available: SQLite, PostgreSQL, and in-memory.

## Knowledge Base (RAG)

```python
from definable.knowledge import Knowledge, Document
from definable.embedder import OpenAIEmbedder
from definable.vectordb import InMemoryVectorDB

kb = Knowledge(
    vector_db=InMemoryVectorDB(),
    embedder=OpenAIEmbedder(),
    top_k=3,
)
kb.add(Document(content="Company vacation policy: 20 days PTO per year."))

agent = Agent(
    model="gpt-4o-mini",
    instructions="You are an HR assistant.",
    knowledge=kb,
)

output = agent.run("How many vacation days do I get?")
```

The agent retrieves relevant documents before responding. Supports embedders (OpenAI, Voyage), vector DBs (in-memory, PostgreSQL, Qdrant, ChromaDB, MongoDB, Redis, Pinecone), rerankers (Cohere), and chunkers.

> **Note:** `Agent(knowledge=True)` raises `ValueError` — unlike `memory=True`, knowledge requires explicit configuration with a vector DB.

## Guardrails

```python
from definable.agent.guardrail import Guardrails, max_tokens, pii_filter, tool_blocklist

agent = Agent(
    model="gpt-4o-mini",
    instructions="You are a support agent.",
    tools=[get_weather],
    guardrails=Guardrails(
        input=[max_tokens(500)],
        output=[pii_filter()],
        tool=[tool_blocklist({"dangerous_tool"})],
    ),
)

output = agent.run("What's the weather?")
```

Guardrails check, modify, or block content at input, output, and tool-call checkpoints. Built-ins include token limits, PII redaction, topic blocking, and regex filters. Compose rules with `ALL`, `ANY`, `NOT`, and `when()`.

## Skills

```python
from definable.skill import Calculator, WebSearch, DateTime

agent = Agent(
    model="gpt-4o-mini",
    skills=[Calculator(), WebSearch(), DateTime()],
    instructions="You are a helpful assistant.",
)

output = agent.run("What is 15% of 230?")
```

Skills bundle domain expertise (instructions) with tools. Built-in skills include Calculator, WebSearch, DateTime, HTTPRequests, JSONOperations, TextProcessing, Shell, FileOperations, and MacOS. Create custom skills by subclassing `Skill`.

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

async with MCPToolkit(config=config) as toolkit:
    agent = Agent(model="gpt-4o-mini", toolkits=[toolkit])
    await agent.arun("List files in /tmp")
```

Connect to any MCP server. Use the same tools as Claude Desktop.

## File Readers

```python
from definable.media import File

agent = Agent(
    model="gpt-4o-mini",
    readers=True,
    instructions="Summarize the uploaded document.",
)

output = agent.run("Summarize this.", files=[File(filepath="report.pdf")])
```

Pass `readers=True` to enable automatic parsing. Supports PDF, DOCX, PPTX, XLSX, ODS, RTF, HTML, images, and audio. AI-powered OCR available via Mistral, OpenAI, Anthropic, and Google providers.

## Deploy It

```python
from definable.agent.trigger import Webhook, Cron
from definable.agent.auth import APIKeyAuth

agent = Agent(model="gpt-4o-mini", instructions="You are a support agent.")

agent.on(Webhook(path="/support", method="POST"))
agent.on(Cron(schedule="0 9 * * *"))
agent.auth = APIKeyAuth(keys={"sk-my-secret-key"})
agent.serve(host="0.0.0.0", port=8000, dev=True)
```

`agent.serve()` starts an HTTP server with registered webhooks, cron triggers, and interfaces in a single process. Add `dev=True` for hot-reload during development.

## Connect to Platforms

```python
from definable.agent.interface.telegram import TelegramInterface, TelegramConfig

telegram = TelegramInterface(
    config=TelegramConfig(bot_token="BOT_TOKEN"),
)

agent = Agent(model="gpt-4o-mini", instructions="You are a Telegram bot.")
agent.serve(telegram)
```

One agent, multiple platforms. Discord and Signal interfaces also available.

## Thinking (Reasoning Layer)

```python
from definable.agent.reasoning import Thinking

agent = Agent(
    model="gpt-4o-mini",
    thinking=Thinking(),      # or thinking=True for defaults
    instructions="Think step by step.",
)

output = await agent.arun("What is 127 * 43?")
```

The thinking layer adds chain-of-thought reasoning before the final response.

## Tracing

```python
from definable.agent.tracing import Tracing, JSONLExporter

agent = Agent(
    model="gpt-4o-mini",
    tracing=Tracing(exporters=[JSONLExporter("./traces")]),
    instructions="You are a helpful assistant.",
)

output = agent.run("Hello!")
# Traces saved to ./traces/{session_id}.jsonl
```

Or use `tracing=True` for default console tracing.

## Replay & Compare

```python
from definable.agent.testing import MockModel

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
from definable.agent import Agent
from definable.agent.testing import MockModel

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
| **Models** | OpenAI, DeepSeek, Moonshot, xAI, any OpenAI-compatible provider. String shorthand: `Agent(model="gpt-4o")` resolves automatically |
| **Agents** | Multi-turn conversations, structured output, configurable retries, max iterations |
| **Agentic Loop** | Parallel tool calls via `asyncio.gather`, HITL pause/resume, cooperative cancellation, EventBus |
| **Tools** | `@tool` decorator with automatic parameter extraction from type hints and docstrings |
| **Toolkits** | Composable tool groups, `KnowledgeToolkit` for explicit RAG search |
| **Skills** | Domain expertise + tools in one package; 9 built-in skills (incl. MacOS), custom `Skill` subclass |
| **Knowledge / RAG** | Embedders, vector DBs, rerankers (Cohere), chunkers, automatic retrieval |
| **Memory** | LLM-driven memory with tool-based extraction (add/update/delete) |
| **Memory Stores** | SQLite, PostgreSQL, in-memory |
| **Readers** | PDF, DOCX, PPTX, XLSX, ODS, RTF, HTML, images, audio |
| **Reader Providers** | Mistral OCR, OpenAI, Anthropic, Google (AI-powered document parsing) |
| **Guardrails** | Input/output/tool checkpoints, PII redaction, token limits, topic blocking, regex filters |
| **Guardrails Composition** | `ALL`, `ANY`, `NOT`, `when()` combinators for complex policy rules |
| **Interfaces** | Telegram, Discord, Signal, Desktop, session management, identity resolution |
| **Browser Toolkit** | 50 browser automation tools via SeleniumBase CDP — CSS selectors, screenshots, cookie/storage management |
| **Claude Code Agent** | Zero-dep subprocess wrapper for Claude Code CLI with full Definable ecosystem integration |
| **Runtime** | `agent.serve()`, webhooks, cron triggers, event triggers, `dev=True` hot-reload |
| **Auth** | `APIKeyAuth`, `JWTAuth`, `AllowlistAuth`, `CompositeAuth`, pluggable `AuthProvider` protocol |
| **Streaming** | Real-time response and tool call streaming |
| **Replay** | Inspect past runs, re-execute with overrides, `agent.compare()` for side-by-side diffs |
| **Middleware** | Request/response transforms via `agent.use()`, logging, retry, metrics |
| **Tracing** | JSONL trace export for debugging and analysis |
| **Thinking** | Chain-of-thought reasoning layer with configurable triggers |
| **Compression** | Automatic context window management for long conversations |
| **Testing** | `MockModel`, `AgentTestCase`, `create_test_agent` utilities |
| **MCP** | Model Context Protocol client for external tool servers |
| **Types** | Full Pydantic models, `py.typed` marker, mypy verified |

## Supported Models

```python
from definable.model.openai import OpenAIChat      # GPT-4o, GPT-4o-mini, o1, o3, ...
from definable.model.deepseek import DeepSeekChat   # deepseek-chat, deepseek-reasoner
from definable.model.moonshot import MoonshotChat   # moonshot-v1-8k, moonshot-v1-128k
from definable.model.xai import xAI                 # grok-3, grok-2-latest

# Or use string shorthand — no model import needed:
agent = Agent(model="gpt-4o-mini")
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
pip install definable[browser]          # Browser automation (SeleniumBase CDP)
pip install definable[desktop]          # macOS Desktop Bridge
pip install definable[postgres-memory]  # PostgreSQL memory store
pip install definable[research]         # Deep research (DuckDuckGo + curl-cffi)
pip install definable[mistral-ocr]      # Mistral AI document parsing
pip install definable[mem0-memory]      # Mem0 hosted memory store
```

**Vector DB backends:**

```bash
pip install definable[pgvector]         # PostgreSQL + pgvector
pip install definable[qdrant]           # Qdrant
pip install definable[chroma]           # ChromaDB
pip install definable[mongodb]          # MongoDB
pip install definable[redis]            # Redis
pip install definable[pinecone]         # Pinecone
```

## Documentation

Full documentation: [docs.definable.ai](https://docs.definable.ai)

## Project Structure

```
definable/definable/
├── agent/              # Agent orchestration, config, middleware, loop
│   ├── auth/           # APIKeyAuth, JWTAuth, AllowlistAuth, CompositeAuth
│   ├── compression/    # Context window compression
│   ├── guardrail/      # Input/output/tool policy, PII, token limits, composable rules
│   ├── interface/      # Telegram, Discord, Signal, Desktop integrations
│   ├── reasoning/      # Thinking layer (chain-of-thought)
│   ├── replay/         # Run inspection, re-execution, comparison
│   ├── research/       # Deep research: multi-wave web search, CKU, gap analysis
│   ├── run/            # RunOutput, RunEvent types
│   ├── runtime/        # AgentRuntime, AgentServer, dev mode
│   ├── tracing/        # JSONL trace export
│   └── trigger/        # Webhook, Cron, EventTrigger
├── browser/            # BrowserToolkit — 50 tools via SeleniumBase CDP
├── claude_code/        # ClaudeCodeAgent — subprocess wrapper for Claude Code CLI
├── knowledge/          # RAG: embedders, vector DBs, rerankers, chunkers
├── mcp/                # Model Context Protocol client
├── media.py            # Image, Audio, Video, File types
├── memory/             # LLM-driven memory + 3 store backends
├── model/              # OpenAI, DeepSeek, Moonshot, xAI providers
├── reader/             # File parsers + AI reader providers
├── skill/              # Built-in + custom skills, skill registry
├── tool/               # @tool decorator, Function wrappers
├── toolkit/            # Toolkit base class
├── vectordb/           # Vector database interfaces (7 backends)
└── utils/              # Logging, supervisor, shared utilities
```

## Contributing

Contributions welcome! To get started:

1. Fork the repo and clone it locally
2. Install for development: `pip install -e .`
3. Make your changes — follow existing code patterns (2-space indentation, 150 char lines)
4. Add tests in `definable/tests/` for new features
5. Run `ruff check` and `ruff format` for linting
6. Run `mypy` for type checking
7. Open a pull request

See `definable/examples/` for usage patterns.

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
