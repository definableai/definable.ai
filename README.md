# Definable

Build LLM agents that work in production.

[![PyPI](https://img.shields.io/pypi/v/definable-ai)](https://pypi.org/project/definable-ai/)
[![Python](https://img.shields.io/pypi/pyversions/definable-ai)](https://pypi.org/project/definable-ai/)
[![License](https://img.shields.io/github/license/definable-ai/definable)](https://github.com/definable-ai/definable/blob/main/LICENSE)

A Python framework for building agent applications with tools, RAG, persistent memory, file readers, messaging platform integrations, and the Model Context Protocol. Switch providers without rewriting agent code.

---

## Install

```bash
pip install definable-ai
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
# Later...
agent.run("What's my name?", user_id="alice")  # Recalls "Alice"
```

Memory is automatic: the agent stores interactions and recalls relevant context on each turn. Eight store backends available (SQLite, PostgreSQL, Redis, Qdrant, Chroma, Pinecone, MongoDB, in-memory).

## Deploy It

```python
from definable.triggers import Webhook

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions="You are a support agent.",
)

agent.on(Webhook(path="/support", method="POST"))
agent.serve(host="0.0.0.0", port=8000)
```

`agent.serve()` starts an HTTP server with registered webhooks, cron triggers, and interfaces in a single process.

## Knowledge Base

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

## Connect to Platforms

```python
from definable.interfaces import TelegramInterface

telegram = TelegramInterface(token="BOT_TOKEN")

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

---

## Features

| Category | Details |
|---|---|
| **Models** | OpenAI, DeepSeek, Moonshot, xAI, any OpenAI-compatible provider |
| **Agents** | Multi-turn conversations, configurable retries, max iterations |
| **Tools** | `@tool` decorator with automatic parameter extraction from type hints and docstrings |
| **Toolkits** | Composable tool groups, `KnowledgeToolkit` for explicit RAG search |
| **Knowledge / RAG** | Embedders, vector DBs, rerankers (Cohere), chunkers, automatic retrieval |
| **Memory** | `CognitiveMemory` with multi-tier recall, distillation, topic prediction |
| **Memory Stores** | SQLite, PostgreSQL, Redis, Qdrant, Chroma, Pinecone, MongoDB, in-memory |
| **Readers** | PDF, DOCX, PPTX, XLSX, ODS, RTF, HTML, images, audio |
| **Reader Providers** | Mistral OCR, OpenAI, Anthropic, Google (AI-powered document parsing) |
| **Interfaces** | Telegram, Discord, Signal, session management, identity resolution |
| **Runtime** | `agent.serve()`, webhooks, cron triggers, event triggers |
| **Auth** | `APIKeyAuth`, `JWTAuth`, pluggable `AuthProvider` protocol |
| **Streaming** | Real-time response and tool call streaming |
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
pip install definable-ai[readers]          # PDF, DOCX, PPTX, XLSX, ODS, RTF parsers
pip install definable-ai[serve]            # FastAPI + Uvicorn for agent.serve()
pip install definable-ai[cron]             # Cron trigger support
pip install definable-ai[jwt]              # JWT authentication
pip install definable-ai[runtime]          # serve + cron combined
pip install definable-ai[discord]          # Discord interface
pip install definable-ai[postgres-memory]  # PostgreSQL memory store
pip install definable-ai[redis-memory]     # Redis memory store
pip install definable-ai[qdrant-memory]    # Qdrant memory store
pip install definable-ai[chroma-memory]    # Chroma memory store
pip install definable-ai[mongodb-memory]   # MongoDB memory store
pip install definable-ai[pinecone-memory]  # Pinecone memory store
pip install definable-ai[mistral-ocr]      # Mistral AI document parsing
```

## Documentation

Full documentation: [definable.ai/docs](https://definable.ai/docs)

## Project Structure

```
definable/definable/
├── agents/        # Agent orchestration, config, middleware, tracing, testing
├── auth/          # APIKeyAuth, JWTAuth, AuthProvider protocol
├── compression/   # Context window compression
├── interfaces/    # Telegram, Discord, Signal integrations
├── knowledge/     # RAG: embedders, vector DBs, rerankers, chunkers
├── mcp/           # Model Context Protocol client
├── media.py       # Image, Audio, Video, File types
├── memory/        # CognitiveMemory + 8 store backends
├── models/        # OpenAI, DeepSeek, Moonshot, xAI providers
├── readers/       # File parsers + AI reader providers
├── reasoning/     # Reasoning capabilities
├── run/           # RunOutput, RunEvent types
├── runtime/       # AgentRuntime, AgentServer
├── tools/         # @tool decorator, tool wrappers
├── triggers/      # Webhook, Cron, EventTrigger
├── utils/         # Logging, supervisor, shared utilities
└── vectordbs/     # Vector database interfaces
```

## Contributing

Contributions welcome.

- Add tests for new features
- Run `ruff check` and `ruff format` for linting
- Run `mypy` for type checking
- Follow existing code patterns

## License

MIT
