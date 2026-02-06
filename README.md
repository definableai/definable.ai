# Definable

Build LLM agents that actually work in production.

A Python framework for building agent applications with support for multiple LLM providers, RAG, tools, and the Model Context Protocol (MCP).

## Quick Start

```python
from definable.agents import Agent
from definable.models.openai import OpenAIChat

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions="You are a helpful assistant."
)

output = agent.run("What is the capital of Japan?")
print(output.content)
```

That's it. Six lines to a working agent.

## Add Tools

Use the `@tool` decorator to give your agent capabilities:

```python
from definable.tools.decorator import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 72°F in {city}"

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[get_weather],
    instructions="Help users check the weather."
)

output = agent.run("What's the weather in Tokyo?")
```

The agent automatically calls tools when needed. No manual function routing.

## Streaming

Stream responses as they're generated:

```python
for event in agent.run_stream("Tell me a story about a robot"):
    if hasattr(event, "content") and event.content:
        print(event.content, end="", flush=True)
```

Works with tool calls too—see which tools are being used in real-time.

## RAG / Knowledge Base

Add a knowledge base and the agent automatically retrieves relevant context:

```python
from definable.agents import AgentConfig, KnowledgeConfig
from definable.knowledge import Knowledge, InMemoryVectorDB, Document

# Create and populate knowledge base
kb = Knowledge(vector_db=InMemoryVectorDB(dimensions=1536), embedder=embedder)
kb.add(Document(content="Company vacation policy: 20 days PTO per year"))

# Agent automatically retrieves relevant documents
agent = Agent(
    model=model,
    instructions="You are an HR assistant.",
    config=AgentConfig(
        knowledge=KnowledgeConfig(knowledge=kb, top_k=3)
    )
)

output = agent.run("How many vacation days do I get?")
```

The agent pulls relevant documents based on the query. No manual retrieval logic.

## Model Context Protocol (MCP)

Connect to any MCP server—Claude Desktop tools, filesystem access, databases, custom servers:

```python
from definable.mcp import MCPConfig, MCPServerConfig, MCPToolkit

config = MCPConfig(
    servers=[
        MCPServerConfig(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        )
    ]
)

async with MCPToolkit(config) as toolkit:
    agent = Agent(model=model, toolkits=[toolkit])
    await agent.arun("List files in /tmp")
```

Your agent can now use the same tools as Claude Desktop.

## What You Get

| Feature | Implementation |
|---------|---------------|
| **Multiple LLM providers** | OpenAI, DeepSeek, Moonshot, xAI |
| **RAG / Knowledge** | Embedders (OpenAI, Voyage), vector DBs (in-memory, PostgreSQL), rerankers (Cohere), chunkers |
| **Tools** | `@tool` decorator with automatic parameter extraction |
| **External tools** | MCP protocol support for any MCP server |
| **Streaming** | Real-time response and tool execution streaming |
| **Production features** | Middleware, JSONL tracing, async/await, error handling |
| **Type safety** | Full Pydantic models, mypy verified |
| **Testing** | Built-in test utilities and fixtures |

## Installation

```bash
pip install definable-ai
```

Then set your API keys:

```bash
export OPENAI_API_KEY=sk-...
# Optional: for other providers
export DEEPSEEK_API_KEY=...
export VOYAGE_API_KEY=...
export COHERE_API_KEY=...
```

## Examples

The repo includes 40+ working examples showing real usage patterns:

```bash
git clone https://github.com/yourusername/definable.ai
cd definable.ai/examples

# Basic agents
python agents/01_simple_agent.py
python agents/02_agent_with_tools.py
python agents/05_streaming_agent.py

# RAG / Knowledge
python knowledge/01_basic_rag.py
python knowledge/06_agent_with_knowledge.py

# MCP integration
python mcp/01_basic_mcp.py
python mcp/02_multiple_servers.py

# Advanced patterns
python advanced/middleware_example.py
python advanced/tracing_example.py
```

Each example is self-contained and demonstrates a specific pattern.

## What You Can Build

Real applications people are shipping:

- **Customer support bots** with company knowledge base (RAG + tools)
- **Research assistants** that access external data sources (MCP + web tools)
- **Data analysis agents** with filesystem and database access (MCP servers)
- **Workflow automation** with multi-step tool chains (toolkits + middleware)
- **Internal Q&A systems** over company docs (RAG + reranking)

## Why This Library

**Production-tested**: Built from patterns used in real agent deployments, not theory.

**Not just an OpenAI wrapper**: Supports multiple LLM providers with a unified interface. Switch providers without rewriting code.

**Complete RAG stack**: Don't cobble together multiple libraries. Embedders, vector DBs, rerankers, and chunkers all work together.

**MCP integration**: The only Python agent framework with first-class Model Context Protocol support. Use Claude Desktop tools, or build your own MCP servers.

**Actually tested**: Comprehensive test suite with unit, integration, and end-to-end tests. Type-checked with mypy.

**Async-first**: Built on async/await for streaming, concurrent tool calls, and high throughput.

## Multi-Model Support

Switch LLM providers with one line:

```python
# OpenAI
from definable.models.openai import OpenAIChat
model = OpenAIChat(id="gpt-4o-mini")

# DeepSeek
from definable.models.deepseek import DeepSeekChat
model = DeepSeekChat(id="deepseek-chat")

# xAI
from definable.models.xai import xAIChat
model = xAIChat(id="grok-2-latest")

# Same agent code works with all of them
agent = Agent(model=model, instructions="...")
```

## Advanced Features

**Middleware**: Transform requests/responses, add logging, implement rate limiting:

```python
from definable.agents.middleware import Middleware

class LoggingMiddleware(Middleware):
    async def process_request(self, request):
        print(f"Request: {request.messages[-1].content}")
        return request

agent = Agent(model=model, middleware=[LoggingMiddleware()])
```

**Tracing**: Export execution traces to JSONL for debugging and analysis:

```python
from definable.tracing import JSONLTracer

tracer = JSONLTracer(filepath="traces.jsonl")
agent = Agent(model=model, tracer=tracer)
```

**Error handling**: Automatic retries with exponential backoff:

```python
from definable.agents import AgentConfig

agent = Agent(
    model=model,
    config=AgentConfig(max_retries=3, max_iterations=10)
)
```

## Documentation

- **Examples**: 40+ working examples in `examples/`
- **API Reference**: Full Pydantic models with docstrings
- **Tests**: See `tests_e2e/` for integration patterns

## Requirements

- Python 3.12+
- OpenAI API key (or other provider keys)
- Optional: Voyage AI for embeddings, Cohere for reranking
- Optional: MCP servers via npm (for MCP features)

## Project Structure

```
definable/
├── agents/          # Agent orchestration, config, middleware
├── models/          # LLM provider implementations
├── tools/           # Tool system and toolkits
├── knowledge/       # RAG: embedders, vector DBs, rerankers
├── mcp/             # Model Context Protocol client
├── tracing/         # Execution tracing and logging
└── examples/        # 40+ working examples
```

## Contributing

Contributions welcome. Please:

- Add tests for new features
- Run `ruff` for linting
- Run `mypy` for type checking
- Follow existing code patterns

## License

MIT License. See LICENSE file for details.

## Built With Definable

Using this in production? Let us know—we'd love to hear what you're building.
