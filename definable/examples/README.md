# Definable Examples

This directory contains runnable code examples demonstrating all major features of the Definable library.

## Setup

1. Install the library:
   ```bash
   pip install definable
   ```

2. Set your API keys as environment variables:
   ```bash
   export OPENAI_API_KEY=sk-...
   export DEEPSEEK_API_KEY=sk-...       # Optional: for DeepSeek examples
   export MOONSHOT_API_KEY=sk-...       # Optional: for Moonshot examples
   export XAI_API_KEY=...               # Optional: for xAI/Grok examples
   export VOYAGE_API_KEY=pa-...         # Optional: for VoyageAI embeddings
   export COHERE_API_KEY=...            # Optional: for Cohere reranking
   ```

3. Run any example:
   ```bash
   python examples/models/01_basic_invoke.py
   ```

## Directory Structure

```
examples/
├── models/                    # LLM model invocation
│   ├── 01_basic_invoke.py     # Sync invocation
│   ├── 02_async_invoke.py     # Async invocation
│   ├── 03_streaming.py        # Streaming responses
│   ├── 04_structured_output.py# Pydantic response models
│   ├── 05_multi_provider.py   # OpenAI, DeepSeek, Moonshot, xAI
│   └── 06_vision_and_audio.py # Multimodal inputs
│
├── agents/                    # Agent framework
│   ├── 01_simple_agent.py     # Basic agent setup
│   ├── 02_agent_with_tools.py # Agent using @tool decorator
│   ├── 03_agent_with_toolkit.py # Custom Toolkit class
│   ├── 04_multi_turn.py       # Conversation sessions
│   ├── 05_streaming_agent.py  # run_stream() usage
│   ├── 06_async_agent.py      # arun() and arun_stream()
│   └── 07_thinking_agent.py   # Thinking layer with reasoning steps
│
├── tools/                     # Tool definitions
│   ├── 01_basic_tool.py       # Simple @tool decorator
│   ├── 02_tool_parameters.py  # Complex parameter types
│   ├── 03_async_tools.py      # Async tool functions
│   ├── 04_tool_hooks.py       # pre_hook and post_hook
│   ├── 05_tool_caching.py     # cache_results and cache_ttl
│   └── 06_tool_dependencies.py# Injected dependencies
│
├── toolkits/                  # Toolkit classes
│   ├── 01_custom_toolkit.py   # Building a Toolkit class
│   ├── 02_toolkit_dependencies.py # Shared dependencies
│   └── 03_knowledge_toolkit.py# KnowledgeToolkit usage
│
├── knowledge/                 # RAG and knowledge bases
│   ├── 01_basic_rag.py        # Simple RAG setup
│   ├── 02_document_management.py # Adding/removing documents
│   ├── 03_chunking_strategies.py # TextChunker vs RecursiveChunker
│   ├── 04_custom_embedder.py  # OpenAI and VoyageAI embedders
│   ├── 05_vector_databases.py # InMemory vs PgVector
│   ├── 06_agent_with_knowledge.py # Knowledge integration
│   └── 07_reranking.py        # CohereReranker usage
│
├── memory/                    # Cognitive memory
│   ├── 01_basic_memory.py     # Agent with persistent memory
│   ├── 02_store_protocol.py   # MemoryStore protocol walkthrough (no deps)
│   └── 03_store_backends.py   # Smoke-test all store backends
│
├── runtime/                   # Agent-centric runtime
│   ├── 01_webhook_basic.py    # Webhook trigger + agent.serve()
│   ├── 02_cron_basic.py       # Cron trigger + agent.serve()
│   └── 03_unified.py         # Interface + webhook + cron + auth + hooks
│
├── interfaces/                # Messaging interfaces
│   ├── 01_discord_bot.py      # Discord bot interface
│   ├── 02_signal_bot.py       # Signal bot interface
│   └── 03_multi_interface.py  # Multiple interfaces on one agent
│
├── readers/                   # File reading and parsing
│   ├── 01_basic_readers.py    # Read common file formats
│   ├── 02_custom_reader.py    # Custom parser implementation
│   ├── 03_standalone_usage.py # Readers without an agent
│   ├── 04_provider_override.py# Override format detection
│   ├── 05_mistral_ocr.py     # Mistral OCR provider
│   └── 06_multimodal_agent.py # Agent with reader integration
│
├── auth/                      # Authentication
│   └── 01_unified_auth.py     # APIKeyAuth, AllowlistAuth, CompositeAuth
│
├── guardrails/                # Content policy and safety
│   ├── 01_basic_guardrails.py # Built-in guardrails with Agent
│   └── 02_custom_guardrails.py# Custom + composable guardrails
│
├── skills/                    # Skills and skill registry
│   └── 01_markdown_skills.py  # SkillRegistry eager/lazy/auto modes
│
├── replay/                    # Run inspection and comparison
│   └── 01_basic_replay.py     # Replay inspection + compare_runs
│
├── research/                  # Deep research
│   ├── 01_basic_research.py   # Standalone DeepResearch usage
│   └── 02_agent_with_research.py # Agent with deep_research enabled
│
├── mcp/                       # Model Context Protocol
│   ├── 01_basic_mcp.py        # Basic MCP server connection
│   ├── 02_multiple_servers.py # Multiple MCP servers
│   ├── 03_resources.py        # MCP resource access
│   ├── 04_config_file.py      # Config file-based setup
│   ├── 05_mock_server_basics.py # Mock server for testing
│   ├── 06_prompts_provider.py # MCP prompts provider
│   ├── 07_error_handling.py   # Error handling patterns
│   └── 08_mock_server_agent.py# Mock server with agent integration
│
└── advanced/                  # Advanced features
    ├── 01_middleware.py       # Custom middleware
    ├── 02_tracing.py          # JSONLExporter for debugging
    ├── 03_error_handling.py   # Retry logic and exceptions
    └── 04_cost_tracking.py    # Metrics and pricing
```

## Quick Start Examples

### Basic Model Invocation

```python
from definable.model.openai import OpenAIChat
from definable.model.message import Message

model = OpenAIChat(id="gpt-4o-mini")
response = model.invoke(
    messages=[Message(role="user", content="Hello!")],
    assistant_message=Message(role="assistant", content=""),
)
print(response.content)
```

### Basic Agent with Tools

```python
from definable.agent import Agent
from definable.model.openai import OpenAIChat
from definable.tool.decorator import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

model = OpenAIChat(id="gpt-4o-mini")
agent = Agent(model=model, tools=[add])
output = agent.run("What is 5 + 3?")
print(output.content)
```

### Agent with Knowledge Base (RAG)

```python
from definable.agent import Agent
from definable.embedder import VoyageAIEmbedder
from definable.knowledge import Document, Knowledge
from definable.vectordb import InMemoryVectorDB
from definable.model.openai import OpenAIChat

# Setup knowledge base
kb = Knowledge(
    vector_db=InMemoryVectorDB(),
    embedder=VoyageAIEmbedder(),
    top_k=3,
)
kb.add(Document(content="Company policy: Employees get 20 days PTO per year."))

# Create agent with automatic RAG
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions="You are a helpful HR assistant.",
    knowledge=kb,
)

output = agent.run("How many vacation days do I get?")
print(output.content)
```

## Notes

- Each example is self-contained and can be run independently
- Examples use environment variables for API keys (never hardcode secrets)
- Some examples require specific API keys (noted in each file)
- Knowledge examples can work without API keys using pre-computed embeddings
