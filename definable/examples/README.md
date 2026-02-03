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
│   └── 06_async_agent.py      # arun() and arun_stream()
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
│   ├── 06_agent_with_knowledge.py # KnowledgeConfig integration
│   └── 07_reranking.py        # CohereReranker usage
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
from definable.models.openai import OpenAIChat
from definable.models.message import Message

model = OpenAIChat(id="gpt-4o-mini")
response = model.invoke(
    messages=[Message(role="user", content="Hello!")],
    assistant_message=Message(role="assistant", content=""),
)
print(response.content)
```

### Basic Agent with Tools

```python
from definable.agents import Agent
from definable.models.openai import OpenAIChat
from definable.tools.decorator import tool

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
from definable.agents import Agent, AgentConfig, KnowledgeConfig
from definable.knowledge import Knowledge, Document, InMemoryVectorDB, VoyageAIEmbedder
from definable.models.openai import OpenAIChat

# Setup knowledge base
kb = Knowledge(
    vector_db=InMemoryVectorDB(),
    embedder=VoyageAIEmbedder(),
)
kb.add(Document(content="Company policy: Employees get 20 days PTO per year."))

# Create agent with automatic RAG
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions="You are a helpful HR assistant.",
    config=AgentConfig(
        knowledge=KnowledgeConfig(knowledge=kb, top_k=3),
    ),
)

output = agent.run("How many vacation days do I get?")
print(output.content)
```

## Notes

- Each example is self-contained and can be run independently
- Examples use environment variables for API keys (never hardcode secrets)
- Some examples require specific API keys (noted in each file)
- Knowledge examples can work without API keys using pre-computed embeddings
