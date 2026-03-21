# Definable Examples

Runnable code examples demonstrating all major features of the Definable library.

## Setup

1. Install the library:
   ```bash
   pip install definable
   ```

2. Set your API keys:
   ```bash
   export OPENAI_API_KEY=sk-...
   export DEEPSEEK_API_KEY=sk-...       # Optional: for DeepSeek
   export MOONSHOT_API_KEY=sk-...       # Optional: for Moonshot
   export XAI_API_KEY=...               # Optional: for xAI/Grok
   export VOYAGE_API_KEY=pa-...         # Optional: for VoyageAI embeddings
   export COHERE_API_KEY=...            # Optional: for Cohere reranking
   ```

3. Run any example:
   ```bash
   python examples/agents/01_simple_agent.py
   ```

## Directory Structure

```
examples/
├── agents/                     # Agent framework (start here)
│   ├── 01_simple_agent.py      # Basic agent setup + config
│   ├── 02_agent_with_tools.py  # @tool decorator
│   ├── 03_agent_with_toolkit.py# Custom Toolkit class
│   ├── 04_multi_turn.py        # Conversation sessions
│   ├── 05_streaming_agent.py   # run_stream() usage
│   ├── 06_async_agent.py       # arun() and arun_stream()
│   ├── 07_tracing.py           # JSONLExporter for debugging
│   └── 08_error_handling.py    # Retry logic and exceptions
│
├── models/                     # LLM model invocation
│   ├── 01_basic_invoke.py      # Sync invocation
│   ├── 02_async_invoke.py      # Async + parallel requests
│   ├── 03_streaming.py         # Streaming responses
│   ├── 04_structured_output.py # Pydantic response models
│   ├── 05_multi_provider.py    # OpenAI, DeepSeek, Moonshot, xAI
│   └── 06_vision_and_audio.py  # Image + audio inputs
│
├── tools/                      # Tool definitions
│   ├── 01_tool_parameters.py   # Complex parameter types
│   ├── 02_async_tools.py       # Async tool functions
│   ├── 03_tool_hooks.py        # pre_hook and post_hook
│   ├── 04_tool_caching.py      # cache_results and cache_ttl
│   └── 05_tool_dependencies.py # Injected dependencies via Toolkit
│
├── knowledge/                  # RAG and knowledge bases
│   ├── 01_basic_rag.py         # Setup, add docs, search
│   ├── 02_chunking_strategies.py# TextChunker vs RecursiveChunker
│   ├── 03_custom_embedder.py   # OpenAI, VoyageAI, custom embedders
│   ├── 04_agent_with_knowledge.py# Agent + Knowledge integration
│   └── 05_reranking.py         # CohereReranker usage
│
├── memory/                     # Session-history memory
│   ├── 01_basic_memory.py      # Agent with persistent SQLite memory
│   ├── 02_stores.py            # MemoryStore protocol + backend smoke test
│   └── 03_cortex_memory.py     # CortexMemory (next-gen memory)
│
├── mcp/                        # Model Context Protocol
│   ├── 01_basic_mcp.py         # MCP server connection + agent
│   ├── 02_resources.py         # MCP resources and prompts
│   └── 03_error_handling.py    # Error handling patterns
│
├── skills/                     # Skills and skill registry
│   ├── 01_markdown_skills.py   # SkillRegistry eager/lazy/auto modes
│   ├── 02_coding_agent_skills.py# Programmatic MarkdownSkill creation
│   ├── 03_macos.py             # macOS Desktop skill
│   └── 04_library_skill_discovery.py# Full library inventory + on-demand mode
│
├── interfaces/                 # Messaging interfaces
│   ├── 01_discord_bot.py       # Discord bot interface
│   ├── 02_multi_interface.py   # Telegram + Discord on one agent
│   ├── 03_desktop_control_via_telegram.py
│   └── 04_gateway_telegram.py  # InterfaceGateway + identity linking
│
├── readers/                    # File reading and parsing
│   ├── 01_basic_readers.py     # Read common file formats
│   ├── 02_custom_reader.py     # Custom parser implementation
│   ├── 03_standalone_usage.py  # Readers without an agent
│   ├── 04_provider_override.py # Override format detection
│   ├── 05_mistral_ocr.py       # Mistral OCR provider
│   └── 06_multimodal_agent.py  # Agent with files + images + audio
│
├── runtime/                    # Agent-centric runtime
│   ├── 01_webhook_basic.py     # Webhook trigger
│   ├── 02_cron_basic.py        # Cron trigger
│   └── 03_unified.py           # Interfaces + webhooks + cron + auth
│
├── observability/              # Dashboard and metrics
│   ├── 01_basic_dashboard.py   # observability=True one-liner
│   └── 02_custom_config.py     # Custom ObservabilityConfig
│
├── call/                       # Voice call interfaces
│   ├── 01_managed_voice_agent.py# Twilio ConversationRelay
│   ├── 02_cascading_pipeline.py # Deepgram STT + Cartesia TTS
│   ├── 03_realtime_pipeline.py  # OpenAI Realtime API
│   └── 04_plivo_cascading.py    # Plivo provider
│
├── claude_code/                # Claude Code model
│   ├── 01_basic.py             # ClaudeCodeAgent basics
│   ├── 02_full_features.py     # Memory + knowledge + guardrails
│   ├── 03_coding_agent.py      # Full coding assistant
│   └── 04_chatbot_full_stack.py# Agent with ClaudeCode as model
│
├── slack/                      # Slack integration
│   ├── 01_slack_bot.py         # Socket Mode bot
│   ├── 02_slack_webhook.py     # HTTP Events API
│   └── 03_slack_with_tools.py  # Tools + memory
│
└── whatsapp/                   # WhatsApp integration
    └── 01_basic_agent.py       # Baileys (QR-based) setup
```

## Quick Start

### Basic Agent with Tools

```python
from definable.agent import Agent
from definable.model.openai import OpenAIChat
from definable.tool.decorator import tool


@tool
def add(a: int, b: int) -> int:
  """Add two numbers together."""
  return a + b


agent = Agent(model=OpenAIChat(id="gpt-4o-mini"), tools=[add])
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

kb = Knowledge(vector_db=InMemoryVectorDB(), embedder=VoyageAIEmbedder(), top_k=3)
kb.add(Document(content="Employees get 20 days PTO per year."))

agent = Agent(model=OpenAIChat(id="gpt-4o-mini"), knowledge=kb)
output = agent.run("How many vacation days do I get?")
print(output.content)
```

## Notes

- Each example is self-contained and runnable independently
- Examples use environment variables for API keys (never hardcode secrets)
- Knowledge/memory examples can work without API keys using mock embedders
- Interface/call/slack/whatsapp examples require respective service credentials
