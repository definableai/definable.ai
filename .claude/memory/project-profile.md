# Project Profile — Definable v0.2.8

> Last updated: 2026-02-17 (eval run #3)

## Package Info
- **Name**: definable
- **Version**: 0.2.8
- **Python**: >=3.12 (3.12.10 in .venv)
- **Source**: `definable/definable/` (215 .py files)
- **Tests**: `definable/tests_e2e/` (975 tests collected)

## Key Correct Import Paths

These are the ACTUAL working imports (not what docs say):

```python
# Agents
from definable.agents import Agent, AgentConfig, Middleware, Toolkit, KnowledgeToolkit
from definable.agents import ThinkingConfig, KnowledgeConfig, TracingConfig, DeepResearchConfig
from definable.agents import MockModel, create_test_agent, AgentTestCase

# Models
from definable.models import OpenAIChat, DeepSeekChat, MoonshotChat, xAI, OpenAILike
from definable.models.message import Message
from definable.models.response import ModelResponse

# Tools
from definable.tools.decorator import tool  # Returns Function object
# Function.entrypoint is sync; Function.parameters returns OpenAI-format dict

# Skills
from definable.skills import Skill  # or from definable.skills.base import Skill
from definable.skills.registry import SkillRegistry  # uses .get_skill(), .list_skills()

# Knowledge
from definable.knowledge import Knowledge, Document
from definable.knowledge.vector_dbs import InMemoryVectorDB  # .add(docs), .search(query_embedding, top_k)
from definable.knowledge.embedders.voyageai import VoyageAIEmbedder  # NOT re-exported!
from definable.knowledge.embedders.openai import OpenAIEmbedder  # NOT re-exported!
from definable.knowledge.rerankers.cohere import CohereReranker  # NOT re-exported!
# Document(content=str, meta_data=dict)  — note: meta_data, NOT metadata

# Memory
from definable.memory import CognitiveMemory
from definable.memory.store import SQLiteMemoryStore, InMemoryStore
# Store API: store_episode, get_episodes, store_atom, get_atoms (NOT add/recall)

# Guardrails
from definable.guardrails import Guardrails, max_tokens, pii_filter, block_topics
from definable.guardrails import regex_filter, tool_allowlist, tool_blocklist
from definable.guardrails import ALL, ANY, NOT, InputGuardrail, OutputGuardrail

# Auth
from definable.auth import APIKeyAuth, JWTAuth, AllowlistAuth, CompositeAuth
# APIKeyAuth(keys=set), AllowlistAuth(user_ids=set), JWTAuth(secret=str)

# MCP
from definable.mcp.toolkit import MCPToolkit  # MCPToolkit(config=MCPConfig)

# Readers
from definable.readers import FileReader  # .read(file: File), NOT .read(path: str)

# Research
from definable.research.engine import DeepResearch
from definable.research.config import DeepResearchConfig

# Reasoning
from definable.reasoning.step import ReasoningStep, ThinkingOutput, thinking_output_to_reasoning_steps

# Replay/Tracing
from definable.replay import Replay, ReplayComparison
from definable.agents.tracing import JSONLExporter  # NOT definable.tracing

# Runtime
from definable.runtime import AgentServer

# Other
from definable.compression import CompressionManager
from definable.triggers import BaseTrigger, EventTrigger
from definable.filters import FilterExpr
from definable.media import Image, Audio, Video, File
```

## Agent API

```python
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),  # REQUIRED
    tools=[...],             # List[Function]
    toolkits=[...],          # List[Toolkit|MCPToolkit]
    skills=[...],            # List[Skill]
    instructions="...",      # str
    memory=CognitiveMemory(store=InMemoryStore()),
    guardrails=Guardrails(...),
    thinking=True,           # or ThinkingConfig(...)
    deep_research=True,      # or DeepResearchConfig(...)
    config=AgentConfig(...), # Optional config
)

# Run
result = await agent.arun(
    "prompt",
    messages=[...],       # Optional history (for multi-turn, pass r.messages)
    output_schema=Person, # NOT response_model
    user_id="...",
)

# Stream
async for event in agent.arun_stream("prompt"):
    ...  # RunStartedEvent, RunContentEvent, RunCompletedEvent, ToolCall*

# Middleware — __call__ protocol
class MyMiddleware:
    async def __call__(self, context, next_handler):
        result = await next_handler(context)
        return result
agent.use(MyMiddleware())
```

## RunOutput Attributes
- `content`, `messages`, `metrics`, `model`, `run_id`, `session_id`, `user_id`
- `events`, `status`, `reasoning_steps`, `reasoning_content`, `reasoning_messages`
- `images`, `audio`, `videos`, `files`, `citations`, `references`
- `tools`, `tools_awaiting_external_execution`, `tools_requiring_confirmation`
- NOTE: NO `model_response` attribute — use `metrics` for token counts

## Multi-Turn Conversations
- `session_id` alone does NOT maintain history — requires CognitiveMemory or explicit message passing
- For message passing: `r2 = await agent.arun("follow up", messages=r1.messages)`
- For automatic memory: `Agent(memory=CognitiveMemory(store=InMemoryStore()), ...)`

## Static Analysis
- mypy: 0 errors (215 files)
- ruff check: 0 warnings
- ruff format: 0 issues
