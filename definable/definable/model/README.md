# models

LLM provider implementations with a unified interface for chat completions, streaming, tool calling, and structured outputs.

## Quick Start

```python
from definable.model.openai import OpenAIChat

model = OpenAIChat(id="gpt-4o")
response = model.invoke(messages=[...])
print(response.content)
```

## Module Structure

```
models/
├── __init__.py      # Empty — import providers from sub-packages directly
├── base.py          # Model ABC — retry, caching, streaming
├── message.py       # Message, Citations, MessageReferences
├── response.py      # ModelResponse, ToolExecution, ModelResponseEvent
├── metrics.py       # Metrics dataclass (tokens, timing, cost)
├── pricing.py       # ModelPricing, PricingRegistry
├── utils.py         # Shared utilities
├── openai/
│   ├── chat.py      # OpenAIChat
│   └── like.py      # OpenAILike (OpenAI-compatible wrapper)
├── deepseek/
│   └── chat.py      # DeepSeekChat
├── moonshot/
│   └── chat.py      # MoonshotChat
└── xai/
    └── xai.py       # xAI (Grok)
```

**Note:** `__init__.py` is empty. Import providers from their sub-packages directly.

## API Reference

### Model (ABC)

```python
from definable.model.base import Model
```

Abstract base class for all providers. Subclasses must implement:

| Method | Description |
|--------|-------------|
| `invoke(...)` | Synchronous single invocation |
| `ainvoke(...)` | Async single invocation |
| `invoke_stream(...)` | Sync streaming (yields `ModelResponse`) |
| `ainvoke_stream(...)` | Async streaming |

Built-in features: exponential backoff retry, response caching (MD5-keyed, TTL support), structured output enforcement.

### Providers

| Class | Import Path | Default Model | Env Var | Base Class |
|-------|-------------|---------------|---------|------------|
| `OpenAIChat` | `definable.model.openai` | `gpt-4o` | `OPENAI_API_KEY` | `Model` |
| `OpenAILike` | `definable.model.openai` | — | — | `OpenAIChat` |
| `DeepSeekChat` | `definable.model.deepseek` | `deepseek-chat` | `DEEPSEEK_API_KEY` | `OpenAILike` |
| `MoonshotChat` | `definable.model.moonshot` | `kimi-k2-turbo-preview` | `MOONSHOT_API_KEY` | `OpenAILike` |
| `xAI` | `definable.model.xai` | `grok-3` | `XAI_API_KEY` | `OpenAILike` |

**OpenAILike** is a generic wrapper for any provider using an OpenAI-compatible API — set `base_url` and `api_key` for third-party providers.

### Message

```python
from definable.model.message import Message
```

Pydantic model representing a message in the conversation.

| Field | Type | Description |
|-------|------|-------------|
| `role` | `str` | `"system"`, `"user"`, `"assistant"`, `"tool"` |
| `content` | `Optional[Union[List, str]]` | Message content (text or multimodal blocks) |
| `tool_calls` | `Optional[List[Dict]]` | Tool calls from the model |
| `tool_call_id` | `Optional[str]` | ID for tool result messages |
| `images` / `audio` / `videos` / `files` | `Optional[Sequence]` | Media attachments |
| `reasoning_content` | `Optional[str]` | Model reasoning trace |
| `citations` | `Optional[Citations]` | URL and document citations |

### ModelResponse

```python
from definable.model.response import ModelResponse
```

| Field | Type | Description |
|-------|------|-------------|
| `content` | `Optional[Any]` | Response text |
| `tool_calls` | `List[Dict]` | Tool calls to execute |
| `tool_executions` | `Optional[List[ToolExecution]]` | Completed tool execution results |
| `reasoning_content` | `Optional[str]` | Chain-of-thought reasoning |
| `response_usage` | `Optional[Metrics]` | Token usage and cost |
| `images` / `videos` / `audios` / `files` | `Optional[List]` | Generated media |

### ToolExecution

```python
from definable.model.response import ToolExecution
```

Tracks tool/function call execution including HITL (human-in-the-loop) fields:

| Field | Type | Description |
|-------|------|-------------|
| `tool_name` | `Optional[str]` | Name of the tool called |
| `tool_args` | `Optional[Dict]` | Arguments passed |
| `result` | `Optional[str]` | Execution result |
| `requires_confirmation` | `Optional[bool]` | Needs user confirmation |
| `requires_user_input` | `Optional[bool]` | Needs user input |
| `external_execution_required` | `Optional[bool]` | Needs external execution |

### Metrics

```python
from definable.model.metrics import Metrics
```

| Field | Type | Description |
|-------|------|-------------|
| `input_tokens` | `int` | Input token count |
| `output_tokens` | `int` | Output token count |
| `total_tokens` | `int` | Total token count |
| `cost` | `Optional[float]` | Computed cost (USD) |
| `reasoning_tokens` | `int` | Reasoning/thinking tokens |
| `cache_read_tokens` | `int` | Cache hit tokens |
| `time_to_first_token` | `Optional[float]` | TTFT in seconds |
| `duration` | `Optional[float]` | Total duration in seconds |

Supports `+` operator for aggregation and `sum()` over lists.

### ModelPricing

```python
from definable.model.pricing import ModelPricing, get_pricing, calculate_cost
```

- `ModelPricing` — Pricing rates per million tokens (input, output, cached, audio, reasoning).
- `PricingRegistry` — Singleton that loads pricing from `model_pricing.json`.
- `get_pricing(provider, model_id)` — Look up pricing by provider and model.
- `calculate_cost(provider, model_id, metrics)` — Calculate cost from metrics.

## See Also

- `agents/` — Agent wraps a model with tools, middleware, and orchestration
- `tools/` — `Function` class for tool definitions
- `run/` — `RunOutput` and event types returned by agent execution
