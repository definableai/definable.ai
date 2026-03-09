# model

LLM provider implementations with a unified interface for chat completions, streaming, tool calling, and structured outputs. Supports 10 providers, lazy-loaded imports, and a resilience layer with key rotation and provider failover.

## Architecture

```
model/
├── __init__.py           # Public API — 20+ exports, all lazy-loaded
├── base.py               # Model ABC — retry, caching, streaming, tool dispatch
├── message.py            # Message, Citations, MessageReferences
├── response.py           # ModelResponse, ToolExecution, ModelResponseEvent
├── metrics.py            # Metrics (tokens, timing, cost) with + operator
├── pricing.py            # ModelPricing, PricingRegistry, get_pricing, calculate_cost
├── utils.py              # resolve_model_string, get_supported_providers, get_model
├── openai/               # OpenAIChat, OpenAILike (OpenAI-compatible wrapper)
├── deepseek/             # DeepSeekChat (OpenAILike)
├── moonshot/             # MoonshotChat (OpenAILike)
├── xai/                  # xAI / Grok (OpenAILike)
├── anthropic/            # Claude (native Anthropic SDK)
├── mistral/              # MistralChat (native Mistral SDK)
├── google/               # Gemini (native Google SDK, Vertex AI support)
├── perplexity/           # Perplexity (OpenAILike, web search via citations)
├── ollama/               # Ollama (native Ollama SDK, local models)
├── openrouter/           # OpenRouter (OpenAILike, dynamic model routing)
└── resilience/           # ResilientModel, KeyPool, FailoverChain, FailoverEntry
    ├── key_pool.py       # Thread-safe key rotation with health tracking
    ├── failover.py       # FailoverChain + FailoverEntry
    ├── resilient.py      # ResilientModel wrapper
    └── events.py         # KeyRotatedEvent, ProviderFailoverEvent
```

## Quick Start

```python
from definable.model import OpenAIChat, Message

model = OpenAIChat(id="gpt-4o")
response = model.invoke(
  messages=[Message(role="user", content="Hello")],
  assistant_message=Message(role="assistant", content=""),
)
print(response.content)
```

## Providers

| Class | Default Model | Env Var | Base | Install |
|-------|--------------|---------|------|---------|
| `OpenAIChat` | `gpt-4o` | `OPENAI_API_KEY` | `Model` | core |
| `OpenAILike` | — | — | `OpenAIChat` | core |
| `DeepSeekChat` | `deepseek-chat` | `DEEPSEEK_API_KEY` | `OpenAILike` | core |
| `MoonshotChat` | `kimi-k2-turbo-preview` | `MOONSHOT_API_KEY` | `OpenAILike` | core |
| `xAI` | `grok-3` | `XAI_API_KEY` | `OpenAILike` | core |
| `Perplexity` | `sonar` | `PERPLEXITY_API_KEY` | `OpenAILike` | core |
| `OpenRouter` | `gpt-4o` | `OPENROUTER_API_KEY` | `OpenAILike` | core |
| `Claude` | `claude-sonnet-4-5-20250929` | `ANTHROPIC_API_KEY` | `Model` | `pip install "definable[anthropic]"` |
| `MistralChat` | `mistral-large-latest` | `MISTRAL_API_KEY` | `Model` | `pip install "definable[mistral]"` |
| `Gemini` | `gemini-2.0-flash-001` | `GOOGLE_API_KEY` | `Model` | `pip install google-genai` |
| `Ollama` | `llama3.1` | `OLLAMA_API_KEY` (optional) | `Model` | `pip install ollama` |

**OpenAILike** is a generic wrapper for any provider with an OpenAI-compatible API. Set `base_url` and `api_key` to point it at third-party services.

**Gemini** and **Ollama** are lazy-loaded — the SDK is only imported when the class is first accessed.

## Imports

All providers and types are importable from `definable.model`:

```python
from definable.model import OpenAIChat, OpenAILike, DeepSeekChat, MoonshotChat, xAI
from definable.model import Claude, MistralChat, Perplexity
from definable.model import ResilientModel, KeyPool, FailoverChain, FailoverEntry
from definable.model import resolve_model_string
from definable.model import Message, ModelResponse, Metrics, ToolExecution
from definable.model import Citations, MessageReferences, Model

# Lazy-loaded (require optional dependencies):
from definable.model import Gemini  # requires: pip install google-genai
from definable.model import Ollama  # requires: pip install ollama
from definable.model import OpenRouter
```

## String Model Shorthand

`resolve_model_string` maps `"provider/model-id"` strings to instantiated provider objects. This is what powers `Agent(model="openai/gpt-4o-mini")`.

```python
from definable.model import resolve_model_string

model = resolve_model_string("openai/gpt-4o-mini")  # → OpenAIChat(id="gpt-4o-mini")
model = resolve_model_string("anthropic/claude-sonnet-4-5-20250929")  # → Claude(...)
model = resolve_model_string("deepseek/deepseek-chat")  # → DeepSeekChat(...)
model = resolve_model_string("google/gemini-2.0-flash-001")  # → Gemini(...)
model = resolve_model_string("gpt-4o")  # bare name → OpenAI default
```

Supported providers: `anthropic`, `deepseek`, `google`, `mistral`, `moonshot`, `ollama`, `openai`, `openrouter`, `perplexity`, `xai`

Raises `ValueError` for unknown providers. Raises `ImportError` with install hint for providers that need an optional dependency.

## API Reference

### Model (ABC)

```python
from definable.model import Model
```

Abstract base class for all providers. Subclasses must implement four methods:

| Method | Description |
|--------|-------------|
| `invoke(messages, assistant_message, ...)` | Synchronous single invocation |
| `ainvoke(messages, assistant_message, ...)` | Async single invocation |
| `invoke_stream(messages, assistant_message, ...)` | Sync streaming — yields `ModelResponse` |
| `ainvoke_stream(messages, assistant_message, ...)` | Async streaming — yields `ModelResponse` |

**Important:** `assistant_message` is a required second positional argument on all four methods. Pass an empty assistant `Message` as a container for the response:

```python
from definable.model import Message

response = model.invoke(
  messages=[Message(role="user", content="Hello")],
  assistant_message=Message(role="assistant", content=""),
)
```

Built-in features (configured on the Model subclass):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | provider default | Model identifier |
| `name` | `Optional[str]` | `None` | Human-readable name |
| `provider` | `Optional[str]` | `None` | Provider name |
| `supports_native_structured_outputs` | `bool` | `False` | Provider supports native JSON schema |
| `supports_json_schema_outputs` | `bool` | `False` | Provider supports JSON schema |
| `cache_response` | `bool` | `False` | Enable MD5-keyed response caching |
| `cache_ttl` | `Optional[int]` | `None` | Cache TTL in seconds |
| `retries` | `int` | `0` | Number of retry attempts |
| `delay_between_retries` | `int` | `1` | Seconds between retries |
| `exponential_backoff` | `bool` | `False` | Use exponential backoff for retries |
| `retry_with_guidance` | `bool` | `False` | Send failure context on retry |
| `retry_with_guidance_limit` | `int` | `3` | Max guided retry attempts |
| `system_prompt` | `Optional[str]` | `None` | Injected system prompt |
| `instructions` | `Optional[str]` | `None` | Additional instructions |

### Message

```python
from definable.model import Message
```

Pydantic model representing a single turn in a conversation.

| Field | Type | Description |
|-------|------|-------------|
| `role` | `str` | `"system"`, `"user"`, `"assistant"`, or `"tool"` |
| `content` | `Optional[Union[List, str]]` | Text or multimodal content blocks |
| `tool_calls` | `Optional[List[Dict]]` | Tool calls the model requested |
| `tool_call_id` | `Optional[str]` | ID linking a tool result to its call |
| `images` | `Optional[Sequence[Image]]` | Image attachments (use `Image` objects, not plain strings) |
| `audio` | `Optional[Audio]` | Audio attachment |
| `videos` | `Optional[Sequence[Video]]` | Video attachments |
| `files` | `Optional[Sequence[File]]` | File attachments |
| `reasoning_content` | `Optional[str]` | Model's reasoning trace |
| `citations` | `Optional[Citations]` | URL and document citations |

**Note:** The `images` field expects `Image` objects from `definable.media`, not plain strings.

### ModelResponse

```python
from definable.model import ModelResponse
```

Returned by all invoke methods.

| Field | Type | Description |
|-------|------|-------------|
| `content` | `Optional[Any]` | Response text (or parsed Pydantic model for structured output) |
| `tool_calls` | `List[Dict]` | Tool calls to execute |
| `tool_executions` | `Optional[List[ToolExecution]]` | Completed tool results |
| `reasoning_content` | `Optional[str]` | Chain-of-thought reasoning |
| `response_usage` | `Optional[Metrics]` | Token usage and cost |
| `images` / `videos` / `audios` / `files` | `Optional[List]` | Generated media |

### ToolExecution

```python
from definable.model import ToolExecution
```

Tracks a single tool/function call including HITL (human-in-the-loop) fields.

| Field | Type | Description |
|-------|------|-------------|
| `tool_name` | `Optional[str]` | Name of the called tool |
| `tool_args` | `Optional[Dict]` | Arguments passed to the tool |
| `result` | `Optional[str]` | Execution result |
| `requires_confirmation` | `Optional[bool]` | Needs user confirmation before running |
| `requires_user_input` | `Optional[bool]` | Needs additional user input |
| `external_execution_required` | `Optional[bool]` | Must be executed externally |

### Metrics

```python
from definable.model import Metrics
```

Token and timing statistics for a single model call.

| Field | Type | Description |
|-------|------|-------------|
| `input_tokens` | `int` | Input token count |
| `output_tokens` | `int` | Output token count |
| `total_tokens` | `int` | Total token count |
| `cost` | `Optional[float]` | Computed cost in USD |
| `reasoning_tokens` | `int` | Tokens used for thinking/reasoning |
| `cache_read_tokens` | `int` | Tokens served from cache |
| `time_to_first_token` | `Optional[float]` | TTFT in seconds |
| `duration` | `Optional[float]` | Total invocation duration in seconds |

Supports `+` for aggregation across calls:

```python
combined = metrics_a + metrics_b
print(combined.total_tokens)  # sum of both
```

### Citations and MessageReferences

```python
from definable.model import Citations, MessageReferences
```

- `Citations` — URL and document citations returned alongside a response (used by Perplexity, Claude, etc.).
- `MessageReferences` — Cross-message reference linking for multi-turn context.

## Provider-Specific Features

### Claude (Anthropic)

```python
from definable.model import Claude

model = Claude(id="claude-sonnet-4-5-20250929")
# Or: Claude(id="claude-opus-4-6")
```

Special features:
- **MCP servers** — pass MCP server configurations for tool augmentation
- **Extended thinking** — native thinking/reasoning token support
- **Extended cache** — Anthropic prompt caching (cache_response=True)

### Gemini (Google)

```python
from definable.model import Gemini  # requires: pip install google-genai

model = Gemini(id="gemini-2.0-flash-001")
# Vertex AI: configure with project/location credentials
```

Special features:
- **Vertex AI support** — authenticate with Google Cloud credentials
- **Google Search grounding** — enable live web search in responses
- **Thinking budget/level** — control extended thinking token budget

### Ollama (Local)

```python
from definable.model import Ollama  # requires: pip install ollama

model = Ollama(id="llama3.1")
# Or: Ollama(id="mistral")
```

Special features:
- **Local models** — runs fully offline via the Ollama daemon
- **Host config** — set a custom `host` to point at a remote Ollama instance
- **Format options** — raw/json output format control
- **keep_alive** — control how long the model stays loaded in memory

### OpenRouter

```python
from definable.model import OpenRouter

model = OpenRouter(id="gpt-4o")
# Dynamic routing across multiple models:
# model = OpenRouter(models=["gpt-4o", "claude-opus-4-6"])
```

Special features:
- **Dynamic model routing** via `models=` param — OpenRouter picks the best available provider
- Routes to 100+ models from a single API key

### Perplexity

```python
from definable.model import Perplexity

model = Perplexity(id="sonar")
```

Special features:
- **Web search** — responses include live search results with `Citations` attached to `ModelResponse`

### MistralChat

```python
from definable.model import MistralChat

model = MistralChat(id="mistral-large-latest")
```

Special features:
- **safe_mode / safe_prompt** — enable Mistral's built-in content safety layer

### OpenAILike (Custom Endpoints)

```python
from definable.model import OpenAILike

model = OpenAILike(
  id="my-model",
  base_url="https://my-provider.com/v1",
  api_key="my-key",
)
```

Use this for any provider with an OpenAI-compatible API that is not already listed above.

## Pricing

```python
from definable.model.pricing import get_pricing, calculate_cost, ModelPricing

# Look up pricing by provider and model
p = get_pricing("openai", "gpt-4o")
# p.input_per_million    — USD per 1M input tokens
# p.output_per_million   — USD per 1M output tokens
# p.cached_input_per_million  — USD per 1M cached input tokens
# p.reasoning_per_million     — USD per 1M reasoning tokens

# Calculate cost from a Metrics object
cost = calculate_cost("openai", "gpt-4o", metrics)
```

Pricing data is loaded from `model_pricing.json` via `PricingRegistry` (singleton). Returns `None` for unknown models rather than raising.

## Resilience

The resilience layer adds key rotation and provider failover without changing the model's interface. `ResilientModel` wraps any `Model` subclass and delegates all four invoke methods transparently.

### KeyPool

Thread-safe pool of API keys with rotation strategies and health tracking.

```python
from definable.model import KeyPool

pool = KeyPool(keys=["sk-key1", "sk-key2", "sk-key3"])
# pool.size == 3

# Selection strategies: round_robin (default) or lru
from definable.model.resilience.key_pool import SelectionStrategy

pool = KeyPool(keys=["sk-1", "sk-2"], strategy=SelectionStrategy.LEAST_RECENTLY_USED)

# Manual key lifecycle
key = pool.acquire()
pool.mark_success(key)
pool.mark_failure(key)
pool.mark_rate_limited(key)  # triggers exponential backoff cooldown

# Inspect health
health = pool.get_health(key)
# health.success_count, failure_count, rate_limit_count
# health.is_available, health.success_rate, health.error_rate
print(pool.available_count())  # keys not in cooldown
```

Constructor parameters:

| Param | Default | Description |
|-------|---------|-------------|
| `keys` | required | List of unique API key strings |
| `strategy` | `round_robin` | `round_robin` or `lru` |
| `base_cooldown` | `60.0` | Base cooldown seconds for rate-limited keys |
| `max_cooldown` | `300.0` | Maximum cooldown cap in seconds |

Rate-limited keys are placed in exponential backoff: `base_cooldown * 2^(consecutive_failures - 1)`, capped at `max_cooldown`.

### FailoverChain

Ordered list of provider fallbacks, sorted by priority (lower = tried first).

```python
from definable.model import FailoverChain, FailoverEntry, OpenAIChat, Claude

primary = OpenAIChat(id="gpt-4o")
backup = Claude(id="claude-sonnet-4-5-20250929")

chain = FailoverChain(
  entries=[
    FailoverEntry(model=primary, key_pool=KeyPool(keys=["sk-1"]), priority=0),
    FailoverEntry(model=backup, priority=1),
  ]
)
# len(chain) == 2
# chain.primary → the priority=0 entry
```

`FailoverEntry` parameters:

| Param | Default | Description |
|-------|---------|-------------|
| `model` | required | The `Model` instance for this entry |
| `key_pool` | `None` | Optional `KeyPool` for key rotation on this provider |
| `priority` | `0` | Lower values are tried first |

### ResilientModel

Wraps any `Model` with automatic key rotation on 429 errors and provider failover on other errors.

```python
from definable.model import ResilientModel, KeyPool, FailoverChain, FailoverEntry
from definable.model import OpenAIChat, Claude

pool = KeyPool(keys=["sk-prod-1", "sk-prod-2", "sk-prod-3"])

model = ResilientModel(
  inner=OpenAIChat(id="gpt-4o"),
  key_pool=pool,
)

# With failover chain:
model = ResilientModel(
  inner=OpenAIChat(id="gpt-4o"),
  key_pool=pool,
  failover=FailoverChain(
    entries=[
      FailoverEntry(model=Claude(id="claude-sonnet-4-5-20250929"), priority=0),
    ]
  ),
  max_key_retries=3,
  on_key_rotated=lambda event: print(f"Rotated: {event.old_key_prefix} → {event.new_key_prefix}"),
  on_failover=lambda event: print(f"Failover: {event.from_model_id} → {event.to_model_id}"),
)

# Use with Agent — transparent to the rest of the system:
from definable.agent import Agent

agent = Agent(model=model)
```

Constructor parameters:

| Param | Default | Description |
|-------|---------|-------------|
| `inner` | required | The primary `Model` to wrap |
| `key_pool` | `None` | `KeyPool` for automatic key rotation on 429s |
| `failover` | `None` | `FailoverChain` for provider-level failover |
| `max_key_retries` | `3` | Max key rotation attempts before failing over |
| `on_key_rotated` | `None` | Callback receiving a `KeyRotatedEvent` |
| `on_failover` | `None` | Callback receiving a `ProviderFailoverEvent` |

Resilience events:

```python
from definable.model.resilience.events import KeyRotatedEvent, ProviderFailoverEvent

# KeyRotatedEvent fields: old_key_prefix (str), new_key_prefix (str), reason (str)
# ProviderFailoverEvent fields: from_model_id (str), to_model_id (str), reason (str)
```

**Note:** `ResilientModel` is NOT a subclass of `Model`. It is a wrapper that proxies all `invoke`/`ainvoke`/`invoke_stream`/`ainvoke_stream` calls and delegates unknown attribute access to `inner`.

## Gotchas

| Mistake | Correct approach |
|---------|-----------------|
| `images=["url"]` — plain strings in images field | Use `Image` objects from `definable.media` |
| Forgetting `assistant_message` parameter | All four invoke methods require it as a positional arg |
| `resolve_model_string("unknown/model")` | Raises `ValueError` — check `get_supported_providers()` first |
| `KeyPool(keys=[])` | Raises `ValueError` — at least one key required |
| `KeyPool(keys=["k", "k"])` | Raises `ValueError` — keys must be unique |
| `FailoverChain(entries=[])` | Raises `ValueError` — at least one entry required |
| `ResilientModel(inner=None)` | Raises `ValueError` — `inner` is required |
| `Gemini` without `google-genai` | Raises `ImportError` at import time |
| `Ollama` without `ollama` | Raises `ImportError` at import time |
| `Claude` without `anthropic` SDK | Raises `ImportError` with install hint |
| Calling `model.invoke()` synchronously in multi-turn loops | Use async (`ainvoke`) for sequential multi-turn calls |

## See Also

- `agent/` — `Agent` wraps a model with tools, middleware, memory, and orchestration
- `tool/` — `Function` class and `@tool` decorator for tool definitions
- `agent/run/` — `RunOutput` and event types returned by agent execution
- `knowledge/` — RAG pipeline connecting knowledge sources to agents
