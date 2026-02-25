# agent/compression

Context window compression for long-running agents that make many tool calls. When a tool result count or token threshold is crossed, the `CompressionManager` summarizes accumulated tool outputs in place — preserving key facts while shrinking context. Compression is non-destructive: the original content is stored in `Message.compressed_content` alongside the original, so nothing is lost.

## Module structure

```
compression/
├── __init__.py    # Public API — exports CompressionManager
└── manager.py     # CompressionManager implementation
```

## Quick start

The recommended entry point is `CompressionConfig` on the agent, not `CompressionManager` directly.

```python
from definable.agent import Agent
from definable.agent.config import AgentConfig, CompressionConfig

agent = Agent(
  model="openai/gpt-4o-mini",
  tools=[search, browse],
  config=AgentConfig(
    compression=CompressionConfig(
      enabled=True,
      tool_results_limit=3,   # compress after 3 uncompressed tool results
    ),
  ),
)

result = await agent.arun("Research the latest developments in fusion energy.")
```

The agent loop calls `CompressionManager.ashould_compress()` after every tool round and invokes `acompress()` automatically when the threshold is crossed.

## Configuration — CompressionConfig

`CompressionConfig` lives in `definable.agent.config` and is the primary API for enabling compression.

```python
from definable.agent.config import CompressionConfig

CompressionConfig(
  enabled=True,                    # bool — master switch
  model=None,                      # str | Model | None — compression model (default: agent's model)
  tool_results_limit=3,            # int | None — compress after N uncompressed tool results
  token_limit=None,                # int | None — compress when token count exceeds this
  instructions=None,               # str | None — custom compression prompt
)
```

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `True` | Whether compression is active. Set `False` to disable without removing config. |
| `model` | `None` | Model for summarization. `None` uses the agent's own model. A `Model` instance uses that model exclusively for compression calls. String model specs resolve to the agent's model. |
| `tool_results_limit` | `3` | Trigger compression after this many uncompressed tool result messages accumulate. Set `None` to disable count-based triggering. |
| `token_limit` | `None` | Trigger compression when the context token count reaches this threshold. Requires a model with token counting support. |
| `instructions` | `None` | Custom prompt for the compressor. Replaces the default prompt entirely. |

**At least one threshold must be set.** If both `tool_results_limit` and `token_limit` are `None`, the `CompressionManager` defaults to `tool_results_limit=3`.

**Thresholds are OR-combined.** Either condition triggering causes compression. Token-based triggering is checked first when `token_limit` is set.

## CompressionManager — internal API

`CompressionManager` is instantiated internally by the agent. Direct use is for testing and custom pipelines only.

```python
from definable.agent.compression import CompressionManager
from definable.model.openai import OpenAIChat

manager = CompressionManager(
  model=OpenAIChat(id="gpt-4o-mini"),
  compress_tool_results=True,
  compress_tool_results_limit=3,    # mirrors CompressionConfig.tool_results_limit
  compress_token_limit=None,        # mirrors CompressionConfig.token_limit
  compress_tool_call_instructions=None,  # mirrors CompressionConfig.instructions
)
```

**Methods**

| Method | Signature | Description |
|--------|-----------|-------------|
| `should_compress` | `(messages, tools=None, model=None, response_format=None) -> bool` | Sync threshold check. |
| `ashould_compress` | `async (messages, tools=None, model=None, response_format=None) -> bool` | Async threshold check (uses `model.acount_tokens` for token-based triggering). |
| `compress` | `(messages: list[Message]) -> None` | Compress all uncompressed tool results. Sync. Mutates messages in place. |
| `acompress` | `async (messages: list[Message]) -> None` | Async version — compresses all uncompressed tool results in parallel via `asyncio.gather`. |

**Stats tracking**

After `compress()` or `acompress()`, the `manager.stats` dict is updated:

```python
{
  "tool_results_compressed": int,  # total tool results compressed this session
  "original_size": int,            # total original content length (chars)
  "compressed_size": int,          # total compressed content length (chars)
}
```

These stats are also surfaced in `CompressionCompletedEvent`.

## Events

Compression emits two events into the agent's event stream:

```python
from definable.agent.events import CompressionStartedEvent, CompressionCompletedEvent

# Emitted just before compression begins
CompressionStartedEvent(
  tool_results_count=3,  # number of tool results about to be compressed
)

# Emitted after compression finishes
CompressionCompletedEvent(
  tool_results_compressed=3,   # count of results compressed
  original_size=4820,          # total chars before compression
  compressed_size=312,         # total chars after compression
  duration_ms=1240.5,          # wall time
)
```

Subscribe via `agent.on_event()` or inspect the `RunOutput.events` list:

```python
from definable.agent.events import CompressionCompletedEvent

result = await agent.arun("Long research task")

for event in result.events:
  if isinstance(event, CompressionCompletedEvent):
    ratio = 1 - event.compressed_size / event.original_size
    print(f"Compressed {event.tool_results_compressed} results, saved {ratio:.0%}")
```

## Default compression prompt

When `instructions` is `None`, the manager uses a built-in prompt that instructs the compression model to preserve facts, entities, numbers, URLs, dates, and identifiers while stripping prose, hedging language, and formatting artifacts. The example in the prompt shows an 87% size reduction while retaining all critical facts.

To override:

```python
CompressionConfig(
  instructions="Summarize the tool output in one concise sentence, preserving all URLs and numbers.",
)
```

The custom `instructions` string replaces the entire system prompt sent to the compression model.

## Integration with the agent loop

The agent initializes a `CompressionManager` during `__init__` via `_init_compression()` when `config.compression.enabled` is `True`. The compression manager is then passed to the pipeline loop (`AgentLoop`) which calls `ashould_compress()` and `acompress()` after each tool round, before the next model invocation.

```
[Tool round complete]
  → ashould_compress(messages)
  → if True:
      emit CompressionStartedEvent
      acompress(messages)           ← parallel compression via asyncio.gather
      emit CompressionCompletedEvent
  → model.ainvoke(messages)         ← compressor summaries are used going forward
```

Compression does not discard the original content — it sets `Message.compressed_content` on tool messages. If the pipeline re-reads the original content (e.g. for tracing or replay), it is still available.

## Choosing thresholds

**`tool_results_limit`** works well when individual tool results are large and varied in size. A limit of 3–5 is a good starting point for research agents calling web search or browser tools.

**`token_limit`** is more precise but requires the model to support token counting. Use it when you want to compress based on remaining context budget:

```python
CompressionConfig(
  tool_results_limit=None,
  token_limit=80_000,  # compress at 80k tokens for a 128k context model
)
```

**Both thresholds together** is also valid — the first to trigger wins:

```python
CompressionConfig(
  tool_results_limit=5,
  token_limit=60_000,
)
```

## Using a separate compression model

By default, the agent's own model compresses its tool results. For cost or latency reasons, you may prefer a smaller model for compression:

```python
from definable.model.openai import OpenAIChat
from definable.agent.config import CompressionConfig

CompressionConfig(
  model=OpenAIChat(id="gpt-4o-mini"),  # fast, cheap, good at summarization
  tool_results_limit=3,
)
```

Pass a `Model` instance — string shorthand is accepted by `CompressionConfig` but resolves to the agent's model at init time.

## Gotchas

**`acompress` parallelizes; `compress` does not.** Always use the async path (`agent.arun` + the default async loop) in production. The sync `compress()` method compresses results sequentially and blocks the event loop.

**Compression mutates messages in place.** `CompressionManager.acompress()` sets `msg.compressed_content` on each tool message directly. The messages list passed to the agent loop is modified. This is intentional — the agent carries forward the compressed representation while retaining the original for observability.

**A failed compression falls back to the original content.** If the compression model call raises, the original tool content is stored in `compressed_content` unchanged and a warning is logged. The agent continues without interruption.

**Compression counts uncompressed tool results, not total tool calls.** `should_compress()` counts messages with `role == "tool"` and `compressed_content is None`. Once a batch is compressed, those messages no longer count toward future triggers.

**Token-based triggering counts the full context.** `ashould_compress()` calls `model.acount_tokens(messages, tools, response_format)` which counts all messages including system, user, and assistant turns — not just tool results. Set `token_limit` with the model's actual context window in mind.

**`enabled=False` is a hard off.** Setting `config.compression = CompressionConfig(enabled=False)` prevents the `CompressionManager` from being created at all. `should_compress()` returns `False` immediately regardless of message count or token count.
