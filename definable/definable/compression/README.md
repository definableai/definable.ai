# compression

Context window compression for long conversations with many tool results.

## Overview

The `CompressionManager` automatically compresses tool result messages to keep conversations within model context limits. It can be triggered by either a tool result count threshold or a token count threshold. Compression is non-destructive — original content is preserved alongside compressed summaries.

## API Reference

### CompressionManager

```python
from definable.compression import CompressionManager
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `Optional[Model]` | `None` | Model used for summarization (defaults to agent's model) |
| `compress_tool_results` | `bool` | `True` | Enable compression |
| `compress_tool_results_limit` | `Optional[int]` | `None` | Compress after N uncompressed tool results |
| `compress_token_limit` | `Optional[int]` | `None` | Compress when token count exceeds limit |
| `compress_tool_call_instructions` | `Optional[str]` | `None` | Custom compression prompt |

**Methods:**

| Method | Description |
|--------|-------------|
| `should_compress(messages, ...)` | Check if compression is needed (sync) |
| `ashould_compress(messages, ...)` | Check if compression is needed (async) |
| `compress(messages)` | Compress tool results in place (sync) |
| `acompress(messages)` | Compress tool results in place (async) |

Typically configured via `CompressionConfig` on `AgentConfig` rather than used directly.

## See Also

- `agents/config.py` — `CompressionConfig` for agent-level configuration
- `agents/agent.py` — Automatic compression during agent execution
