# Model Resilience

> Key rotation, provider failover, and health tracking for Definable AI model calls.

The resilience layer wraps any Model with automatic key rotation on rate limits (429s) and provider failover on errors. Keys are managed in a thread-safe pool with exponential backoff cooldowns. Multiple providers can be chained in priority order so that if the primary fails, the system automatically falls through to backups.

## Quick Start

```python
from definable.model.resilience import KeyPool, ResilientModel, FailoverChain, FailoverEntry
from definable.model.openai import OpenAIChat

# Key rotation: rotate through 3 API keys on rate limits
pool = KeyPool(keys=["sk-key1", "sk-key2", "sk-key3"])

model = ResilientModel(
  inner=OpenAIChat(id="gpt-4o-mini"),
  key_pool=pool,
  max_key_retries=3,
)

# Provider failover: fall through to backup on errors
chain = FailoverChain(
  entries=[
    FailoverEntry(model=OpenAIChat(id="gpt-4o-mini"), priority=0),
    FailoverEntry(model=OpenAIChat(id="gpt-4o"), priority=1),
  ]
)

model = ResilientModel(
  inner=OpenAIChat(id="gpt-4o-mini"),
  key_pool=pool,
  failover=chain,
)

# Use as a drop-in Model replacement
from definable.agent import Agent

agent = Agent(model=model)
```

## Architecture

```
ResilientModel (wrapper)
  |
  +-- inner: Model                  -- the primary model provider
  +-- key_pool: KeyPool             -- multi-key rotation with health tracking
  |     +-- KeyHealth[]             -- per-key success/failure/rate-limit stats
  |     +-- SelectionStrategy       -- round_robin | lru
  |     +-- exponential backoff     -- cooldown on 429s
  |
  +-- failover: FailoverChain       -- ordered provider list
  |     +-- FailoverEntry[]         -- (model, key_pool, priority)
  |
  +-- Callbacks
        +-- on_key_rotated(KeyRotatedEvent)
        +-- on_failover(ProviderFailoverEvent)
```

### Module Structure

```
model/resilience/
+-- __init__.py         # Public API: KeyPool, KeyHealth, SelectionStrategy,
|                       #   FailoverChain, FailoverEntry, ResilientModel
+-- key_pool.py         # KeyPool, KeyHealth, SelectionStrategy
+-- failover.py         # FailoverChain, FailoverEntry
+-- resilient.py        # ResilientModel (delegates with retry + failover)
+-- events.py           # KeyRotatedEvent, ProviderFailoverEvent
```

### How It Connects

```
Agent
  +-- model: Model | ResilientModel
        |
        ResilientModel
          +-- invoke() -----> try inner.invoke()
          |                     |-- success --> mark_success(key)
          |                     |-- 429 -----> rotate_key, retry (up to max_key_retries)
          |                     |-- error ---> mark_failure(key), failover to next provider
          |
          +-- ainvoke() ----> same pattern, async
          +-- invoke_stream() --> wraps generator with failover
          +-- ainvoke_stream() -> wraps async generator with failover
```

## API Reference

### SelectionStrategy

Enum for key selection strategy in the pool.

```python
from definable.model.resilience import SelectionStrategy

SelectionStrategy.ROUND_ROBIN  # "round_robin" -- cycle through keys in order
SelectionStrategy.LEAST_RECENTLY_USED  # "lru" -- pick the key used least recently
```

### KeyHealth

Per-key health tracking dataclass.

```python
from definable.model.resilience import KeyHealth

health = KeyHealth(
  key="sk-key1",  # The API key string
  success_count=0,  # Successful requests
  failure_count=0,  # Failed requests (non-429)
  rate_limit_count=0,  # Rate-limited requests (429)
  last_used=0.0,  # Unix timestamp of last use
  cooldown_until=0.0,  # Unix timestamp when cooldown expires
  consecutive_failures=0,  # Consecutive failures (resets on success)
)
```

**Properties:**

| Property | Return | Description |
|----------|--------|-------------|
| `is_available` | `bool` | True if not currently in cooldown |
| `total_requests` | `int` | `success_count + failure_count + rate_limit_count` |
| `success_rate` | `float` | 0.0-1.0 (returns 1.0 when no requests) |
| `error_rate` | `float` | 0.0-1.0 (failures + rate limits / total) |

```python
from definable.model.resilience import KeyPool

pool = KeyPool(keys=["sk-key1", "sk-key2"])
key = pool.acquire()
pool.mark_success(key)

health = pool.get_health(key)
print(health.success_count)  # 1
print(health.success_rate)  # 1.0
print(health.is_available)  # True
```

### KeyPool

Thread-safe pool of API keys with rotation and health tracking.

```python
from definable.model.resilience import KeyPool, SelectionStrategy

pool = KeyPool(
  keys=["sk-key1", "sk-key2", "sk-key3"],  # Must be unique, at least 1
  strategy=SelectionStrategy.ROUND_ROBIN,  # Default: round_robin
  base_cooldown=60.0,  # Base cooldown for rate limits (seconds)
  max_cooldown=300.0,  # Max cooldown cap (seconds)
)
```

**Properties:**

| Property | Return | Description |
|----------|--------|-------------|
| `size` | `int` | Total number of keys in the pool |
| `strategy` | `SelectionStrategy` | Current selection strategy |

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `acquire` | `pool.acquire() -> str` | Get the next available key. Raises `RuntimeError` if all in cooldown |
| `mark_success` | `pool.mark_success(key)` | Record a successful request (resets consecutive failures) |
| `mark_failure` | `pool.mark_failure(key)` | Record a failed request |
| `mark_rate_limited` | `pool.mark_rate_limited(key)` | Record a 429 and apply exponential backoff cooldown |
| `get_health` | `pool.get_health(key) -> KeyHealth \| None` | Get health info for a specific key |
| `all_health` | `pool.all_health() -> list[KeyHealth]` | Get health info for all keys |
| `available_count` | `pool.available_count() -> int` | Number of keys not in cooldown |
| `reset` | `pool.reset(key=None)` | Reset health for one key (or all if None) |

```python
from definable.model.resilience import KeyPool, SelectionStrategy

pool = KeyPool(keys=["sk-key1", "sk-key2", "sk-key3"], strategy=SelectionStrategy.ROUND_ROBIN)
print(pool.size)  # 3
print(pool.available_count())  # 3

key = pool.acquire()  # "sk-key1"
pool.mark_success(key)
print(pool.get_health(key).success_count)  # 1

pool.mark_rate_limited("sk-key1")
print(pool.get_health("sk-key1").is_available)  # False (on cooldown)
print(pool.available_count())  # 2

# Exponential backoff: base_cooldown * 2^(consecutive_failures - 1)
# 1st rate limit: 60s cooldown
# 2nd rate limit: 120s cooldown
# 3rd rate limit: 240s cooldown (capped at max_cooldown=300s)

pool.reset()  # Reset all keys
print(pool.available_count())  # 3
```

### FailoverEntry

A single entry in a failover chain.

```python
from definable.model.resilience import FailoverEntry
from definable.model.openai import OpenAIChat

entry = FailoverEntry(
  model=OpenAIChat(id="gpt-4o-mini"),  # Required: the model provider
  key_pool=None,  # Optional: per-provider key rotation
  priority=0,  # Lower = tried first (default 0)
)
```

### FailoverChain

Ordered list of failover providers, sorted by priority.

```python
from definable.model.resilience import FailoverChain, FailoverEntry
from definable.model.openai import OpenAIChat

chain = FailoverChain(
  entries=[
    FailoverEntry(model=OpenAIChat(id="gpt-4o-mini"), priority=0),
    FailoverEntry(model=OpenAIChat(id="gpt-4o"), priority=1),
  ]
)

print(chain.primary.model.id)  # "gpt-4o-mini" (lowest priority number)
print(len(chain))  # 2

# Iterable -- iterate in priority order
for entry in chain:
  print(entry.model.id)
# Output: gpt-4o-mini, gpt-4o
```

**Properties:**

| Property | Return | Description |
|----------|--------|-------------|
| `primary` | `FailoverEntry` | Highest-priority (lowest number) entry |
| `entries` | `list[FailoverEntry]` | Copy of sorted entries list |

### ResilientModel

Wraps a Model with transparent key rotation and provider failover. Delegates all Model methods (`invoke`, `ainvoke`, `invoke_stream`, `ainvoke_stream`) with resilience logic.

```python
from definable.model.resilience import ResilientModel, KeyPool, FailoverChain, FailoverEntry
from definable.model.openai import OpenAIChat

model = ResilientModel(
  inner=OpenAIChat(id="gpt-4o-mini"),  # Required: primary model
  key_pool=KeyPool(keys=["sk-1", "sk-2"]),  # Optional: key rotation
  failover=FailoverChain(
    entries=[  # Optional: provider failover
      FailoverEntry(model=OpenAIChat(id="gpt-4o"), priority=1),
    ]
  ),
  max_key_retries=3,  # Max key rotation attempts per call (default 3)
  on_key_rotated=None,  # Callback for KeyRotatedEvent
  on_failover=None,  # Callback for ProviderFailoverEvent
)
```

**Identity mirroring:** ResilientModel copies `id`, `name`, and `provider` from the inner model. Unknown attribute access is proxied to the inner model.

```python
rm = ResilientModel(
  inner=OpenAIChat(id="gpt-4o-mini"),
  key_pool=KeyPool(keys=["sk-key1", "sk-key2"]),
  max_key_retries=3,
)
print(rm.id)  # "gpt-4o-mini"
```

**Resilience flow:**

1. Call `inner.invoke(...)` with current key
2. On success: `mark_success(key)`, return result
3. On `ModelRateLimitError` (429): `mark_rate_limited(key)`, rotate to next key, retry (up to `max_key_retries`)
4. On `ModelProviderError`: `mark_failure(key)`, attempt failover
5. Failover: iterate through `FailoverChain` entries in priority order until one succeeds
6. If all fail: raise `ModelProviderError("All providers exhausted")`

### Events

```python
from definable.model.resilience.events import KeyRotatedEvent, ProviderFailoverEvent

# Emitted on key rotation
KeyRotatedEvent(
  old_key_prefix="sk-key1..",  # First 8 chars of old key
  new_key_prefix="sk-key2..",  # First 8 chars of new key
  reason="rate_limited",  # Always "rate_limited"
)

# Emitted on provider failover
ProviderFailoverEvent(
  from_model_id="gpt-4o-mini",  # Model that failed
  to_model_id="gpt-4o",  # Model that succeeded
  reason="...",  # Error message from the failure
)
```

## Patterns

### Key Rotation Only

```python
from definable.model.resilience import ResilientModel, KeyPool
from definable.model.openai import OpenAIChat

model = ResilientModel(
  inner=OpenAIChat(id="gpt-4o-mini"),
  key_pool=KeyPool(keys=["sk-key1", "sk-key2", "sk-key3"]),
)
```

### Provider Failover Only

```python
from definable.model.resilience import ResilientModel, FailoverChain, FailoverEntry
from definable.model.openai import OpenAIChat

model = ResilientModel(
  inner=OpenAIChat(id="gpt-4o-mini"),
  failover=FailoverChain(
    entries=[
      FailoverEntry(model=OpenAIChat(id="gpt-4o"), priority=1),
    ]
  ),
)
```

### Full Resilience Stack

```python
from definable.model.resilience import (
  ResilientModel,
  KeyPool,
  FailoverChain,
  FailoverEntry,
  SelectionStrategy,
)
from definable.model.openai import OpenAIChat

primary_pool = KeyPool(
  keys=["sk-primary-1", "sk-primary-2"],
  strategy=SelectionStrategy.ROUND_ROBIN,
  base_cooldown=30.0,
)

backup_pool = KeyPool(
  keys=["sk-backup-1"],
  strategy=SelectionStrategy.ROUND_ROBIN,
)

model = ResilientModel(
  inner=OpenAIChat(id="gpt-4o-mini"),
  key_pool=primary_pool,
  failover=FailoverChain(
    entries=[
      FailoverEntry(
        model=OpenAIChat(id="gpt-4o"),
        key_pool=backup_pool,
        priority=1,
      ),
    ]
  ),
  max_key_retries=3,
  on_key_rotated=lambda e: print(f"Key rotated: {e.old_key_prefix} -> {e.new_key_prefix}"),
  on_failover=lambda e: print(f"Failover: {e.from_model_id} -> {e.to_model_id}"),
)
```

### Monitoring Key Health

```python
from definable.model.resilience import KeyPool

pool = KeyPool(keys=["sk-key1", "sk-key2", "sk-key3"])

# After some usage...
for health in pool.all_health():
  print(
    f"Key {health.key[:8]}... | "
    f"success={health.success_count} fail={health.failure_count} "
    f"rate_limited={health.rate_limit_count} | "
    f"success_rate={health.success_rate:.1%} | "
    f"available={health.is_available}"
  )
```

## Gotchas

| Issue | Solution |
|-------|----------|
| `KeyPool(keys=[])` | Raises `ValueError` -- at least one key required |
| `KeyPool` with duplicate keys | Raises `ValueError` -- keys must be unique |
| All keys in cooldown | `acquire()` raises `RuntimeError("All keys are in cooldown")` |
| `FailoverChain(entries=[])` | Raises `ValueError` -- at least one entry required |
| `ResilientModel(inner=None)` | Raises `ValueError` -- inner model is required |
| ResilientModel is NOT a Model subclass | It is a dataclass wrapper. It delegates all attribute access to `inner` via `__getattr__` |
| Failover changes identity | After failover, `rm.id` and `rm.provider` update to the new provider |
| Key injection clears cached clients | `_client` and `_async_client` are set to None when a key is injected |
| Streaming failover is per-generator | If the stream fails mid-generation, failover starts a new stream from scratch |

## Related Modules

- **[Model](../README.md)** -- Base Model class that ResilientModel wraps
- **[Agent](../../agent/README.md)** -- Agent accepts ResilientModel via the `model=` parameter
- **[Security](../../agent/security/README.md)** -- Rate limiting at the interface level (complementary)
