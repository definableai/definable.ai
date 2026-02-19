---
name: definable-code-patterns
description: Authoritative guide for writing code in the Definable AI framework. Covers architecture, conventions, patterns, formatting, testing, and module-specific idioms. Use when writing, reviewing, or modifying any code in the definable package, adding new modules, creating new store/provider/interface implementations, writing tools, or extending the agent system.
---

# Definable AI Framework — Code Patterns

## Formatting Rules

- **2-space indentation** (not 4, not tabs)
- **Double quotes** for strings
- **150-character line length** max
- **Target Python 3.10+** syntax compatibility (runtime requires 3.12+)
- Ruff is the linter/formatter. Config lives in `ruff.toml`.

## Import Conventions

**Absolute imports** from `definable.*`:

```python
from definable.agents.agent import Agent
from definable.models.openai import OpenAIChat
from definable.utils.log import log_debug, log_warning
```

**TYPE_CHECKING guard** for circular or heavy imports:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.knowledge import Knowledge
  from definable.models.base import Model
```

**Lazy imports in `__init__.py`** — use the `_LAZY_IMPORTS` dict pattern:

```python
_LAZY_IMPORTS = {
  "SQLiteMemoryStore": ("definable.memory.store.sqlite", "SQLiteMemoryStore"),
  "InMemoryStore": ("definable.memory.store.in_memory", "InMemoryStore"),
}

def __getattr__(name: str):
  if name in _LAZY_IMPORTS:
    module_path, class_name = _LAZY_IMPORTS[name]
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

**Optional dependency imports** — inside the method that needs them, with actionable error:

```python
try:
  from pinecone import Pinecone
except ImportError as e:
  raise ImportError(
    "pinecone is required for PineconeMemoryStore. "
    "Install it with: pip install definable[pinecone-memory]"
  ) from e
```

## Type Annotations

- Use `typing` capitalized forms: `List[X]`, `Dict[K, V]`, `Optional[T]`, `Union[A, B]`
- `Optional[T]` for nullable params (not `T | None`)
- `field(default_factory=list)` for mutable defaults in dataclasses
- Pydantic `BaseModel` for structured data objects (Function, FunctionCall, Message)
- `@dataclass` for configs and internal types (AgentConfig, Episode, KnowledgeAtom)
- `@runtime_checkable Protocol` for pluggable interfaces (MemoryStore, Middleware, AuthProvider)
- `ABC` with `@abstractmethod` for base classes with shared logic (Model, VectorDB, Embedder, Chunker)

## Configuration Pattern

All configs are **dataclasses with sensible defaults**:

```python
@dataclass
class ThinkingConfig:
  enabled: bool = True
  model: Optional["Model"] = None
  instructions: Optional[str] = None
  trigger: Literal["always", "auto", "never"] = "always"
  description: Optional[str] = None
```

For immutable configs, use `@dataclass(frozen=True)` and provide `with_updates()`:

```python
def with_updates(self, **kwargs) -> "AgentConfig":
  current = {k: v for k, v in asdict(self).items() if k not in non_serializable}
  current.update(kwargs)
  return AgentConfig(**current)
```

**Environment variable fallback** — in constructors, not configs:

```python
self._api_key = api_key or os.environ.get("PINECONE_API_KEY", "")
```

## Logging

Use module-level functions from `definable.utils.log`:

```python
from definable.utils.log import log_debug, log_info, log_warning, log_error

log_debug("Store initialized", log_level=2)     # Verbose debug (level 2)
log_debug("Query executed", log_level=1)         # Standard debug (level 1)
log_info("Agent started")                        # Lifecycle events
log_warning("Fallback to default embedder")      # Recoverable issues
log_error(f"Failed to connect: {e}")             # Errors
```

- `log_level=2` for noisy internals (per-record operations)
- `log_level=1` for standard debug (initialization, major steps)
- Never log secrets, API keys, or PII

## Error Handling

**Custom exception hierarchy** — defined in `definable/definable/exceptions.py`:

- `DefinableError(Exception)` — base, has `message`, `status_code`, `type`, `error_id`
- `ModelProviderError(DefinableError)` — provider API failures
- `ModelAuthenticationError(DefinableError)` — auth failures (401)
- `ModelRateLimitError(ModelProviderError)` — rate limits (429)
- `RemoteServerUnavailableError(DefinableError)` — connection failures (503)
- `AgentRunException(Exception)` — tool/agent-level errors with optional user/agent messages
- `RetryAgentRun(AgentRunException)` — retry the tool call
- `StopAgentRun(AgentRunException)` — halt execution entirely
- `InputCheckError` / `OutputCheckError` — guardrail violations

**Patterns:**

```python
# ValueError for bad arguments
if not collection:
  raise ValueError("Collection name must be provided.")

# ImportError for missing optional deps (with install instructions)
raise ImportError(
  "qdrant-client is required. Install with: pip install definable[qdrant-memory]"
)

# Non-fatal: log and continue
try:
  await self._distill_batch(episodes)
except Exception as e:
  log_warning(f"Distillation failed: {e}")

# Wrap external errors
except openai.APIError as e:
  raise ModelProviderError(str(e), model_name=self.name) from e
```

## Async Patterns

**Async is primary; sync wraps it:**

```python
# Primary async implementation
async def arun(self, message, **kwargs) -> RunOutput:
  ...

# Sync wrapper
def run(self, message, **kwargs) -> RunOutput:
  try:
    loop = asyncio.get_running_loop()
  except RuntimeError:
    loop = None

  if loop and loop.is_running():
    with concurrent.futures.ThreadPoolExecutor() as executor:
      future = executor.submit(asyncio.run, self.arun(message, **kwargs))
      return future.result()
  else:
    return asyncio.run(self.arun(message, **kwargs))
```

**Naming convention for sync/async pairs:**

| Sync | Async |
|------|-------|
| `run()` | `arun()` |
| `invoke()` | `ainvoke()` |
| `execute()` | `aexecute()` |
| `search()` | `asearch()` |
| `close()` | `async_close()` or `close()` (if always async) |

**Wrapping sync libraries** with `asyncio.to_thread()`:

```python
self._index = await asyncio.to_thread(_setup)
await asyncio.to_thread(self._index.upsert, vectors=[(id, vec, meta)], namespace="episodes")
```

**Fire-and-forget tasks** — store references, drain before exit:

```python
self._pending_tasks: list[asyncio.Task] = []
# Schedule
task = asyncio.create_task(self._store_memory(messages))
self._pending_tasks.append(task)
# Drain
await asyncio.gather(*self._pending_tasks, return_exceptions=True)
```

**Lazy initialization** — `_ensure_initialized()` pattern:

```python
async def _ensure_initialized(self) -> None:
  if not self._initialized:
    await self.initialize()
```

**Context managers** — always support both sync and async:

```python
async def __aenter__(self) -> "MyStore":
  await self.initialize()
  return self

async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
  await self.close()
```

## Adding a New Store Backend (Memory/VectorDB)

Follow this exact pattern — reference `pinecone.py` or `chroma.py` as templates.

1. **Create** `definable/definable/memory/store/<name>.py`
2. **Implement every method** from the `MemoryStore` protocol in `store/base.py`
3. **Constructor**: `api_key` with env-var fallback, backend-specific params, `vector_size` if applicable
4. **`initialize()`**: lazy-import the SDK, validate config, establish connection
5. **`close()`**: release resources, set `_initialized = False`
6. **Metadata mapping**: `_episode_to_metadata()` / `_metadata_to_episode()` helper pairs
7. **Register** in both `__init__.py` files:
   - `definable/definable/memory/store/__init__.py` → add to `_LAZY_IMPORTS`
   - `definable/definable/memory/__init__.py` → add to `_LAZY_IMPORTS`
8. **Add optional dep** in `pyproject.toml`: `<name>-memory = ["<sdk>>=<version>"]`
9. **Update docs** in `definable/docs/memory/stores.mdx` (table + section)
10. **Add to smoke test** in `definable/examples/memory/03_store_backends.py`

## Adding a New Model Provider

Follow the `OpenAILike` pattern for OpenAI-compatible APIs.

1. **Create** `definable/definable/models/<provider>/chat.py`
2. **Inherit** from `Model` (or `OpenAILike` if OpenAI-compatible)
3. **Implement**: `invoke`, `ainvoke`, `invoke_stream`, `ainvoke_stream`, `_parse_provider_response`, `_parse_provider_response_delta`
4. **Return** `ModelResponse` from all methods
5. **Register** in `definable/definable/models/__init__.py` lazy imports
6. **Set** `provider`, `supports_native_structured_outputs`, `supports_json_schema_outputs`

## Adding a New Tool

```python
from definable.tools.decorator import tool

@tool
def my_tool(query: str, limit: int = 10) -> str:
  """Search for documents matching the query.

  Args:
    query: The search query string.
    limit: Maximum number of results to return.
  """
  return f"Found {limit} results for: {query}"
```

Rules:
- Docstring is **required** — first line becomes the tool description
- Docstring `Args:` block provides parameter descriptions for the LLM
- Type hints are **required** — they generate the JSON schema
- Return type should be `str` (or something the LLM can consume)
- Use `Optional[T] = None` for optional params
- Async tools: just use `async def` — detected automatically

**Special injectable parameters** (excluded from schema, injected by agent):
- `agent` — the Agent instance
- `run_context` — current RunContext
- `session_state` — mutable dict persisted across turns
- `dependencies` — shared dependencies dict
- `images`, `videos`, `audios`, `files` — media from the user message

## Adding a New Toolkit

```python
from definable.agents.toolkit import Toolkit
from definable.tools.decorator import tool

@tool
def search(query: str) -> str:
  """Search the index."""
  return f"Results for {query}"

@tool
def summarize(text: str) -> str:
  """Summarize text."""
  return f"Summary: {text[:100]}"

class MyToolkit(Toolkit):
  search = search
  summarize = summarize
```

Toolkit auto-discovers all `Function`-typed class attributes. Shared dependencies can be injected via `Toolkit(dependencies={...})`.

## Adding a New Interface

1. **Inherit** from `BaseInterface` in `interfaces/base.py`
2. **Implement** abstract methods: `start()`, `stop()`, `send_response()`, `get_platform_name()`
3. **Respect** the 10-step message pipeline (auth, session, hooks, agent execution, response)
4. **Use** `SessionManager` for conversation state
5. **Register** in `definable/definable/interfaces/__init__.py` lazy imports

## Writing Tests

**File location**: `definable/tests_e2e/`

**Markers** (from `pytest.ini`):
- `@pytest.mark.unit` — pure logic, no API calls
- `@pytest.mark.integration` — real API calls required
- `@pytest.mark.behavioral` — agent-level outcome assertions
- `@pytest.mark.contract` — ABC compliance
- `@pytest.mark.regression` — snapshot-based
- `@pytest.mark.openai` — requires `OPENAI_API_KEY`

**Fixtures** — session-scoped for expensive resources:

```python
@pytest.fixture(scope="session")
def openai_model():
  api_key = os.getenv("OPENAI_API_KEY")
  if not api_key:
    pytest.skip("OPENAI_API_KEY not set")
  return OpenAIChat(api_key=api_key, id="gpt-4o-mini")
```

**MockModel** for deterministic tests:

```python
from definable.agents.testing import MockModel

agent = Agent(
  model=MockModel(responses=["Expected response"]),
  tools=[my_tool],
)
output = agent.run("test query")
assert "Expected" in output.content
```

**Async tests**: use `@pytest.mark.asyncio` (asyncio mode is auto).

## Writing Examples

**Location**: `definable/examples/<category>/NN_description.py`

**Structure**:

```python
"""
Title.

This example shows how to ...

Requirements:
    export OPENAI_API_KEY=sk-...
"""

from definable.agents import Agent
from definable.models.openai import OpenAIChat
from definable.tools.decorator import tool


def main():
  model = OpenAIChat(id="gpt-4o-mini")
  agent = Agent(model=model, instructions="...")
  output = agent.run("query")
  print(output.content)


if __name__ == "__main__":
  main()
```

## Writing Mintlify Docs

**Location**: `definable/docs/<section>/<page>.mdx`

**Navigation**: `definable/docs/docs.json`

**Rules**:
- Escape `<` in non-JSX contexts (e.g., use `fewer than 100` instead of `<100`)
- Use `<ParamField>`, `<CardGroup>`, `<CodeGroup>`, `<Tip>`, `<Warning>` components
- Code examples in docs must be copy-pasteable and correct
- When adding a new page, add it to `docs.json` navigation

## Private Method Naming

| Prefix | Purpose |
|--------|---------|
| `_init_*` | Initialization/setup |
| `_resolve_*` | Configuration resolution |
| `_build_*` | Content/message construction |
| `_format_*` | Formatting/serialization |
| `_parse_*` | Parsing/deserialization |
| `_emit()` | Event emission (tracing) |
| `_ensure_*` | Lazy initialization guards |
| `_drain_*` | Flush pending async tasks |

## Composition Philosophy

- Agent is the **composition root** — it accepts model, tools, toolkits, skills, memory, knowledge, readers, guardrails, and tracing as constructor params
- **No deep inheritance** — use composition and protocols
- Middleware chains via `.use()` (fluent, chainable)
- Configs are plain dataclasses, not builders or fluent objects
- Optional features default to `None` or `False` — zero-config by default
- Every optional capability is an explicit opt-in parameter

## Checklist for Any Change

1. 2-space indent, double quotes, 150-char lines
2. Absolute imports from `definable.*`
3. Optional deps: lazy import with `ImportError` + install instructions
4. New public class: register in relevant `__init__.py` lazy imports
5. New optional dep: add `pyproject.toml` extras group
6. Async methods are primary; sync wraps async
7. Errors: use framework exceptions, include actionable messages
8. Logging: `log_debug`/`log_info`/`log_warning`/`log_error` — no print()
9. Tests: add or update in `definable/tests_e2e/`
10. Docs: update if public API changed
