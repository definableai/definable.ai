# tools

Tool system — define functions that agents can call during execution.

## Quick Start

```python
from definable.tool.decorator import tool

@tool
def get_weather(city: str) -> str:
  """Get the current weather for a city."""
  return f"Sunny in {city}"

@tool(name="search", description="Search the web")
async def web_search(query: str, max_results: int = 5) -> str:
  """Search the web for information."""
  ...
```

## Module Structure

```
tools/
├── decorator.py   # @tool decorator
└── function.py    # Function class, UserInputField
```

**Note:** No `__init__.py` — import directly from submodules.

## API Reference

### @tool Decorator

```python
from definable.tool.decorator import tool
```

Converts a Python function into a `Function` object that agents can call. Supports both `@tool` and `@tool(...)` syntax. Works with sync, async, and async generator functions.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | function name | Tool name (a-z, A-Z, 0-9, _, -, max 64 chars) |
| `description` | `str` | docstring | Description for the model |
| `strict` | `bool` | `None` | Strict parameter validation |
| `instructions` | `str` | `None` | Additional instructions |
| `show_result` | `bool` | `True` | Show result to the model |
| `stop_after_tool_call` | `bool` | `False` | Stop agent loop after this tool |
| `requires_confirmation` | `bool` | `False` | Pause for user confirmation (HITL) |
| `requires_user_input` | `bool` | `False` | Pause for user input (HITL) |
| `external_execution` | `bool` | `False` | Pause for external execution (HITL) |
| `user_input_fields` | `list` | `None` | Schema for user input fields |
| `pre_hook` | `callable` | `None` | Called before tool execution |
| `post_hook` | `callable` | `None` | Called after tool execution |
| `cache_results` | `bool` | `False` | Cache tool results |
| `cache_dir` | `str` | `None` | Cache directory |
| `cache_ttl` | `int` | `None` | Cache TTL in seconds |

### Function

```python
from definable.tool.function import Function
```

Pydantic model representing a callable tool. Created by the `@tool` decorator or manually.

The `Function` class auto-generates a JSON Schema from the function signature for the model to use when calling the tool. It also provides internal fields (`_agent`, `_run_context`, `_session_state`, `_dependencies`) that are injected by the agent at execution time and available in tool functions.

### UserInputField

```python
from definable.tool.function import UserInputField
```

Defines a field for HITL user input:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Field name |
| `field_type` | `Type` | Python type |
| `description` | `Optional[str]` | Field description |
| `value` | `Optional[Any]` | Collected value |

## Usage with Agent

```python
from definable.agent import Agent
from definable.tool.decorator import tool

@tool
def calculate(expression: str) -> str:
  """Evaluate a math expression."""
  return str(eval(expression))

agent = Agent(
  model=model,
  tools=[calculate],
)
```

Tools can access agent context via special parameter names:

```python
@tool
def my_tool(query: str, run_context=None, session_state=None, dependencies=None):
  # run_context: RunContext for the current execution
  # session_state: mutable session state dict
  # dependencies: injected dependencies from AgentConfig
  ...
```

## See Also

- `agents/` — Agent accepts `tools=[...]` parameter
- `agents/toolkit.py` — `Toolkit` base class for grouping related tools
- `run/` — `RunContext` injected into tools at execution time
