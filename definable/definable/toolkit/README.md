# Toolkit

> Base class for grouping related tools with shared dependencies.

Toolkits aggregate multiple tools that share configuration or resources. The base `Toolkit` class provides auto-discovery of `Function` attributes and a dependency injection pattern.

## Quick Start

```python
from definable.toolkit import Toolkit
from definable.tool.decorator import tool
from definable.tool.function import Function


class WeatherToolkit(Toolkit):
  def __init__(self, api_key: str):
    super().__init__(dependencies={"api_key": api_key})
    self._api_key = api_key

  @property
  def tools(self) -> list[Function]:
    @tool
    def get_weather(city: str) -> str:
      """Get current weather for a city."""
      return f"Weather in {city}: Sunny, 72F (key={self._api_key[:4]}...)"

    @tool
    def get_forecast(city: str, days: int = 3) -> str:
      """Get weather forecast for a city."""
      return f"{days}-day forecast for {city}: Sunny"

    return [get_weather, get_forecast]


# Use with Agent
from definable.agent import Agent

toolkit = WeatherToolkit(api_key="my-key")
agent = Agent(model="openai/gpt-4o-mini", toolkits=[toolkit])
```

## API Reference

### Toolkit

```python
from definable.toolkit import Toolkit


class Toolkit:
  def __init__(self, dependencies=None):
    """
    Args:
        dependencies: Dict of shared dependencies injected into tools.
    """

  @property
  def tools(self) -> list[Function]:
    """Override to define tools. Default: auto-discovers Function attributes."""

  @property
  def dependencies(self) -> dict:
    """Get shared dependencies dict."""

  @property
  def name(self) -> str:
    """Toolkit name (defaults to class name)."""
```

## Built-in Toolkits

| Toolkit | Module | Description |
|---------|--------|-------------|
| `BrowserToolkit` | `definable.browser` | 50+ browser automation tools via Playwright |
| `MCPToolkit` | `definable.mcp` | Tools from MCP servers |

## Async Lifecycle Toolkits

Toolkits with external resources (browser, MCP) follow the async lifecycle pattern:

```python
# Context manager
async with BrowserToolkit() as toolkit:
  agent = Agent(model="openai/gpt-4o", toolkits=[toolkit])
  await agent.arun("...")

# Manual lifecycle
toolkit = MCPToolkit(config=mcp_config)
await toolkit.initialize()
agent = Agent(model="openai/gpt-4o", toolkits=[toolkit])
await agent.arun("...")
await toolkit.shutdown()
```

## Related Modules

- **[Browser](../browser/README.md)** — BrowserToolkit implementation
- **[MCP](../mcp/README.md)** — MCPToolkit implementation
- **[Tool](../tool/README.md)** — `@tool` decorator and `Function` class
- **[Agent](../agent/README.md)** — Toolkits plug into Agent via `toolkits=`
