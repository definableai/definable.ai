"""Toolkit base class + AsyncToolkit lifecycle protocol.

`Toolkit` aggregates tools with shared dependencies. `AsyncToolkit` is the
protocol any toolkit needing async setup/teardown (MCP, browsers, DB pools)
must satisfy — the Agent calls `aopen()` / `aclose()` on the way in and out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
  from definable.agent.toolkit.function import Function


class Toolkit:
  """Base class for tool collections with shared dependencies.

  Tools can be discovered automatically from `@tool`-decorated attributes
  or explicitly defined by overriding the `tools` property.

  Example::

      class WebSearchToolkit(Toolkit):
        def __init__(self, api_key: str):
          super().__init__(dependencies={"api_key": api_key})

        @tool
        def search(self, query: str) -> str:
          '''Search the web for a query.'''
          ...

  Usage::

      agent = Agent(model=my_model, toolkits=[WebSearchToolkit(api_key="...")])
  """

  def __init__(self, dependencies: Optional[Dict[str, Any]] = None):
    self._dependencies = dependencies or {}

  @property
  def tools(self) -> List["Function"]:
    """Auto-discovered Function-typed attributes. Override to define explicitly."""
    from definable.agent.toolkit.function import Function

    discovered: List[Function] = []
    for name in dir(self):
      if name.startswith("_"):
        continue
      try:
        attr = getattr(self, name)
        if isinstance(attr, Function):
          discovered.append(attr)
      except Exception:
        # Skip attributes that raise on access
        continue
    return discovered

  @property
  def dependencies(self) -> Dict[str, Any]:
    return self._dependencies

  @property
  def name(self) -> str:
    """Toolkit name — defaults to the class name."""
    return self.__class__.__name__

  def __repr__(self) -> str:
    tool_count = len(self.tools)
    return f"{self.name}(tools={tool_count})"


@runtime_checkable
class AsyncToolkit(Protocol):
  """Protocol for toolkits that need async setup/teardown.

  Implementations: `MCPToolkit`, plus user toolkits that hold pooled clients.

  The Agent calls `aopen()` on enter (async with agent / before first arun)
  and `aclose()` on exit. Idempotency is the implementation's responsibility.
  """

  @property
  def tools(self) -> list: ...

  async def aopen(self) -> None: ...

  async def aclose(self) -> None: ...
