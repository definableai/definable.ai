"""Toolkit base class for aggregating tools with shared dependencies."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
  from definable.tool.function import Function


class Toolkit:
  """
  Base class for tool collections with shared dependencies.

  Toolkits allow you to group related tools together with shared
  configuration and dependencies. Tools can be discovered automatically
  from @tool decorated methods or explicitly defined.

  Example:
      class WebSearchToolkit(Toolkit):
          def __init__(self, api_key: str):
              super().__init__(dependencies={"api_key": api_key})

          @property
          def tools(self) -> List[Function]:
              return [self.search, self.fetch_page]

          @tool
          def search(self, query: str) -> str:
              '''Search the web for a query.'''
              # Implementation using self._dependencies["api_key"]
              pass

          @tool
          def fetch_page(self, url: str) -> str:
              '''Fetch content from a URL.'''
              pass

  Usage:
      agent = Agent(
          model=my_model,
          toolkits=[WebSearchToolkit(api_key="...")],
      )
  """

  def __init__(self, dependencies: Optional[Dict[str, Any]] = None):
    """
    Initialize the toolkit.

    Args:
        dependencies: Shared dependencies to inject into all tools.
            These will be available as self._dependencies in tool methods
            and will be merged with agent-level dependencies.
    """
    self._dependencies = dependencies or {}

  @property
  def tools(self) -> List["Function"]:
    """
    Return the list of tools provided by this toolkit.

    Override this property to explicitly define tools, or rely on
    auto-discovery which finds all Function-typed attributes.

    Returns:
        List of Function objects representing the toolkit's tools.
    """
    from definable.tool.function import Function

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
    """
    Get the shared dependencies for this toolkit.

    Returns:
        Dictionary of dependencies to inject into tools.
    """
    return self._dependencies

  @property
  def name(self) -> str:
    """
    Get the toolkit name (defaults to class name).

    Returns:
        Name identifier for this toolkit.
    """
    return self.__class__.__name__

  def __repr__(self) -> str:
    tool_count = len(self.tools)
    return f"{self.name}(tools={tool_count})"
