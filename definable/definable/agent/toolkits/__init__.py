"""Agent toolkits for common functionality."""

from definable.agent.toolkits.knowledge import KnowledgeToolkit

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.mcp.toolkit import MCPToolkit


# Lazy import to avoid circular dependency
# MCPToolkit imports from agents.toolkit, but agents imports from toolkits
def __getattr__(name: str):
  if name == "MCPToolkit":
    from definable.mcp.toolkit import MCPToolkit

    return MCPToolkit
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["KnowledgeToolkit", "MCPToolkit"]
