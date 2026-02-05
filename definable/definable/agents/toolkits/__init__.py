"""Agent toolkits for common functionality."""

from definable.agents.toolkits.knowledge import KnowledgeToolkit


# Lazy import to avoid circular dependency
# MCPToolkit imports from agents.toolkit, but agents imports from toolkits
def __getattr__(name: str):
  if name == "MCPToolkit":
    from definable.mcp.toolkit import MCPToolkit

    return MCPToolkit
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Define MCPToolkit for static analysis (actual import is lazy)
MCPToolkit: type

__all__ = ["KnowledgeToolkit", "MCPToolkit"]
