"""DEPRECATED — use `definable.agent.mcp` instead.

Shim for backward compatibility during the harness-v2 migration. Will be
deleted in Phase 13.
"""

from definable.agent.mcp import (
  MCPClient,
  MCPConfig,
  MCPServerConfig,
  MCPToolkit,
)

__all__ = ["MCPClient", "MCPConfig", "MCPServerConfig", "MCPToolkit"]
