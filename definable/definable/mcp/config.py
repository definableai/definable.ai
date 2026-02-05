"""MCP configuration dataclasses.

Configuration for MCP servers and client settings.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union


@dataclass
class MCPServerConfig:
  """Configuration for a single MCP server connection.

  Supports both stdio (subprocess) and SSE (HTTP) transports.

  Example (stdio):
      MCPServerConfig(
          name="filesystem",
          command="npx",
          args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      )

  Example (SSE):
      MCPServerConfig(
          name="web",
          transport="sse",
          url="http://localhost:3000/sse",
      )
  """

  # Server identifier (must be unique across all servers)
  name: str

  # Transport type
  # - "stdio": Subprocess-based (command + args)
  # - "sse": HTTP with Server-Sent Events (legacy)
  # - "http": Streamable HTTP (POST with SSE responses, newer MCP spec)
  transport: Literal["stdio", "sse", "http"] = "stdio"

  # For stdio transport: command and arguments
  command: Optional[str] = None
  args: Optional[List[str]] = None
  env: Optional[Dict[str, str]] = None
  cwd: Optional[str] = None

  # For SSE transport: server URL
  url: Optional[str] = None
  headers: Optional[Dict[str, str]] = None

  # Timeouts
  connect_timeout: float = 30.0
  request_timeout: float = 60.0

  # Tool filtering (whitelist/blacklist)
  allowed_tools: Optional[List[str]] = None
  blocked_tools: Optional[List[str]] = None

  # Auto-reconnection settings
  reconnect_on_failure: bool = True
  max_reconnect_attempts: int = 3
  reconnect_delay: float = 1.0  # Seconds between reconnection attempts

  def __post_init__(self) -> None:
    """Validate configuration after initialization."""
    if self.transport == "stdio" and not self.command:
      raise ValueError(f"MCPServerConfig '{self.name}': stdio transport requires 'command'")
    if self.transport == "sse" and not self.url:
      raise ValueError(f"MCPServerConfig '{self.name}': sse transport requires 'url'")
    if self.transport == "http" and not self.url:
      raise ValueError(f"MCPServerConfig '{self.name}': http transport requires 'url'")

  def is_tool_allowed(self, tool_name: str) -> bool:
    """Check if a tool is allowed by the filter configuration.

    Args:
        tool_name: Name of the tool to check.

    Returns:
        True if tool is allowed, False if blocked.
    """
    # If whitelist is set, tool must be in it
    if self.allowed_tools is not None:
      if tool_name not in self.allowed_tools:
        return False

    # If blacklist is set, tool must not be in it
    if self.blocked_tools is not None:
      if tool_name in self.blocked_tools:
        return False

    return True

  def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary representation."""
    result: Dict[str, Any] = {
      "name": self.name,
      "transport": self.transport,
    }

    if self.transport == "stdio":
      result["command"] = self.command
      if self.args:
        result["args"] = self.args
      if self.env:
        result["env"] = self.env
      if self.cwd:
        result["cwd"] = self.cwd
    elif self.transport == "sse":
      result["url"] = self.url
      if self.headers:
        result["headers"] = self.headers

    if self.connect_timeout != 30.0:
      result["connect_timeout"] = self.connect_timeout
    if self.request_timeout != 60.0:
      result["request_timeout"] = self.request_timeout
    if self.allowed_tools:
      result["allowed_tools"] = self.allowed_tools
    if self.blocked_tools:
      result["blocked_tools"] = self.blocked_tools

    return result

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "MCPServerConfig":
    """Create from dictionary representation."""
    return cls(
      name=data["name"],
      transport=data.get("transport", "stdio"),
      command=data.get("command"),
      args=data.get("args"),
      env=data.get("env"),
      cwd=data.get("cwd"),
      url=data.get("url"),
      headers=data.get("headers"),
      connect_timeout=data.get("connect_timeout", 30.0),
      request_timeout=data.get("request_timeout", 60.0),
      allowed_tools=data.get("allowed_tools"),
      blocked_tools=data.get("blocked_tools"),
      reconnect_on_failure=data.get("reconnect_on_failure", True),
      max_reconnect_attempts=data.get("max_reconnect_attempts", 3),
      reconnect_delay=data.get("reconnect_delay", 1.0),
    )


@dataclass
class MCPConfig:
  """Configuration for the MCP client.

  Manages configuration for multiple MCP server connections.

  Example:
      config = MCPConfig(
          servers=[
              MCPServerConfig(name="fs", command="mcp-server-filesystem", args=["/tmp"]),
              MCPServerConfig(name="web", transport="sse", url="http://localhost:3000"),
          ],
          auto_connect=True,
      )
  """

  # List of server configurations
  servers: List[MCPServerConfig] = field(default_factory=list)

  # Connect to all servers automatically
  auto_connect: bool = True

  # Global reconnection setting (can be overridden per-server)
  reconnect_on_failure: bool = True

  def __post_init__(self) -> None:
    """Validate configuration after initialization."""
    # Check for duplicate server names
    names = [s.name for s in self.servers]
    if len(names) != len(set(names)):
      duplicates = [n for n in names if names.count(n) > 1]
      raise ValueError(f"Duplicate server names in MCPConfig: {set(duplicates)}")

  def get_server(self, name: str) -> Optional[MCPServerConfig]:
    """Get server configuration by name.

    Args:
        name: Server name.

    Returns:
        Server configuration if found, None otherwise.
    """
    for server in self.servers:
      if server.name == name:
        return server
    return None

  def add_server(self, server: MCPServerConfig) -> None:
    """Add a server configuration.

    Args:
        server: Server configuration to add.

    Raises:
        ValueError: If server with same name already exists.
    """
    if self.get_server(server.name):
      raise ValueError(f"Server '{server.name}' already exists in configuration")
    self.servers.append(server)

  def remove_server(self, name: str) -> bool:
    """Remove a server configuration by name.

    Args:
        name: Server name to remove.

    Returns:
        True if server was removed, False if not found.
    """
    for i, server in enumerate(self.servers):
      if server.name == name:
        self.servers.pop(i)
        return True
    return False

  def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary representation."""
    return {
      "servers": [s.to_dict() for s in self.servers],
      "auto_connect": self.auto_connect,
      "reconnect_on_failure": self.reconnect_on_failure,
    }

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "MCPConfig":
    """Create from dictionary representation."""
    servers = [MCPServerConfig.from_dict(s) for s in data.get("servers", [])]
    return cls(
      servers=servers,
      auto_connect=data.get("auto_connect", True),
      reconnect_on_failure=data.get("reconnect_on_failure", True),
    )

  @classmethod
  def from_file(cls, path: Union[str, Path]) -> "MCPConfig":
    """Load configuration from a JSON file.

    Args:
        path: Path to JSON configuration file.

    Returns:
        MCPConfig instance.

    Raises:
        FileNotFoundError: If file doesn't exist.
        json.JSONDecodeError: If file isn't valid JSON.
        ValueError: If configuration is invalid.

    Example file format:
        {
            "servers": [
                {
                    "name": "filesystem",
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
                },
                {
                    "name": "web",
                    "transport": "sse",
                    "url": "http://localhost:3000/sse"
                }
            ],
            "auto_connect": true,
            "reconnect_on_failure": true
        }
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
      data = json.load(f)
    return cls.from_dict(data)

  def to_file(self, path: Union[str, Path]) -> None:
    """Save configuration to a JSON file.

    Args:
        path: Path to write configuration file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
      json.dump(self.to_dict(), f, indent=2)
