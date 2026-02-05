"""MCP-specific exceptions.

Custom exceptions for MCP protocol errors, connection issues, and timeouts.
"""

from typing import Optional

from definable.exceptions import DefinableError


class MCPError(DefinableError):
  """Base exception for MCP-related errors."""

  def __init__(
    self,
    message: str,
    status_code: int = 500,
    server_name: Optional[str] = None,
  ):
    super().__init__(message, status_code)
    self.server_name = server_name
    self.type = "mcp_error"
    self.error_id = "mcp_error"


class MCPConnectionError(MCPError):
  """Raised when connection to an MCP server fails.

  This can happen due to:
  - Server process failed to start
  - Network connection refused
  - Server URL unreachable
  - TLS/SSL errors
  """

  def __init__(
    self,
    message: str,
    server_name: Optional[str] = None,
    original_error: Optional[Exception] = None,
  ):
    super().__init__(message, status_code=503, server_name=server_name)
    self.original_error = original_error
    self.type = "mcp_connection_error"
    self.error_id = "mcp_connection_error"


class MCPTimeoutError(MCPError):
  """Raised when an MCP operation times out.

  This can happen during:
  - Initial connection handshake
  - Tool execution taking too long
  - Resource read operations
  """

  def __init__(
    self,
    message: str,
    server_name: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
  ):
    super().__init__(message, status_code=504, server_name=server_name)
    self.timeout_seconds = timeout_seconds
    self.type = "mcp_timeout_error"
    self.error_id = "mcp_timeout_error"


class MCPProtocolError(MCPError):
  """Raised when the MCP protocol is violated.

  This can happen due to:
  - Invalid JSON-RPC messages
  - Unsupported protocol version
  - Missing required fields
  - Server sent unexpected response
  """

  def __init__(
    self,
    message: str,
    server_name: Optional[str] = None,
    error_code: Optional[int] = None,
    error_data: Optional[dict] = None,
  ):
    super().__init__(message, status_code=502, server_name=server_name)
    self.error_code = error_code
    self.error_data = error_data
    self.type = "mcp_protocol_error"
    self.error_id = "mcp_protocol_error"


class MCPToolNotFoundError(MCPError):
  """Raised when a requested tool is not found on the server.

  This can happen when:
  - Tool name is misspelled
  - Tool was removed from server
  - Tool is on a different server
  """

  def __init__(
    self,
    message: str,
    tool_name: str,
    server_name: Optional[str] = None,
    available_tools: Optional[list] = None,
  ):
    super().__init__(message, status_code=404, server_name=server_name)
    self.tool_name = tool_name
    self.available_tools = available_tools
    self.type = "mcp_tool_not_found_error"
    self.error_id = "mcp_tool_not_found_error"


class MCPServerNotFoundError(MCPError):
  """Raised when a requested MCP server is not found.

  This can happen when:
  - Server name is misspelled
  - Server was not configured
  - Server failed to connect
  """

  def __init__(
    self,
    message: str,
    server_name: str,
    available_servers: Optional[list] = None,
  ):
    super().__init__(message, status_code=404, server_name=server_name)
    self.available_servers = available_servers
    self.type = "mcp_server_not_found_error"
    self.error_id = "mcp_server_not_found_error"


class MCPResourceNotFoundError(MCPError):
  """Raised when a requested resource is not found.

  This can happen when:
  - Resource URI is invalid
  - Resource was deleted
  - Resource is on a different server
  """

  def __init__(
    self,
    message: str,
    resource_uri: str,
    server_name: Optional[str] = None,
  ):
    super().__init__(message, status_code=404, server_name=server_name)
    self.resource_uri = resource_uri
    self.type = "mcp_resource_not_found_error"
    self.error_id = "mcp_resource_not_found_error"


class MCPPromptNotFoundError(MCPError):
  """Raised when a requested prompt is not found.

  This can happen when:
  - Prompt name is misspelled
  - Prompt was removed from server
  - Prompt is on a different server
  """

  def __init__(
    self,
    message: str,
    prompt_name: str,
    server_name: Optional[str] = None,
  ):
    super().__init__(message, status_code=404, server_name=server_name)
    self.prompt_name = prompt_name
    self.type = "mcp_prompt_not_found_error"
    self.error_id = "mcp_prompt_not_found_error"
