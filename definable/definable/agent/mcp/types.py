"""MCP protocol type definitions.

Pydantic models for JSON-RPC 2.0 and Model Context Protocol types.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# =============================================================================
# JSON-RPC 2.0 Types
# =============================================================================


class JSONRPCRequest(BaseModel):
  """JSON-RPC 2.0 request message."""

  jsonrpc: Literal["2.0"] = "2.0"
  method: str
  params: Optional[Dict[str, Any]] = None
  id: Optional[Union[str, int]] = None


class JSONRPCNotification(BaseModel):
  """JSON-RPC 2.0 notification (no id = no response expected)."""

  jsonrpc: Literal["2.0"] = "2.0"
  method: str
  params: Optional[Dict[str, Any]] = None


class JSONRPCErrorData(BaseModel):
  """JSON-RPC 2.0 error object."""

  code: int
  message: str
  data: Optional[Any] = None


class JSONRPCResponse(BaseModel):
  """JSON-RPC 2.0 response message."""

  jsonrpc: Literal["2.0"] = "2.0"
  id: Optional[Union[str, int]] = None
  result: Optional[Any] = None
  error: Optional[JSONRPCErrorData] = None


# Standard JSON-RPC 2.0 error codes
class JSONRPCErrorCode(int, Enum):
  """Standard JSON-RPC 2.0 error codes."""

  PARSE_ERROR = -32700
  INVALID_REQUEST = -32600
  METHOD_NOT_FOUND = -32601
  INVALID_PARAMS = -32602
  INTERNAL_ERROR = -32603


# =============================================================================
# MCP Protocol Types
# =============================================================================


class MCPImplementation(BaseModel):
  """MCP implementation info."""

  name: str
  version: str


class MCPCapabilities(BaseModel):
  """MCP server capabilities."""

  tools: Optional[Dict[str, Any]] = None
  resources: Optional[Dict[str, Any]] = None
  prompts: Optional[Dict[str, Any]] = None
  logging: Optional[Dict[str, Any]] = None
  experimental: Optional[Dict[str, Any]] = None


class MCPServerInfo(BaseModel):
  """MCP server info returned from initialize."""

  protocolVersion: str
  capabilities: MCPCapabilities
  serverInfo: MCPImplementation


class MCPClientInfo(BaseModel):
  """MCP client info sent during initialize."""

  protocolVersion: str = "2024-11-05"
  capabilities: Dict[str, Any] = Field(default_factory=dict)
  clientInfo: MCPImplementation = Field(default_factory=lambda: MCPImplementation(name="definable", version="0.1.0"))


# =============================================================================
# Tool Types
# =============================================================================


class MCPToolInputSchema(BaseModel):
  """JSON Schema for tool input parameters."""

  type: Literal["object"] = "object"
  properties: Dict[str, Any] = Field(default_factory=dict)
  required: Optional[List[str]] = None
  additionalProperties: Optional[bool] = None


class MCPToolDefinition(BaseModel):
  """MCP tool definition returned from tools/list."""

  name: str
  description: Optional[str] = None
  inputSchema: MCPToolInputSchema


class MCPTextContent(BaseModel):
  """Text content in tool results."""

  type: Literal["text"] = "text"
  text: str


class MCPImageContent(BaseModel):
  """Image content in tool results."""

  type: Literal["image"] = "image"
  data: str  # Base64 encoded
  mimeType: str


class MCPEmbeddedResource(BaseModel):
  """Embedded resource content in tool results."""

  type: Literal["resource"] = "resource"
  resource: "MCPResourceContent"


MCPToolContent = Union[MCPTextContent, MCPImageContent, MCPEmbeddedResource]


class MCPToolCallResult(BaseModel):
  """Result from tools/call."""

  content: List[MCPToolContent]
  isError: Optional[bool] = None


# =============================================================================
# Resource Types
# =============================================================================


class MCPResource(BaseModel):
  """MCP resource definition returned from resources/list."""

  uri: str
  name: str
  description: Optional[str] = None
  mimeType: Optional[str] = None


class MCPResourceTemplate(BaseModel):
  """MCP resource template for parameterized resources."""

  uriTemplate: str
  name: str
  description: Optional[str] = None
  mimeType: Optional[str] = None


class MCPTextResourceContent(BaseModel):
  """Text content for a resource."""

  uri: str
  mimeType: Optional[str] = None
  text: str


class MCPBlobResourceContent(BaseModel):
  """Binary (blob) content for a resource."""

  uri: str
  mimeType: Optional[str] = None
  blob: str  # Base64 encoded


MCPResourceContent = Union[MCPTextResourceContent, MCPBlobResourceContent]


class MCPResourceReadResult(BaseModel):
  """Result from resources/read."""

  contents: List[MCPResourceContent]


class MCPResourceListResult(BaseModel):
  """Result from resources/list."""

  resources: List[MCPResource]
  nextCursor: Optional[str] = None


class MCPResourceTemplateListResult(BaseModel):
  """Result from resources/templates/list."""

  resourceTemplates: List[MCPResourceTemplate]
  nextCursor: Optional[str] = None


# =============================================================================
# Prompt Types
# =============================================================================


class MCPPromptArgument(BaseModel):
  """Argument definition for a prompt."""

  name: str
  description: Optional[str] = None
  required: Optional[bool] = None


class MCPPromptDefinition(BaseModel):
  """MCP prompt definition returned from prompts/list."""

  name: str
  description: Optional[str] = None
  arguments: Optional[List[MCPPromptArgument]] = None


class MCPPromptListResult(BaseModel):
  """Result from prompts/list."""

  prompts: List[MCPPromptDefinition]
  nextCursor: Optional[str] = None


class MCPPromptMessageRole(str, Enum):
  """Role for prompt messages."""

  USER = "user"
  ASSISTANT = "assistant"


class MCPPromptMessage(BaseModel):
  """Message in a prompt result."""

  role: MCPPromptMessageRole
  content: MCPTextContent


class MCPPromptGetResult(BaseModel):
  """Result from prompts/get."""

  description: Optional[str] = None
  messages: List[MCPPromptMessage]


# =============================================================================
# Tool List Result
# =============================================================================


class MCPToolListResult(BaseModel):
  """Result from tools/list."""

  tools: List[MCPToolDefinition]
  nextCursor: Optional[str] = None


# Update forward references
MCPEmbeddedResource.model_rebuild()
