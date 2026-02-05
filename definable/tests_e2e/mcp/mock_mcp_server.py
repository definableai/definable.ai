#!/usr/bin/env python3
"""Mock MCP Server for E2E testing.

A minimal MCP server implementation that communicates via stdio using
JSON-RPC 2.0. This server is designed for testing MCP client functionality.

Features:
- Configurable tools, resources, and prompts
- Can simulate errors, timeouts, and malformed responses
- Runs as a subprocess (stdio transport)

Usage:
    python mock_mcp_server.py [options]

Options can be passed via environment variables:
    MCP_TOOLS: JSON list of tool definitions
    MCP_RESOURCES: JSON list of resource definitions
    MCP_PROMPTS: JSON list of prompt definitions
    MCP_SLOW_START: Delay in seconds before server starts
    MCP_RESPONSE_DELAY: Delay in seconds before responding
    MCP_FAIL_INITIALIZE: If set, fail the initialize handshake
    MCP_MALFORMED_JSON: If set, send malformed JSON responses
"""

import asyncio
import base64
import json
import os
import sys
from typing import Any, Dict, List, Optional


class MockMCPServer:
  """Mock MCP server implementation."""

  # Protocol version
  PROTOCOL_VERSION = "2024-11-05"

  def __init__(self) -> None:
    """Initialize the mock server."""
    self._request_id = 0
    self._initialized = False

    # Default tools
    self._tools: List[Dict[str, Any]] = [
      {
        "name": "echo",
        "description": "Returns the input message",
        "inputSchema": {
          "type": "object",
          "properties": {"message": {"type": "string", "description": "Message to echo"}},
          "required": ["message"],
        },
      },
      {
        "name": "add_numbers",
        "description": "Adds two numbers",
        "inputSchema": {
          "type": "object",
          "properties": {
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"},
          },
          "required": ["a", "b"],
        },
      },
      {
        "name": "slow_tool",
        "description": "A tool that takes time to respond",
        "inputSchema": {
          "type": "object",
          "properties": {"delay": {"type": "number", "description": "Delay in seconds"}},
          "required": ["delay"],
        },
      },
      {
        "name": "error_tool",
        "description": "A tool that always returns an error",
        "inputSchema": {
          "type": "object",
          "properties": {"message": {"type": "string", "description": "Error message"}},
          "required": [],
        },
      },
      {
        "name": "multi_content",
        "description": "Returns multiple content types",
        "inputSchema": {
          "type": "object",
          "properties": {},
          "required": [],
        },
      },
    ]

    # Default resources
    self._resources: List[Dict[str, Any]] = [
      {
        "uri": "file:///test/config.json",
        "name": "Test Config",
        "description": "A test configuration file",
        "mimeType": "application/json",
      },
      {
        "uri": "file:///test/data.txt",
        "name": "Test Data",
        "description": "A test data file",
        "mimeType": "text/plain",
      },
      {
        "uri": "file:///test/image.png",
        "name": "Test Image",
        "description": "A test image",
        "mimeType": "image/png",
      },
    ]

    # Default prompts
    self._prompts: List[Dict[str, Any]] = [
      {
        "name": "greeting",
        "description": "A greeting prompt",
        "arguments": [
          {"name": "name", "description": "Name to greet", "required": True},
        ],
      },
      {
        "name": "code_review",
        "description": "A code review prompt",
        "arguments": [
          {"name": "language", "description": "Programming language", "required": True},
          {"name": "style", "description": "Review style", "required": False},
        ],
      },
    ]

    # Resource content storage
    self._resource_content: Dict[str, Dict[str, Any]] = {
      "file:///test/config.json": {
        "text": '{"key": "value", "number": 42}',
        "mimeType": "application/json",
      },
      "file:///test/data.txt": {
        "text": "This is test data content.\nLine 2.",
        "mimeType": "text/plain",
      },
      "file:///test/image.png": {
        "blob": base64.b64encode(b"\x89PNG\r\n\x1a\n").decode(),
        "mimeType": "image/png",
      },
    }

    # Load custom configuration from environment
    self._load_config()

    # Configuration flags
    self._slow_start = float(os.environ.get("MCP_SLOW_START", "0"))
    self._response_delay = float(os.environ.get("MCP_RESPONSE_DELAY", "0"))
    self._fail_initialize = os.environ.get("MCP_FAIL_INITIALIZE", "").lower() == "true"
    self._malformed_json = os.environ.get("MCP_MALFORMED_JSON", "").lower() == "true"

  def _load_config(self) -> None:
    """Load configuration from environment variables."""
    import contextlib

    if os.environ.get("MCP_TOOLS"):
      with contextlib.suppress(json.JSONDecodeError):
        self._tools = json.loads(os.environ["MCP_TOOLS"])

    if os.environ.get("MCP_RESOURCES"):
      with contextlib.suppress(json.JSONDecodeError):
        self._resources = json.loads(os.environ["MCP_RESOURCES"])

    if os.environ.get("MCP_PROMPTS"):
      with contextlib.suppress(json.JSONDecodeError):
        self._prompts = json.loads(os.environ["MCP_PROMPTS"])

  async def handle_request(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Handle an incoming JSON-RPC request.

    Args:
        data: Parsed JSON-RPC message.

    Returns:
        Response dictionary, or None for notifications.
    """
    method = data.get("method", "")
    params = data.get("params", {})
    request_id = data.get("id")

    # Notifications don't get responses
    if request_id is None:
      await self._handle_notification(method, params)
      return None

    # Add response delay if configured
    if self._response_delay > 0:
      await asyncio.sleep(self._response_delay)

    # Route to handler
    try:
      if method == "initialize":
        result = await self._handle_initialize(params)
      elif method == "tools/list":
        result = await self._handle_tools_list(params)
      elif method == "tools/call":
        result = await self._handle_tools_call(params)
      elif method == "resources/list":
        result = await self._handle_resources_list(params)
      elif method == "resources/read":
        result = await self._handle_resources_read(params)
      elif method == "prompts/list":
        result = await self._handle_prompts_list(params)
      elif method == "prompts/get":
        result = await self._handle_prompts_get(params)
      else:
        return self._error_response(-32601, f"Method not found: {method}", request_id)

      return self._success_response(result, request_id)

    except Exception as e:
      return self._error_response(-32603, str(e), request_id)

  async def _handle_notification(self, method: str, params: Dict[str, Any]) -> None:
    """Handle a notification (no response expected)."""
    if method == "notifications/initialized":
      self._initialized = True

  async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle initialize request."""
    if self._fail_initialize:
      raise Exception("Initialization failed (test mode)")

    return {
      "protocolVersion": self.PROTOCOL_VERSION,
      "capabilities": {
        "tools": {},
        "resources": {},
        "prompts": {},
      },
      "serverInfo": {
        "name": "mock-mcp-server",
        "version": "1.0.0",
      },
    }

  async def _handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle tools/list request."""
    return {"tools": self._tools}

  async def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle tools/call request."""
    tool_name = params.get("name", "")
    arguments = params.get("arguments", {})

    # Find the tool
    tool = None
    for t in self._tools:
      if t["name"] == tool_name:
        tool = t
        break

    if not tool:
      return {
        "content": [{"type": "text", "text": f"Tool not found: {tool_name}"}],
        "isError": True,
      }

    # Execute tool
    if tool_name == "echo":
      message = arguments.get("message", "")
      return {"content": [{"type": "text", "text": message}]}

    elif tool_name == "add_numbers":
      a = arguments.get("a", 0)
      b = arguments.get("b", 0)
      result = a + b
      return {"content": [{"type": "text", "text": str(result)}]}

    elif tool_name == "slow_tool":
      delay = arguments.get("delay", 1.0)
      await asyncio.sleep(delay)
      return {"content": [{"type": "text", "text": f"Waited {delay} seconds"}]}

    elif tool_name == "error_tool":
      message = arguments.get("message", "Tool error occurred")
      return {
        "content": [{"type": "text", "text": message}],
        "isError": True,
      }

    elif tool_name == "multi_content":
      return {
        "content": [
          {"type": "text", "text": "Text content"},
          {
            "type": "image",
            "data": base64.b64encode(b"fake-image-data").decode(),
            "mimeType": "image/png",
          },
        ]
      }

    else:
      return {"content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}]}

  async def _handle_resources_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle resources/list request."""
    return {"resources": self._resources}

  async def _handle_resources_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle resources/read request."""
    uri = params.get("uri", "")

    if uri not in self._resource_content:
      raise Exception(f"Resource not found: {uri}")

    content = self._resource_content[uri]
    result_content = {"uri": uri}

    if "text" in content:
      result_content["text"] = content["text"]
    if "blob" in content:
      result_content["blob"] = content["blob"]
    if "mimeType" in content:
      result_content["mimeType"] = content["mimeType"]

    return {"contents": [result_content]}

  async def _handle_prompts_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle prompts/list request."""
    return {"prompts": self._prompts}

  async def _handle_prompts_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle prompts/get request."""
    prompt_name = params.get("name", "")
    arguments = params.get("arguments", {})

    # Find the prompt
    prompt = None
    for p in self._prompts:
      if p["name"] == prompt_name:
        prompt = p
        break

    if not prompt:
      raise Exception(f"Prompt not found: {prompt_name}")

    # Generate messages based on prompt
    if prompt_name == "greeting":
      name = arguments.get("name", "User")
      return {
        "description": "A greeting prompt",
        "messages": [
          {
            "role": "user",
            "content": {"type": "text", "text": f"Hello, {name}!"},
          }
        ],
      }

    elif prompt_name == "code_review":
      language = arguments.get("language", "python")
      style = arguments.get("style", "standard")
      return {
        "description": "Code review prompt",
        "messages": [
          {
            "role": "user",
            "content": {
              "type": "text",
              "text": f"Please review this {language} code using {style} style.",
            },
          }
        ],
      }

    return {
      "description": prompt.get("description", ""),
      "messages": [
        {
          "role": "user",
          "content": {"type": "text", "text": f"Prompt: {prompt_name}"},
        }
      ],
    }

  def _success_response(self, result: Any, request_id: Any) -> Dict[str, Any]:
    """Create a success response."""
    return {
      "jsonrpc": "2.0",
      "id": request_id,
      "result": result,
    }

  def _error_response(self, code: int, message: str, request_id: Any) -> Dict[str, Any]:
    """Create an error response."""
    return {
      "jsonrpc": "2.0",
      "id": request_id,
      "error": {
        "code": code,
        "message": message,
      },
    }

  async def run(self) -> None:
    """Run the server, reading from stdin and writing to stdout."""
    # Simulate slow startup if configured
    if self._slow_start > 0:
      await asyncio.sleep(self._slow_start)

    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

    writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(lambda: asyncio.streams.FlowControlMixin(), sys.stdout)
    writer = asyncio.StreamWriter(writer_transport, writer_protocol, None, asyncio.get_event_loop())

    while True:
      try:
        line = await reader.readline()
        if not line:
          break

        # Parse JSON
        try:
          data = json.loads(line.decode())
        except json.JSONDecodeError:
          # Send parse error
          error_response = self._error_response(-32700, "Parse error", None)
          await self._send_response(writer, error_response)
          continue

        # Handle request
        response = await self.handle_request(data)

        # Send response (if not a notification)
        if response is not None:
          if self._malformed_json:
            # Send malformed JSON for testing
            writer.write(b'{"jsonrpc": "2.0", invalid json}\n')
            await writer.drain()
          else:
            await self._send_response(writer, response)

      except Exception as e:
        # Log error to stderr
        sys.stderr.write(f"Server error: {e}\n")
        sys.stderr.flush()
        break

  async def _send_response(self, writer: asyncio.StreamWriter, response: Dict[str, Any]) -> None:
    """Send a response to stdout."""
    message = json.dumps(response) + "\n"
    writer.write(message.encode())
    await writer.drain()


async def main() -> None:
  """Main entry point."""
  server = MockMCPServer()
  await server.run()


if __name__ == "__main__":
  asyncio.run(main())
