"""ToolServer — runs a local MCP stdio server for Definable tools.

Generates a standalone Python script using FastMCP that serves tool definitions.
The CLI connects to this server via --mcp-config. Tool execution requests are
forwarded back to this process over a Unix domain socket for actual dispatch.

Lifecycle:
  1. ``start()`` — write temp script, start socket listener
  2. Agent passes ``get_mcp_config()`` to CLI via ``--mcp-config``
  3. CLI calls tools → FastMCP script → Unix socket → ``ToolBridge.execute()``
  4. ``stop()`` — clean up temp files and socket
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from typing import Optional

from definable.claude_code.bridge import ToolBridge
from definable.utils.log import log_debug, log_error


class ToolServer:
  """Manages a local MCP stdio server for Definable tools."""

  def __init__(self, bridge: ToolBridge):
    self._bridge = bridge
    self._socket_path: Optional[str] = None
    self._script_path: Optional[str] = None
    self._server_task: Optional[asyncio.Task] = None
    self._socket_server: Optional[asyncio.AbstractServer] = None

  @property
  def is_running(self) -> bool:
    return self._socket_server is not None and self._socket_server.is_serving()

  async def start(self) -> None:
    """Start the Unix socket listener and write the MCP server script."""
    if self.is_running:
      return

    # 1. Create temp socket path
    tmp_dir = tempfile.mkdtemp(prefix="definable_mcp_")
    self._socket_path = os.path.join(tmp_dir, "tools.sock")

    # 2. Start Unix socket server to receive tool calls from the MCP script
    self._socket_server = await asyncio.start_unix_server(self._handle_connection, path=self._socket_path)
    log_debug(f"Tool server listening on {self._socket_path}")

    # 3. Generate the MCP server script
    self._script_path = os.path.join(tmp_dir, "mcp_server.py")
    self._write_server_script()
    log_debug(f"MCP server script written to {self._script_path}")

  def _write_server_script(self) -> None:
    """Generate a standalone FastMCP server script."""
    tools_json = json.dumps([
      {
        "name": fn.name,
        "description": fn.description or fn.name,
        "parameters": fn.parameters,
      }
      for fn in self._bridge._tools.values()
    ])

    script = f'''#!/usr/bin/env python3
"""Auto-generated MCP server for Definable tools. Do not edit."""
import asyncio
import json
import socket
import sys

from mcp.server.fastmcp import FastMCP

SOCKET_PATH = {self._socket_path!r}
TOOLS = {tools_json}

mcp = FastMCP("definable")


async def call_bridge(tool_name: str, args: dict) -> str:
  """Send tool call to the parent process via Unix socket."""
  reader, writer = await asyncio.open_unix_connection(SOCKET_PATH)
  request = json.dumps({{"tool": tool_name, "args": args}})
  writer.write(request.encode() + b"\\n")
  await writer.drain()
  response = await reader.readline()
  writer.close()
  await writer.wait_closed()
  result = json.loads(response.decode())
  if result.get("error"):
    raise Exception(result["error"])
  return result.get("result", "")


# Dynamically register tools
for _tool_def in TOOLS:
  _name = _tool_def["name"]
  _desc = _tool_def["description"]
  _params = _tool_def.get("parameters", {{}})
  _props = _params.get("properties", {{}})
  _required = _params.get("required", [])

  # Build parameter annotations for FastMCP
  _param_names = list(_props.keys())

  # Create the tool function dynamically
  exec(f"""
async def {{_name}}(**kwargs) -> str:
  \\"\\"\\"{{_desc}}\\"\\"\\"
  return await call_bridge({{_name!r}}, kwargs)
mcp.tool()({{_name}})
""", {{"call_bridge": call_bridge, "mcp": mcp, "__builtins__": __builtins__}})


if __name__ == "__main__":
  mcp.run(transport="stdio")
'''

    with open(self._script_path, "w") as f:  # type: ignore[arg-type]
      f.write(script)

  async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Handle a tool call from the MCP server script."""
    try:
      data = await asyncio.wait_for(reader.readline(), timeout=30.0)
      if not data:
        return

      request = json.loads(data.decode())
      tool_name = request.get("tool", "")
      args = request.get("args", {})

      log_debug(f"Tool server: executing {tool_name}({args})")
      result = await self._bridge.execute(tool_name, args)

      # Extract text from MCP result format
      content = result.get("content", [])
      text = ""
      for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
          text = block.get("text", "")
          break

      is_error = result.get("isError", False)
      response = {"result": text} if not is_error else {"error": text}
      writer.write(json.dumps(response).encode() + b"\n")
      await writer.drain()

    except Exception as exc:
      log_error(f"Tool server error: {exc}")
      try:
        writer.write(json.dumps({"error": str(exc)}).encode() + b"\n")
        await writer.drain()
      except Exception:
        pass
    finally:
      writer.close()
      await writer.wait_closed()

  def get_mcp_config_path(self) -> Optional[str]:
    """Get the --mcp-config JSON string for the CLI."""
    if not self._script_path or not self._socket_path:
      return None

    python_path = sys.executable
    config = {
      "mcpServers": {
        "definable": {
          "command": python_path,
          "args": [self._script_path],
          "type": "stdio",
        }
      }
    }
    return json.dumps(config)

  async def stop(self) -> None:
    """Stop the socket server and clean up temp files."""
    if self._socket_server:
      self._socket_server.close()
      await self._socket_server.wait_closed()
      self._socket_server = None

    # Clean up temp files
    import contextlib

    for path in [self._script_path, self._socket_path]:
      if path and os.path.exists(path):
        with contextlib.suppress(OSError):
          os.unlink(path)

    # Clean up temp directory
    if self._socket_path:
      tmp_dir = os.path.dirname(self._socket_path)
      with contextlib.suppress(OSError):
        os.rmdir(tmp_dir)

    self._script_path = None
    self._socket_path = None
    log_debug("Tool server stopped")
