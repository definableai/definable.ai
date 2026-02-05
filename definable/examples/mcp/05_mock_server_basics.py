"""Basic MCP Mock Server Example.

This example demonstrates basic MCP functionality using the mock server.
No external services or API keys required - fully self-contained.

Features shown:
- MCPClient direct usage
- MCPToolkit usage
- Listing and calling tools

Run with: python 05_mock_server_basics.py
"""

import asyncio
import sys
from pathlib import Path

from definable.mcp import (
  MCPClient,
  MCPConfig,
  MCPServerConfig,
  MCPToolkit,
)

# Path to mock MCP server
MOCK_SERVER = Path(__file__).parent.parent.parent / "tests_e2e" / "mcp" / "mock_mcp_server.py"


def get_mock_config() -> MCPConfig:
  """Create configuration for the mock MCP server."""
  return MCPConfig(
    servers=[
      MCPServerConfig(
        name="mock",
        transport="stdio",
        command=sys.executable,
        args=[str(MOCK_SERVER)],
      ),
    ]
  )


async def demo_direct_client() -> None:
  """Demonstrate direct MCPClient usage."""
  print("=" * 70)
  print("Part 1: Direct MCPClient Usage")
  print("=" * 70)

  config = get_mock_config()
  client = MCPClient(config)

  async with client:
    # List all tools from all servers
    print("\n1. Listing all tools:")
    tools = await client.list_all_tools()
    for server_name, tool_list in tools.items():
      print(f"\n   Server: {server_name}")
      for tool in tool_list:
        print(f"   - {tool.name}: {tool.description}")

    # Call the echo tool
    print("\n2. Calling 'echo' tool:")
    result = await client.call_tool("mock", "echo", {"message": "Hello from MCP!"})
    # result.content is a list of content items
    text = result.content[0].text if result.content else ""
    print(f"   Result: {text}")

    # Call the add_numbers tool
    print("\n3. Calling 'add_numbers' tool:")
    result = await client.call_tool("mock", "add_numbers", {"a": 10, "b": 32})
    text = result.content[0].text if result.content else ""
    print(f"   10 + 32 = {text}")

    # Get server info
    print("\n4. Server connection info:")
    connection = client.get_connection("mock")
    if connection:
      print("   Server name: mock")
      print(f"   Connected: {connection.connected}")


async def demo_toolkit() -> None:
  """Demonstrate MCPToolkit usage."""
  print("\n" + "=" * 70)
  print("Part 2: MCPToolkit Usage")
  print("=" * 70)

  config = get_mock_config()
  toolkit = MCPToolkit(config)

  async with toolkit:
    # Show available tools
    print("\n1. Toolkit tools (agent-compatible):")
    for tool in toolkit.tools:
      print(f"   - {tool.name}: {tool.description}")

    # Tools are designed to be used with agents
    # For direct tool calls, use the MCPClient (shown in Part 1)
    print("\n2. Toolkit is designed for agent integration:")
    print("   Use Agent(toolkits=[toolkit]) to make tools available to agents")
    print("   For direct tool calls, use MCPClient.call_tool() instead")

    # Show tool count
    print(f"\n3. Total tools available: {len(toolkit.tools)}")


async def demo_resources() -> None:
  """Demonstrate listing resources."""
  print("\n" + "=" * 70)
  print("Part 3: MCP Resources")
  print("=" * 70)

  config = get_mock_config()
  client = MCPClient(config)

  async with client:
    # List resources
    print("\n1. Available resources:")
    resources = await client.list_all_resources()
    for server_name, resource_list in resources.items():
      print(f"\n   Server: {server_name}")
      for resource in resource_list:
        print(f"   - {resource.name} ({resource.uri})")
        print(f"     Type: {resource.mimeType}")


async def main() -> None:
  """Run all demonstrations."""
  print("MCP Mock Server Basics Example")
  print("No API keys required - uses local mock server")
  print()

  # Verify mock server exists
  if not MOCK_SERVER.exists():
    print(f"Error: Mock server not found at {MOCK_SERVER}")
    print("Make sure you're running from the repository root.")
    return

  await demo_direct_client()
  await demo_toolkit()
  await demo_resources()

  print("\n" + "=" * 70)
  print("Example completed successfully!")
  print("=" * 70)


if __name__ == "__main__":
  asyncio.run(main())
