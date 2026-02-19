"""MCP Resources Example.

This example demonstrates accessing MCP resources directly,
bypassing the agent for direct resource operations.

Prerequisites:
    npm install -g @modelcontextprotocol/server-filesystem

Usage:
    python 03_resources.py
"""

import asyncio

from definable.mcp import MCPClient, MCPConfig, MCPResourceProvider, MCPServerConfig


async def main() -> None:
  # Configure MCP server
  config = MCPConfig(
    servers=[
      MCPServerConfig(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      ),
    ]
  )

  # Use client directly
  async with MCPClient(config) as client:
    print("Connected to MCP servers")

    # Create resource provider
    resources = MCPResourceProvider(client)

    # List available resources
    print("\n--- Available Resources ---")
    all_resources = await resources.list_resources()
    for server_name, resource_list in all_resources.items():
      print(f"\n[{server_name}]")
      if resource_list:
        for resource in resource_list[:10]:  # Limit to first 10
          print(f"  - {resource.uri}")
          if resource.description:
            print(f"    {resource.description}")
      else:
        print("  (no resources available)")

    # Create a test file
    print("\n--- Creating test file ---")
    test_content = "Hello from MCP resources example!"
    test_file = "/tmp/mcp_test_file.txt"

    # Use tools to write file
    result = await client.call_tool(
      "filesystem",
      "write_file",
      {"path": test_file, "content": test_content},
    )
    print(f"Write result: {result.content[0].text if result.content else 'OK'}")  # type: ignore[union-attr]

    # Read it back as resource
    print("\n--- Reading as resource ---")
    try:
      content = await resources.read_text("filesystem", f"file://{test_file}")
      print(f"Content: {content}")
    except Exception as e:
      # Some servers may not expose files as resources
      print(f"Could not read as resource: {e}")
      # Fall back to using tool
      read_result = await client.call_tool(
        "filesystem",
        "read_file",
        {"path": test_file},
      )
      print(f"Content (via tool): {read_result.content[0].text if read_result.content else 'N/A'}")  # type: ignore[union-attr]


if __name__ == "__main__":
  asyncio.run(main())
