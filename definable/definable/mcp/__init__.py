"""
Definable MCP - Model Context Protocol integration.

This module provides MCP client capabilities for connecting to
MCP servers and using their tools, resources, and prompts with agents.

Quick Start:
    from definable.mcp import MCPConfig, MCPServerConfig, MCPToolkit

    # Configure MCP servers
    config = MCPConfig(servers=[
        MCPServerConfig(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        ),
    ])

    # Use with agent
    toolkit = MCPToolkit(config)

    async def main():
        async with toolkit:
            agent = Agent(model=model, toolkits=[toolkit])
            output = await agent.arun("List files in /tmp")

    asyncio.run(main())

With Multiple Servers:
    config = MCPConfig(servers=[
        MCPServerConfig(name="fs", command="mcp-server-filesystem", args=["/tmp"]),
        MCPServerConfig(name="web", transport="sse", url="http://localhost:3000"),
    ])

    async with MCPToolkit(config) as toolkit:
        # Tools from both servers are available
        agent = Agent(model=model, toolkits=[toolkit])

Direct Client Access:
    from definable.mcp import MCPClient, MCPResourceProvider, MCPPromptProvider

    async with MCPClient(config) as client:
        # List all tools
        tools = await client.list_all_tools()

        # Call a specific tool
        result = await client.call_tool("fs", "read_file", {"path": "/tmp/test.txt"})

        # Access resources
        resources = MCPResourceProvider(client)
        content = await resources.read_text("fs", "file:///tmp/config.json")

        # Use prompts
        prompts = MCPPromptProvider(client)
        prompt = await prompts.get_prompt("assistant", "greeting", {"name": "Alice"})

Configuration from File:
    config = MCPConfig.from_file("mcp.json")
    toolkit = MCPToolkit(config)
"""

# Configuration
from definable.mcp.config import MCPConfig, MCPServerConfig

# Client
from definable.mcp.client import MCPClient, MCPServerConnection

# Toolkit
from definable.mcp.toolkit import MCPToolkit

# Providers
from definable.mcp.prompts import MCPPromptProvider
from definable.mcp.resources import MCPResourceProvider

# Errors
from definable.mcp.errors import (
  MCPConnectionError,
  MCPError,
  MCPPromptNotFoundError,
  MCPProtocolError,
  MCPResourceNotFoundError,
  MCPServerNotFoundError,
  MCPTimeoutError,
  MCPToolNotFoundError,
)

# Types (commonly used)
from definable.mcp.types import (
  MCPPromptDefinition,
  MCPPromptMessage,
  MCPResource,
  MCPResourceContent,
  MCPToolCallResult,
  MCPToolDefinition,
)

__all__ = [
  # Configuration
  "MCPConfig",
  "MCPServerConfig",
  # Client
  "MCPClient",
  "MCPServerConnection",
  # Toolkit
  "MCPToolkit",
  # Providers
  "MCPResourceProvider",
  "MCPPromptProvider",
  # Errors
  "MCPError",
  "MCPConnectionError",
  "MCPTimeoutError",
  "MCPProtocolError",
  "MCPToolNotFoundError",
  "MCPServerNotFoundError",
  "MCPResourceNotFoundError",
  "MCPPromptNotFoundError",
  # Types
  "MCPToolDefinition",
  "MCPToolCallResult",
  "MCPResource",
  "MCPResourceContent",
  "MCPPromptDefinition",
  "MCPPromptMessage",
]
