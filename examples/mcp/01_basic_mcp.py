"""Basic MCP Integration Example.

This example demonstrates connecting to a single MCP server
and using its tools with an agent.

Prerequisites:
    npm install -g @modelcontextprotocol/server-filesystem

Usage:
    OPENAI_API_KEY=... python 01_basic_mcp.py
"""

import asyncio
import os

from definable.agent import Agent
from definable.mcp import MCPConfig, MCPServerConfig, MCPToolkit
from definable.model.openai import OpenAIChat


async def main() -> None:
  # Configure MCP server
  # Using the official filesystem server from Anthropic
  config = MCPConfig(
    servers=[
      MCPServerConfig(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      ),
    ]
  )

  # Create toolkit
  toolkit = MCPToolkit(config)

  # Initialize toolkit and create agent
  async with toolkit:
    print(f"MCPToolkit initialized: {toolkit}")
    print(f"Available tools: {[t.name for t in toolkit.tools]}")

    # Create agent with MCP tools
    agent = Agent(
      model=OpenAIChat(id="gpt-4o-mini"),
      toolkits=[toolkit],
      instructions="""You are a helpful assistant with access to filesystem tools.
            You can read and write files in /tmp directory.""",
    )

    # Test the agent
    print("\n--- Agent Response ---")
    output = await agent.arun("List the files in /tmp directory")
    print(output.content)

    # Show tool usage
    if output.tools:
      print("\n--- Tools Used ---")
      for tool in output.tools:
        print(f"  - {tool.name}: {tool.result[:100]}..." if tool.result else f"  - {tool.name}")  # type: ignore[attr-defined]


if __name__ == "__main__":
  # Check for API key
  if not os.environ.get("OPENAI_API_KEY"):
    print("Please set OPENAI_API_KEY environment variable")
    exit(1)

  asyncio.run(main())
