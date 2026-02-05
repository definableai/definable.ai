"""Multiple MCP Servers Example.

This example demonstrates connecting to multiple MCP servers
and using tools from all of them with a single agent.

Prerequisites:
    npm install -g @modelcontextprotocol/server-filesystem
    npm install -g @modelcontextprotocol/server-memory

Usage:
    OPENAI_API_KEY=... python 02_multiple_servers.py
"""

import asyncio
import os

from definable.agents import Agent
from definable.mcp import MCPConfig, MCPServerConfig, MCPToolkit
from definable.models.openai import OpenAIChat


async def main() -> None:
  # Configure multiple MCP servers
  config = MCPConfig(
    servers=[
      MCPServerConfig(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      ),
      MCPServerConfig(
        name="memory",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-memory"],
      ),
    ]
  )

  # Create toolkit
  toolkit = MCPToolkit(config)

  async with toolkit:
    print(f"MCPToolkit initialized with {len(toolkit.tools)} tools")
    print("\nTools by server:")
    for tool in toolkit.tools:
      server = toolkit.get_tool_server(tool.name)
      print(f"  [{server}] {tool.name}")

    # Create agent
    agent = Agent(
      model=OpenAIChat(id="gpt-4o-mini"),
      toolkits=[toolkit],
      instructions="""You are a helpful assistant with access to:
            1. Filesystem tools - for reading/writing files in /tmp
            2. Memory tools - for storing and retrieving information

            Use the memory tools to remember things across conversations.""",
    )

    # Test with multi-tool workflow
    print("\n--- Storing information in memory ---")
    output1 = await agent.arun("Store the following in memory with key 'project_info': 'This is the definable.ai project for building AI agents.'")
    print(output1.content)

    print("\n--- Retrieving from memory ---")
    output2 = await agent.arun(
      "What information do you have stored about the project?",
      messages=output1.messages,
    )
    print(output2.content)


if __name__ == "__main__":
  if not os.environ.get("OPENAI_API_KEY"):
    print("Please set OPENAI_API_KEY environment variable")
    exit(1)

  asyncio.run(main())
