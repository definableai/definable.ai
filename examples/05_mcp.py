"""05 — MCP server integration.

Pass an `MCPToolkit` via `mcp=[...]`. The Agent's `aopen()` lifecycle
launches the server, lists its tools, and registers them. `aclose()`
shuts the server down.
"""

from __future__ import annotations

import asyncio

from definable import Agent, MCPToolkit
from definable.agent.mcp import MCPConfig, MCPServerConfig


async def main() -> None:
  fs_config = MCPConfig(
    servers=[
      MCPServerConfig(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      ),
    ],
  )
  fs_mcp = MCPToolkit(config=fs_config)

  agent = Agent(
    name="fs_agent",
    model="openai/gpt-5.4-mini",
    instructions="You can read and write files in /tmp via the filesystem MCP server.",
    mcp=[fs_mcp],
  )
  async with agent:
    print((await agent.arun("List the files in /tmp and tell me how many there are.")).content)


if __name__ == "__main__":
  asyncio.run(main())
