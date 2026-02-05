"""MCP Configuration from File Example.

This example demonstrates loading MCP configuration from a JSON file
for easier management of server configurations.

Usage:
    1. Create mcp_config.json (see below)
    2. OPENAI_API_KEY=... python 04_config_file.py

Example mcp_config.json:
{
    "servers": [
        {
            "name": "filesystem",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        }
    ],
    "auto_connect": true,
    "reconnect_on_failure": true
}
"""

import asyncio
import json
import os
from pathlib import Path

from definable.agents import Agent
from definable.mcp import MCPConfig, MCPToolkit
from definable.models.openai import OpenAIChat


async def main() -> None:
  # Path to config file
  config_path = Path(__file__).parent / "mcp_config.json"

  # Create sample config if it doesn't exist
  if not config_path.exists():
    print(f"Creating sample config at {config_path}")
    sample_config = {
      "servers": [
        {
          "name": "filesystem",
          "transport": "stdio",
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
          "connect_timeout": 30.0,
          "request_timeout": 60.0,
        }
      ],
      "auto_connect": True,
      "reconnect_on_failure": True,
    }
    with open(config_path, "w") as f:
      json.dump(sample_config, f, indent=2)
    print("Sample config created!\n")

  # Load config from file
  print(f"Loading config from {config_path}")
  config = MCPConfig.from_file(config_path)

  print(f"Loaded {len(config.servers)} server(s):")
  for server in config.servers:
    print(f"  - {server.name} ({server.transport})")

  # Create toolkit and agent
  toolkit = MCPToolkit(config)

  async with toolkit:
    print(f"\nInitialized with {len(toolkit.tools)} tools")

    agent = Agent(
      model=OpenAIChat(id="gpt-4o-mini"),
      toolkits=[toolkit],
      instructions="You are a helpful assistant with filesystem access.",
    )

    # Test
    print("\n--- Agent Response ---")
    output = await agent.arun("What files are in /tmp?")
    print(output.content)

  # Save config back (demonstrates round-trip)
  output_path = Path(__file__).parent / "mcp_config_saved.json"
  config.to_file(output_path)
  print(f"\nConfig saved to {output_path}")


if __name__ == "__main__":
  if not os.environ.get("OPENAI_API_KEY"):
    print("Please set OPENAI_API_KEY environment variable")
    exit(1)

  asyncio.run(main())
