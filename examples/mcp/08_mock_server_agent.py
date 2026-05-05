"""MCP Mock Server Agent Example.

This example demonstrates full agent integration with MCP using the mock server.
Works in two modes:
- Demo mode (no API key): Shows available tools and setup
- Full mode (with OPENAI_API_KEY): Runs actual agent with MCP tools

Features shown:
- Agent with MCP toolkit integration
- Using echo and add_numbers tools
- Streaming responses
- Tool call inspection

Run demo mode: python 08_mock_server_agent.py
Run full mode: OPENAI_API_KEY=... python 08_mock_server_agent.py
"""

import asyncio
import os
import sys
from pathlib import Path

from definable.mcp import MCPConfig, MCPServerConfig, MCPToolkit

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


async def demo_mode() -> None:
  """Demo mode: Show toolkit setup without running agent."""
  print("=" * 70)
  print("MCP Mock Server Agent Demo (OPENAI_API_KEY not set)")
  print("=" * 70)

  config = get_mock_config()
  toolkit = MCPToolkit(config)

  async with toolkit:
    print("\nConnected to mock MCP server")
    print(f"\nDiscovered {len(toolkit.tools)} tools:")
    print("-" * 50)

    for tool in toolkit.tools:
      print(f"\n  Tool: {tool.name}")
      print(f"  Description: {tool.description}")

    print("\n" + "-" * 50)
    print("\nThese tools would be available to an agent.")
    print("Set OPENAI_API_KEY to run the full agent example.")
    print("-" * 50)

    # Show what queries the agent could handle
    print("\nExample queries the agent could handle:")
    print("  1. 'Echo back: Hello World!'")
    print("  2. 'What is 42 + 58?'")
    print("  3. 'Add the numbers 123 and 456'")


async def agent_mode() -> None:
  """Full agent mode: Run agent with MCP tools."""
  from definable.agent import Agent
  from definable.model.openai import OpenAIChat

  print("=" * 70)
  print("MCP Mock Server Agent Example")
  print("=" * 70)

  config = get_mock_config()
  toolkit = MCPToolkit(config)

  async with toolkit:
    print("\nConnected to mock MCP server")
    print(f"Available tools: {len(toolkit.tools)}")
    for tool in toolkit.tools:
      print(f"  - {tool.name}")

    # Create the agent with the MCP toolkit
    model = OpenAIChat(id="gpt-4o-mini")
    agent = Agent(
      model=model,
      toolkits=[toolkit],
      instructions=(
        "You are a helpful assistant with access to tools. "
        "You can echo messages back and perform arithmetic. "
        "Always use the appropriate tool when asked to echo or add numbers."
      ),
    )

    print("\n" + "-" * 70)
    print("Agent is ready. Running test queries...")
    print("-" * 70)

    # Test queries
    queries = [
      "Echo back this message: 'Hello from the MCP agent!'",
      "What is 42 + 58? Please use the add_numbers tool.",
      "Can you add 1234 and 5678 together?",
    ]

    for i, query in enumerate(queries, 1):
      print(f"\n[Query {i}]: {query}")
      print("-" * 50)

      try:
        response = await agent.arun(query)
        print(f"[Response]: {response.content}")

        # Show tool usage
        if response.tools:
          print("\n[Tools Used]:")
          for tool in response.tools:  # type: ignore[assignment]
            result_preview = tool.result[:80] + "..." if tool.result and len(tool.result) > 80 else tool.result  # type: ignore[attr-defined]
            print(f"  - {tool.name}: {result_preview}")

      except Exception as e:
        print(f"[Error]: {type(e).__name__}: {e}")

      print()

    print("=" * 70)
    print("Agent example completed")
    print("=" * 70)


async def streaming_demo() -> None:
  """Demonstrate streaming with MCP tools."""
  from definable.agent import Agent
  from definable.model.openai import OpenAIChat

  print("\n" + "=" * 70)
  print("Streaming Demo with MCP Tools")
  print("=" * 70)

  config = get_mock_config()
  toolkit = MCPToolkit(config)

  async with toolkit:
    model = OpenAIChat(id="gpt-4o-mini")
    agent = Agent(
      model=model,
      toolkits=[toolkit],
      instructions="You are a helpful assistant. Use tools when appropriate.",
    )

    print("\n[Query]: Add 100 and 200, then explain the result")
    print("-" * 50)
    print("[Streaming Response]: ", end="", flush=True)

    async for chunk in agent.astream("Add 100 and 200, then explain the result"):  # type: ignore[attr-defined]
      if chunk.content:
        print(chunk.content, end="", flush=True)

    print("\n")


async def main() -> None:
  """Run the mock server agent example."""
  # Verify mock server exists
  if not MOCK_SERVER.exists():
    print(f"Error: Mock server not found at {MOCK_SERVER}")
    print("Make sure you're running from the repository root.")
    return

  if os.environ.get("OPENAI_API_KEY"):
    await agent_mode()
    await streaming_demo()
  else:
    await demo_mode()


if __name__ == "__main__":
  asyncio.run(main())
