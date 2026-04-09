"""MCP Error Handling Example.

This example demonstrates proper error handling patterns for MCP operations.
No external services or API keys required - fully self-contained.

Features shown:
- MCPServerNotFoundError handling
- Tool error responses (isError: true)
- MCPConnectionError handling
- Timeout handling with slow_tool

Run with: python 07_error_handling.py
"""

import asyncio
import sys
from pathlib import Path

from definable.mcp import (
  MCPClient,
  MCPConfig,
  MCPConnectionError,
  MCPPromptProvider,
  MCPServerConfig,
  MCPServerNotFoundError,
  MCPToolkit,
)
from definable.mcp.errors import MCPPromptNotFoundError

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


async def demo_server_not_found() -> None:
  """Demonstrate handling MCPServerNotFoundError."""
  print("=" * 70)
  print("Part 1: Server Not Found Errors")
  print("=" * 70)

  config = get_mock_config()
  client = MCPClient(config)

  async with client:
    # Try to call a tool on a non-existent server
    print("\n1. Calling tool on non-existent server:")
    try:
      await client.call_tool("nonexistent_server", "echo", {"message": "hello"})
    except MCPServerNotFoundError as e:
      print("   Caught MCPServerNotFoundError!")
      print(f"   Server requested: {e.server_name}")
      print(f"   Available servers: {e.available_servers}")
      print(f"   Message: {e}")

    # Try to list prompts from non-existent server
    print("\n2. Listing prompts from non-existent server:")
    prompts = MCPPromptProvider(client)
    try:
      await prompts.list_prompts(server_name="missing_server")
    except MCPServerNotFoundError as e:
      print("   Caught MCPServerNotFoundError!")
      print(f"   Server: {e.server_name}")


async def demo_tool_errors() -> None:
  """Demonstrate handling tool error responses."""
  print("\n" + "=" * 70)
  print("Part 2: Tool Error Responses")
  print("=" * 70)

  config = get_mock_config()
  client = MCPClient(config)

  async with client:
    # Helper to extract text from content
    def get_text(result):
      return result.content[0].text if result.content else ""

    # Call the error_tool which always returns isError: true
    print("\n1. Calling 'error_tool' (always returns error):")
    result = await client.call_tool(
      "mock",
      "error_tool",
      {"message": "Something went wrong!"},
    )
    print(f"   Result content: {get_text(result)}")
    print(f"   Is error: {result.isError}")

    # Demonstrate checking for errors in your code
    print("\n2. Proper error checking pattern:")
    result = await client.call_tool("mock", "error_tool", {})
    if result.isError:
      print(f"   Tool returned an error: {get_text(result)}")
      print("   Handle error appropriately...")
    else:
      print(f"   Success: {get_text(result)}")

    # Call a tool that doesn't exist
    print("\n3. Calling non-existent tool:")
    result = await client.call_tool("mock", "nonexistent_tool", {})
    print(f"   Is error: {result.isError}")
    print(f"   Message: {get_text(result)}")


async def demo_connection_errors() -> None:
  """Demonstrate handling connection errors."""
  print("\n" + "=" * 70)
  print("Part 3: Connection Errors")
  print("=" * 70)

  # Try to connect to a server with an invalid command
  print("\n1. Connecting to server with invalid command:")
  bad_config = MCPConfig(
    servers=[
      MCPServerConfig(
        name="bad_server",
        transport="stdio",
        command="/nonexistent/path/to/server",
        args=[],
        connect_timeout=2.0,
      ),
    ]
  )

  toolkit = MCPToolkit(bad_config)
  try:
    async with toolkit:
      pass
  except MCPConnectionError as e:
    print("   Caught MCPConnectionError!")
    print(f"   Message: {e}")
  except FileNotFoundError as e:
    print("   Caught FileNotFoundError (command not found)!")
    print(f"   Message: {e}")
  except Exception as e:
    print(f"   Caught {type(e).__name__}: {e}")


async def demo_timeout_handling() -> None:
  """Demonstrate timeout handling with slow tools."""
  print("\n" + "=" * 70)
  print("Part 4: Timeout Handling")
  print("=" * 70)

  config = get_mock_config()
  client = MCPClient(config)

  async with client:
    # Helper to extract text from content
    def get_text(result):
      return result.content[0].text if result.content else ""

    # Call slow_tool with short delay (should succeed)
    print("\n1. Calling slow_tool with 0.5s delay:")
    result = await client.call_tool("mock", "slow_tool", {"delay": 0.5})
    print(f"   Result: {get_text(result)}")

    # Demonstrate asyncio.wait_for for custom timeout
    print("\n2. Using asyncio.wait_for for custom timeout:")
    try:
      # This will timeout because we set a 1s limit but tool takes 3s
      result = await asyncio.wait_for(
        client.call_tool("mock", "slow_tool", {"delay": 3.0}),
        timeout=1.0,
      )
      print(f"   Result: {result.content}")
    except asyncio.TimeoutError:
      print("   Caught asyncio.TimeoutError!")
      print("   The tool call took too long and was cancelled.")


async def demo_prompt_not_found() -> None:
  """Demonstrate handling prompt not found errors."""
  print("\n" + "=" * 70)
  print("Part 5: Prompt Not Found Errors")
  print("=" * 70)

  config = get_mock_config()
  client = MCPClient(config)

  async with client:
    prompts = MCPPromptProvider(client)

    # Try to get a non-existent prompt
    print("\n1. Getting non-existent prompt:")
    try:
      await prompts.get_prompt("mock", "nonexistent_prompt", {})
    except Exception as e:
      print(f"   Caught {type(e).__name__}!")
      print(f"   Message: {e}")

    # Try to get arguments for non-existent prompt
    print("\n2. Getting arguments for non-existent prompt:")
    try:
      await prompts.get_prompt_arguments("mock", "missing_prompt")
    except MCPPromptNotFoundError as e:
      print("   Caught MCPPromptNotFoundError!")
      print(f"   Prompt: {e.prompt_name}")
      print(f"   Server: {e.server_name}")


async def demo_graceful_degradation() -> None:
  """Demonstrate graceful degradation patterns."""
  print("\n" + "=" * 70)
  print("Part 6: Graceful Degradation Patterns")
  print("=" * 70)

  config = get_mock_config()
  client = MCPClient(config)

  async with client:
    prompts = MCPPromptProvider(client)

    # Helper to extract text from content
    def get_text(result):
      return result.content[0].text if result.content else ""

    # Pattern: Try to find a prompt, fallback gracefully
    print("\n1. Try prompt with fallback:")
    prompt_name = "nonexistent_feature"
    server = await prompts.find_prompt(prompt_name)
    if server:
      result = await prompts.get_text(server, prompt_name, {})
      print(f"   Using prompt: {result}")
    else:
      print(f"   Prompt '{prompt_name}' not found, using default behavior")
      print("   Default: Please describe what you need help with.")

    # Pattern: Check tool availability before use
    print("\n2. Check tool availability before use:")
    tools = await client.list_all_tools()
    mock_tools = tools.get("mock", [])
    tool_names = [t.name for t in mock_tools]

    if "echo" in tool_names:
      result = await client.call_tool("mock", "echo", {"message": "Safe call!"})  # type: ignore[assignment]
      print(f"   Echo tool available: {get_text(result)}")
    else:
      print("   Echo tool not available, using alternative...")

    # Pattern: Handle partial failures
    print("\n3. Handle partial failures in batch operations:")
    operations = [
      ("echo", {"message": "test1"}),
      ("nonexistent", {}),
      ("add_numbers", {"a": 1, "b": 2}),
    ]

    results = []
    for tool_name, args in operations:
      result = await client.call_tool("mock", tool_name, args)  # type: ignore[assignment, arg-type]
      if result.isError:  # type: ignore[attr-defined]
        print(f"   {tool_name}: FAILED - {get_text(result)}")
      else:
        print(f"   {tool_name}: OK - {get_text(result)}")
        results.append(result)

    print(f"   Successful operations: {len(results)}/{len(operations)}")


async def main() -> None:
  """Run all demonstrations."""
  print("MCP Error Handling Example")
  print("No API keys required - uses local mock server")
  print()

  # Verify mock server exists
  if not MOCK_SERVER.exists():
    print(f"Error: Mock server not found at {MOCK_SERVER}")
    print("Make sure you're running from the repository root.")
    return

  await demo_server_not_found()
  await demo_tool_errors()
  await demo_connection_errors()
  await demo_timeout_handling()
  await demo_prompt_not_found()
  await demo_graceful_degradation()

  print("\n" + "=" * 70)
  print("Example completed successfully!")
  print("=" * 70)


if __name__ == "__main__":
  asyncio.run(main())
