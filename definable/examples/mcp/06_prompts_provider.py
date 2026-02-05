"""MCP Prompts Provider Example.

This example demonstrates MCPPromptProvider functionality using the mock server.
No external services or API keys required - fully self-contained.

Features shown:
- Listing prompts from servers
- Rendering prompts with arguments
- Convenience methods: get_text(), find_prompt(), get_prompt_arguments()

Run with: python 06_prompts_provider.py
"""

import asyncio
import sys
from pathlib import Path

from definable.mcp import (
  MCPClient,
  MCPConfig,
  MCPPromptProvider,
  MCPServerConfig,
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


async def demo_list_prompts() -> None:
  """Demonstrate listing available prompts."""
  print("=" * 70)
  print("Part 1: Listing Available Prompts")
  print("=" * 70)

  config = get_mock_config()
  client = MCPClient(config)

  async with client:
    prompts = MCPPromptProvider(client)

    # List all prompts from all servers
    print("\n1. All prompts:")
    all_prompts = await prompts.list_prompts()
    for server_name, prompt_list in all_prompts.items():
      print(f"\n   Server: {server_name}")
      for prompt in prompt_list:
        print(f"   - {prompt.name}: {prompt.description}")
        if prompt.arguments:
          args = [f"{a.name}{'*' if a.required else ''}" for a in prompt.arguments]
          print(f"     Arguments: {', '.join(args)}  (* = required)")

    # List prompts from specific server
    print("\n2. Prompts from 'mock' server only:")
    mock_prompts = await prompts.list_prompts(server_name="mock")
    print(f"   Found {len(mock_prompts.get('mock', []))} prompts")


async def demo_render_prompts() -> None:
  """Demonstrate rendering prompts with arguments."""
  print("\n" + "=" * 70)
  print("Part 2: Rendering Prompts")
  print("=" * 70)

  config = get_mock_config()
  client = MCPClient(config)

  async with client:
    prompts = MCPPromptProvider(client)

    # Render the greeting prompt
    print("\n1. Greeting prompt with name='Alice':")
    result = await prompts.get_prompt("mock", "greeting", {"name": "Alice"})
    print(f"   Description: {result.description}")
    for msg in result.messages:
      print(f"   [{msg.role}]: {msg.content.text}")

    # Render with different arguments
    print("\n2. Greeting prompt with name='Bob':")
    result = await prompts.get_prompt("mock", "greeting", {"name": "Bob"})
    for msg in result.messages:
      print(f"   [{msg.role}]: {msg.content.text}")

    # Render code review prompt with multiple arguments
    print("\n3. Code review prompt (language='python', style='detailed'):")
    result = await prompts.get_prompt(
      "mock",
      "code_review",
      {"language": "python", "style": "detailed"},
    )
    for msg in result.messages:
      print(f"   [{msg.role}]: {msg.content.text}")

    # Render with optional argument omitted
    print("\n4. Code review prompt (only required args):")
    result = await prompts.get_prompt(
      "mock",
      "code_review",
      {"language": "javascript"},
    )
    for msg in result.messages:
      print(f"   [{msg.role}]: {msg.content.text}")


async def demo_convenience_methods() -> None:
  """Demonstrate convenience methods."""
  print("\n" + "=" * 70)
  print("Part 3: Convenience Methods")
  print("=" * 70)

  config = get_mock_config()
  client = MCPClient(config)

  async with client:
    prompts = MCPPromptProvider(client)

    # get_text() - Get prompt as concatenated text
    print("\n1. get_text() - Returns formatted text:")
    text = await prompts.get_text("mock", "greeting", {"name": "World"})
    print(f"   Result: {text}")

    # get_messages() - Get messages as a list
    print("\n2. get_messages() - Returns message list:")
    messages = await prompts.get_messages("mock", "code_review", {"language": "rust"})
    print(f"   Got {len(messages)} message(s)")
    for msg in messages:
      print(f"   - Role: {msg.role}, Content: {msg.content.text[:50]}...")

    # find_prompt() - Find which server has a prompt
    print("\n3. find_prompt() - Find prompt location:")
    server = await prompts.find_prompt("greeting")
    print(f"   'greeting' prompt is on server: {server}")

    server = await prompts.find_prompt("nonexistent")
    print(f"   'nonexistent' prompt: {server}")

    # get_prompt_arguments() - Get argument names
    print("\n4. get_prompt_arguments() - Get argument names:")
    args = await prompts.get_prompt_arguments("mock", "greeting")
    print(f"   'greeting' arguments: {args}")

    args = await prompts.get_prompt_arguments("mock", "code_review")
    print(f"   'code_review' arguments: {args}")

    # get_prompt_info() - Get full prompt metadata
    print("\n5. get_prompt_info() - Get prompt metadata:")
    info = await prompts.get_prompt_info("mock", "greeting")
    if info:
      print(f"   Name: {info.name}")
      print(f"   Description: {info.description}")
      if info.arguments:
        print("   Arguments:")
        for arg in info.arguments:
          req = "(required)" if arg.required else "(optional)"
          print(f"     - {arg.name}: {arg.description} {req}")


async def main() -> None:
  """Run all demonstrations."""
  print("MCP Prompts Provider Example")
  print("No API keys required - uses local mock server")
  print()

  # Verify mock server exists
  if not MOCK_SERVER.exists():
    print(f"Error: Mock server not found at {MOCK_SERVER}")
    print("Make sure you're running from the repository root.")
    return

  await demo_list_prompts()
  await demo_render_prompts()
  await demo_convenience_methods()

  print("\n" + "=" * 70)
  print("Example completed successfully!")
  print("=" * 70)


if __name__ == "__main__":
  asyncio.run(main())
