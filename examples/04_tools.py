"""04 — custom tools.

Decorate a function with `@tool` and pass it via `tools=[...]`. The
agent gets the function's docstring as a description and its type
annotations as the JSON schema.
"""

from __future__ import annotations

import asyncio

from definable import Agent, tool


@tool
def add(x: int, y: int) -> int:
  """Add two integers and return the sum."""
  return x + y


@tool
def shout(text: str) -> str:
  """Return the input text in uppercase with three exclamation marks."""
  return text.upper() + "!!!"


async def main() -> None:
  agent = Agent(
    name="calc",
    model="anthropic/claude-haiku-4-5-20251001",
    instructions="Use the available tools. Don't compute in your head when a tool exists.",
    tools=[add, shout],
  )
  async with agent:
    print((await agent.arun("What is 17 + 26? Then shout the result.")).content)


if __name__ == "__main__":
  asyncio.run(main())
