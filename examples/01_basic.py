"""01 — basic agent.

The minimum viable Agent: a name, a model, and an instruction.
"""

from __future__ import annotations

import asyncio

from definable import Agent


async def main() -> None:
  agent = Agent(
    name="echo",
    model="anthropic/claude-haiku-4-5-20251001",
    instructions="You are a concise assistant. Answer in one sentence.",
  )
  async with agent:
    result = await agent.arun("What is 17 * 23?")
    print(result.content)


if __name__ == "__main__":
  asyncio.run(main())
