"""06 — built-in skills.

Skills are pre-built capability bundles. Pass instances via `skills=[...]`.
"""

from __future__ import annotations

import asyncio

from definable import Agent
from definable.agent.skill import Calculator, DateTime


async def main() -> None:
  agent = Agent(
    name="skilled",
    model="openai/gpt-5.4-mini",
    instructions="Use the loaded skills when relevant.",
    skills=[Calculator(), DateTime()],
  )
  async with agent:
    print((await agent.arun("What's 17 * 23?")).content)


if __name__ == "__main__":
  asyncio.run(main())
