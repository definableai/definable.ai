"""02 — streaming output.

Calling `arun(stream=True)` returns an async iterator of Events. Token
deltas arrive as `StreamChunkEvent`; the final answer wraps as
`RunCompleted`.
"""

from __future__ import annotations

import asyncio

from definable import Agent, RunCompleted, StreamChunkEvent


async def main() -> None:
  agent = Agent(
    name="streamer",
    model="openai/gpt-5.4-mini",
    instructions="You write three short rhyming couplets.",
  )
  async with agent:
    async for event in await agent.arun("about a cat", stream=True):
      if isinstance(event, StreamChunkEvent):
        print(event.data, end="", flush=True)
      elif isinstance(event, RunCompleted):
        print(f"\n\n[done in {event.turns} turn(s)]")


if __name__ == "__main__":
  asyncio.run(main())
