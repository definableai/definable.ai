"""Minimal example — boot an Agent with the observability dashboard.

Run::

    .venv/bin/python -m definable.observability.examples.basic_agent

The script prints the dashboard URL on startup, then drives a few runs so
the TRACES tab has content to render. ``await asyncio.sleep(...)`` at the
end keeps the dashboard reachable while you click through it.
"""

from __future__ import annotations

import asyncio

from definable import Agent


async def main() -> None:
  agent = Agent(
    name="demo",
    model="claude-haiku-4-5",
    instructions="You are a concise demo agent. Answer in one short sentence.",
    observability=True,
  )

  prompts = [
    "What is 2 + 2?",
    "Name the capital of France.",
    "Tell me a one-line dad joke.",
  ]
  for p in prompts:
    print(f"\n>>> {p}")
    result = await agent.arun(p)
    print(f"<<< {getattr(result, 'content', None)}")

  print("\nDashboard stays live for 5 minutes. Press Ctrl-C to exit early.")
  await asyncio.sleep(300)


if __name__ == "__main__":
  asyncio.run(main())
