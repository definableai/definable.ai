"""Live observability dashboard demo.

Constructs one Agent with two tools, seeds a few runs so the TRACES tab
has content immediately, then keeps the process alive so you can drive
the agent live from the browser playground.

Run::

    .venv/bin/python agents/dashboard_demo.py

Then open the URL printed in the terminal (default
``http://127.0.0.1:7777``) and:
  - PLAYGROUND — chat with the live agent; see tokens, cost, tool spans
    update per turn
  - TRACES — every `arun` invocation shows up live; click a row for
    the waterfall + event log
  - METRICS — aggregated RPS, p50/p95, tokens, cost over the window

Press Ctrl-C to exit.

Requires the ``[serve]`` extra (``pip install definable[serve]``).
Set ``OPENAI_API_KEY`` or change ``MODEL`` below for a different provider.
"""

from __future__ import annotations

import asyncio
import os
import random
from datetime import datetime, timezone

from definable import Agent, tool

MODEL = os.environ.get("DEFINABLE_DEMO_MODEL", "openai/gpt-5.4-mini")

SEED_PROMPTS = [
  "What is 137 + 264? Use the add tool.",
  "What time is it right now? Use the clock tool.",
  "Give me a fun fact about octopuses in one short sentence.",
  "Roll two six-sided dice and tell me the sum. Use roll_dice.",
]


@tool
def add(x: int, y: int) -> int:
  """Add two integers and return the sum."""
  return x + y


@tool
def clock() -> str:
  """Return the current UTC time as an ISO-8601 string."""
  return datetime.now(timezone.utc).isoformat(timespec="seconds")


@tool
def roll_dice(count: int = 2, sides: int = 6) -> dict:
  """Roll `count` dice of `sides` faces each. Returns rolls and total."""
  rolls = [random.randint(1, sides) for _ in range(count)]
  return {"rolls": rolls, "total": sum(rolls)}


async def main() -> None:
  agent = Agent(
    name="demo",
    model=MODEL,
    instructions=(
      "You are a friendly demo agent showcasing the Definable observability dashboard. "
      "Use the provided tools when relevant. Keep answers short and direct."
    ),
    tools=[add, clock, roll_dice],
    observability=True,
  )

  async with agent:
    # Seed a few runs so the TRACES tab has content the moment the page loads.
    print("\n=== Seeding 4 runs ===")
    for i, prompt in enumerate(SEED_PROMPTS, start=1):
      print(f"\n[{i}/{len(SEED_PROMPTS)}] >>> {prompt}")
      result = await agent.arun(prompt)
      content = getattr(result, "content", None) or "(no content)"
      print(f"        <<< {content}")

    print("\n=== Dashboard live ===")
    print("Open the URL printed above in a browser. Use the PLAYGROUND tab to chat;")
    print("TRACES + METRICS update live as runs complete.")
    print("Press Ctrl-C to exit.\n")

    # Idle the process so the dashboard stays reachable.
    try:
      while True:
        await asyncio.sleep(3600)
    except asyncio.CancelledError:
      pass


if __name__ == "__main__":
  try:
    asyncio.run(main())
  except KeyboardInterrupt:
    print("\n[demo] exit")
