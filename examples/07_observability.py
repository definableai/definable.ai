"""07 — observability via JSONL persistence.

`observability=True` auto-attaches a JSONL writer at
`.definable/traces/{agent_name}.jsonl`. Every event the bus emits gets
serialized one-per-line. Inspect after the run for full visibility.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from definable import Agent


async def main() -> None:
  agent = Agent(
    name="observed",
    model="openai/gpt-5.4-mini",
    instructions="Be concise.",
    observability=True,
  )
  async with agent:
    await agent.arun("In one sentence: what is the speed of light?")

  trace = Path(".definable/traces/observed.jsonl")
  if trace.exists():
    print(f"\nTrace written to {trace}:")
    for line in trace.read_text().splitlines()[-3:]:
      print(f"  {json.loads(line)['_type']}")


if __name__ == "__main__":
  asyncio.run(main())
