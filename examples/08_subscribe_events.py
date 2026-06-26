"""08 — subscribe to events live.

`agent.events.on(EventType)` registers a typed handler. Everything the
agent does is a step: StepBegin/StepEnd carry a `type` of "content",
"reasoning", or "tool". Use this to build dashboards, telemetry pipelines,
or custom UIs without touching the harness.
"""

from __future__ import annotations

import asyncio

from definable import (
  Agent,
  AgentEnd,
  StepBegin,
  StepEnd,
  tool,
)


@tool
def lookup(symbol: str) -> str:
  """Return a fake stock price for the given ticker."""
  return f"{symbol}: $123.45"


async def main() -> None:
  agent = Agent(
    name="watched",
    model="openai/gpt-5.4-mini",
    instructions="Use the lookup tool when asked about a stock.",
    tools=[lookup],
  )

  @agent.events.on(StepBegin)
  def _on_begin(e: StepBegin) -> None:
    if e.type == "tool":
      print(f"  → calling {e.name}({e.args})")

  @agent.events.on(StepEnd)
  def _on_end(e: StepEnd) -> None:
    if e.type == "tool":
      print(f"  ← tool {'ok' if e.success else 'failed'}: {e.data or e.error}")
    elif e.type == "content":
      tokens = (e.usage or {}).get("total_tokens")
      print(f"  · model answered (turn done){f', {tokens} tokens' if tokens else ''}")

  @agent.events.on(AgentEnd)
  def _on_run_end(e: AgentEnd) -> None:
    print(f"  ✓ run done in {e.turns} turn(s), usage={e.usage}")

  async with agent:
    print((await agent.arun("What's the price of AAPL?")).content)


if __name__ == "__main__":
  asyncio.run(main())
