"""08 — subscribe to events live.

`agent.events.on(EventType)` registers a typed handler. Use this to
build dashboards, telemetry pipelines, or custom UIs without touching
the harness.
"""

from __future__ import annotations

import asyncio

from definable import (
  Agent,
  ModelResponded,
  ToolCallCompleted,
  ToolCallStarted,
  TurnStarted,
  tool,
)


@tool
def lookup(symbol: str) -> str:
  """Return a fake stock price for the given ticker."""
  return f"{symbol}: $123.45"


async def main() -> None:
  agent = Agent(
    name="watched",
    model="anthropic/claude-haiku-4-5-20251001",
    instructions="Use the lookup tool when asked about a stock.",
    tools=[lookup],
  )

  @agent.events.on(TurnStarted)
  def _on_turn(e: TurnStarted) -> None:
    s = e.snapshot
    print(f"  ▸ turn {s.turn}: {s.context_tokens} tokens, {len(s.available_tools)} tools")

  @agent.events.on(ToolCallStarted)
  def _on_call(e: ToolCallStarted) -> None:
    print(f"  → calling {e.call.name}({e.call.args})")

  @agent.events.on(ToolCallCompleted)
  def _on_done(e: ToolCallCompleted) -> None:
    print(f"  ← {e.call.name} = {e.output}")

  @agent.events.on(ModelResponded)
  def _on_resp(e: ModelResponded) -> None:
    if e.tool_calls:
      print(f"  · model wants {len(e.tool_calls)} tool call(s)")
    else:
      print("  · model gave final answer")

  async with agent:
    print((await agent.arun("What's the price of AAPL?")).content)


if __name__ == "__main__":
  asyncio.run(main())
