import asyncio
import json

from definable.agent import Agent
from definable.agent.testing import MockModel


thinking_response = json.dumps({
  "chain_of_thought": "Compare the trade-offs directly.",
  "approach": "Recommend the simpler default first.",
  "tool_plan": [],
})


async def main() -> None:
  agent = Agent(
    model=MockModel(
      responses=["Start with a monolith when speed and coordination matter most."],
      structured_responses=[thinking_response],
    ),
    thinking=True,
    instructions="Be concise.",
  )

  events: list[str] = []
  async for event in agent.arun_stream("Should I start with a monolith or microservices?"):
    if hasattr(event, "event"):
      events.append(event.event)

  assert "ReasoningStarted" in events
  assert "ReasoningCompleted" in events


asyncio.run(main())
