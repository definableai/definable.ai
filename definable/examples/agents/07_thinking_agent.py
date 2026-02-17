"""Agent with thinking enabled — adds a reasoning phase before the response."""

import asyncio
import os

from definable.agents import Agent, ThinkingConfig
from definable.models.openai import OpenAIChat


async def main():
  model = OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

  # --- Simple enable ---
  agent = Agent(
    model=model,
    instructions="You are a helpful assistant.",
    thinking=True,
  )

  output = await agent.arun("What are the trade-offs between SQL and NoSQL databases?")

  print("=== Response ===")
  print(output.content)

  if output.reasoning_steps:
    print("\n=== Reasoning Steps ===")
    for step in output.reasoning_steps:
      print(f"  [{step.title}] {step.reasoning}")

  # --- Custom thinking config ---
  agent_custom = Agent(
    model=model,
    instructions="You are a senior architect.",
    thinking=ThinkingConfig(
      model=OpenAIChat(id="gpt-4o"),  # Use a stronger model for thinking
    ),
  )

  output2 = await agent_custom.arun("Design a caching strategy for a social media feed.")
  print("\n=== Custom Thinking Response ===")
  print(output2.content)


if __name__ == "__main__":
  asyncio.run(main())
