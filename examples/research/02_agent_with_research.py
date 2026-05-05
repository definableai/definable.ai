"""Agent with deep research — automatic web research before responding."""

import asyncio
import os

from definable.agent import Agent
from definable.model.openai import OpenAIChat
from definable.agent.research import DeepResearchConfig


async def main():
  model = OpenAIChat(id="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

  # Agent with deep research enabled
  agent = Agent(
    model=model,
    instructions="You are a research assistant. Provide detailed, well-sourced answers.",
    deep_research=DeepResearchConfig(
      depth="standard",  # 3 waves, 15 sources
      include_citations=True,
      include_contradictions=True,
    ),
  )

  output = await agent.arun("Compare the pros and cons of React vs Vue vs Svelte in 2025.")
  print(output.content)


if __name__ == "__main__":
  asyncio.run(main())
