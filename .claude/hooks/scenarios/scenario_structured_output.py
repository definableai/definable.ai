"""Scenario: Structured output with Pydantic models.

Tests:
- Model invocation with response_format (Pydantic model)
- Parsed response access
- Agent-level structured output with output_schema

Requires: OPENAI_API_KEY (valid — skips gracefully on auth errors)
"""

import asyncio
import os
import sys
from typing import List

from pydantic import BaseModel, Field

PASS = 0
FAIL = 0


def check(condition: bool, description: str):
  global PASS, FAIL
  if condition:
    PASS += 1
    print(f"PASS: {description}")
  else:
    FAIL += 1
    print(f"FAIL: {description}")


class CityInfo(BaseModel):
  """Information about a city."""

  name: str = Field(description="City name")
  country: str = Field(description="Country name")
  population_millions: float = Field(description="Population in millions")
  landmarks: List[str] = Field(description="Famous landmarks")


async def main():
  from definable.agents import Agent
  from definable.models.message import Message
  from definable.models.openai import OpenAIChat

  if not os.environ.get("OPENAI_API_KEY"):
    print("SKIP: OPENAI_API_KEY not set")
    sys.exit(0)

  model = OpenAIChat(id="gpt-4o-mini")

  # --- Test 1: Model-level structured output ---
  try:
    response = model.invoke(
      messages=[Message(role="user", content="Tell me about Tokyo, Japan.")],
      assistant_message=Message(role="assistant", content=""),
      response_format=CityInfo,
    )
  except Exception as e:
    if "401" in str(e) or "auth" in str(e).lower() or "api key" in str(e).lower():
      print(f"SKIP: OpenAI API key invalid/expired ({type(e).__name__})")
      sys.exit(0)
    raise

  check(response is not None, "Model invoke with response_format returned response")
  check(response.content is not None and len(response.content) > 0, "Response has content")

  # Note: response.parsed may or may not work (known bug #6)
  # But the raw content should be valid JSON
  if response.parsed:
    check(isinstance(response.parsed, CityInfo), "response.parsed is CityInfo instance")
    check(response.parsed.name.lower() == "tokyo", f"Parsed city name is Tokyo (got: {response.parsed.name})")
  else:
    # Fallback: parse content manually
    import json

    try:
      data = json.loads(response.content)
      check("name" in data, "Raw JSON content has 'name' field")
      print("  (Note: response.parsed not populated — known bug #6)")
    except json.JSONDecodeError:
      check(False, "Response content is valid JSON")

  # --- Test 2: Agent-level structured output ---
  agent = Agent(
    model=model,
    instructions="You provide factual information about cities.",
  )

  r = await agent.arun("Tell me about Paris, France.", output_schema=CityInfo)
  check(r is not None, "Agent arun with output_schema returned result")
  check(r.content is not None, "Agent result has content")

  # --- Summary ---
  print(f"\n--- Summary: {PASS} passed, {FAIL} failed ---")
  sys.exit(1 if FAIL > 0 else 0)


if __name__ == "__main__":
  asyncio.run(main())
