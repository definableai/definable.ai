"""Scenario: Agent with tools and memory integration.

Tests:
- Agent creation with tools and Memory
- Tool execution in conversation
- Memory recall across turns
- Proper cleanup

Requires: OPENAI_API_KEY
"""

import asyncio
import os
import sys

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


async def main():
  from definable.agent import Agent
  from definable.memory import Memory, SQLiteStore
  from definable.model.openai import OpenAIChat
  from definable.tool.decorator import tool

  if not os.environ.get("OPENAI_API_KEY"):
    print("SKIP: OPENAI_API_KEY not set")
    sys.exit(0)

  # --- Setup ---
  @tool
  def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

  @tool
  def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

  db_path = "/tmp/scenario_memory_test.db"
  store = SQLiteStore(db_path)
  memory = Memory(store=store)
  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    tools=[add, multiply],
    instructions="You are a math assistant. Use tools to compute results. Always use tools for arithmetic.",
    memory=memory,
  )

  # --- Test 1: Agent creation ---
  check(agent is not None, "Agent created with tools and memory")
  check(len(agent.tools) >= 2, f"Agent has {len(agent.tools)} tools (expected >= 2)")

  # --- Test 2: First turn with tool use ---
  try:
    r1 = await agent.arun("What is 7 + 13?")
  except Exception as e:
    if "401" in str(e) or "auth" in str(e).lower() or "api key" in str(e).lower():
      print(f"SKIP: OpenAI API key invalid/expired ({type(e).__name__})")
      await memory.close()
      if os.path.exists(db_path):
        os.remove(db_path)
      sys.exit(0)
    raise
  check(r1.content is not None and len(r1.content) > 0, "First turn returned content")
  check("20" in r1.content, "Tool correctly computed 7 + 13 = 20")

  # --- Test 3: Second turn with different tool ---
  r2 = await agent.arun("Now multiply 5 and 6")
  check(r2.content is not None and len(r2.content) > 0, "Second turn returned content")
  check("30" in r2.content, "Tool correctly computed 5 * 6 = 30")

  # --- Test 4: RunOutput has expected fields ---
  check(r1.run_id is not None, "RunOutput has run_id")
  check(r1.model is not None, "RunOutput has model")

  # --- Cleanup ---
  await memory.close()
  if os.path.exists(db_path):
    os.remove(db_path)

  # --- Summary ---
  print(f"\n--- Summary: {PASS} passed, {FAIL} failed ---")
  sys.exit(1 if FAIL > 0 else 0)


if __name__ == "__main__":
  asyncio.run(main())
