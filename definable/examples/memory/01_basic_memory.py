"""
Agent with persistent memory.

This example shows how to:
- Set up an agent with persistent memory using SQLiteStore
- Have the agent remember facts across conversation turns
- Use the memory=True shorthand for quick testing
- Clean up memory data

Requirements:
    export OPENAI_API_KEY=sk-...
"""

import asyncio
import os

from definable.agent import Agent
from definable.memory import Memory, SQLiteStore
from definable.model.openai import OpenAIChat


async def main():
  # 1. Create the memory store (SQLite file — persists across runs)
  store = SQLiteStore("./example_memory.db")

  # 2. Create Memory with the store
  #    - The LLM decides what to remember via tool calls (add/update/delete)
  #    - Store is initialized lazily on first use
  memory = Memory(store=store)

  # 3. Create an agent with memory attached
  model = OpenAIChat(id="gpt-4o-mini")
  agent = Agent(
    model=model,
    instructions="You are a helpful assistant. Use the memory context provided to personalize your responses.",
    memory=memory,
  )

  # 4. First turn — introduce some facts
  print("=" * 50)
  print("Turn 1: Introducing facts")
  print("=" * 50)
  output = await agent.arun("Hi! My name is Alice and I'm a Python developer in San Francisco.")
  print(f"Agent: {output.content}\n")

  # 5. Second turn — the agent should recall the facts from turn 1
  print("=" * 50)
  print("Turn 2: Testing recall")
  print("=" * 50)
  output = await agent.arun("What do you know about me?")
  print(f"Agent: {output.content}\n")

  # 6. Third turn — more context builds up
  print("=" * 50)
  print("Turn 3: Adding more context")
  print("=" * 50)
  output = await agent.arun("I'm working on a FastAPI project that uses PostgreSQL.")
  print(f"Agent: {output.content}\n")

  # 7. Clean up
  await memory.close()

  # Remove the example db file
  if os.path.exists("./example_memory.db"):
    os.remove("./example_memory.db")

  print("Done! Memory store closed and cleaned up.")


if __name__ == "__main__":
  asyncio.run(main())
