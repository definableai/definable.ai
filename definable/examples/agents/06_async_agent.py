"""
Async agent operations.

This example shows how to:
- Use arun() for async agent execution
- Use arun_stream() for async streaming
- Handle multiple concurrent agent calls

Requirements:
    export OPENAI_API_KEY=sk-...
"""

import asyncio

from definable.agent import Agent
from definable.model.openai import OpenAIChat
from definable.tool.decorator import tool


@tool
async def async_lookup(query: str) -> str:
  """Perform an async lookup (simulated with delay)."""
  await asyncio.sleep(0.1)  # Simulate async operation
  return f"Results for '{query}': Found 42 items"


async def basic_async():
  """Basic async agent execution."""
  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    instructions="You are a helpful assistant.",
  )

  print("Async Agent Execution")
  print("-" * 40)

  # Run the agent asynchronously
  output = await agent.arun("What is the speed of light?")

  print(f"Response: {output.content}")
  print("-" * 40)


async def async_with_tools():
  """Async agent with async tools."""
  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    tools=[async_lookup],
    instructions="You are a search assistant. Use the async_lookup tool to search for information.",
  )

  print("\nAsync Agent with Tools")
  print("-" * 40)

  output = await agent.arun("Search for Python tutorials")

  print(f"Response: {output.content}")
  if output.tools:
    print(f"Tools used: {[t.tool_name for t in output.tools]}")
  print("-" * 40)


async def async_streaming():
  """Async streaming with arun_stream()."""
  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    instructions="You are a helpful assistant.",
  )

  print("\nAsync Streaming")
  print("-" * 40)

  # Stream asynchronously
  async for event in agent.arun_stream("Count from 1 to 5."):
    if hasattr(event, "content") and event.content:
      print(event.content, end="", flush=True)

  print("\n" + "-" * 40)


async def parallel_agents():
  """Run multiple agents in parallel."""
  model = OpenAIChat(id="gpt-4o-mini")

  # Create different agents for different tasks
  math_agent = Agent(
    model=model,
    instructions="You are a math assistant. Give concise answers.",
  )

  science_agent = Agent(
    model=model,
    instructions="You are a science assistant. Give concise answers.",
  )

  history_agent = Agent(
    model=model,
    instructions="You are a history assistant. Give concise answers.",
  )

  print("\nParallel Agent Execution")
  print("-" * 40)

  # Run all agents in parallel
  math_task = math_agent.arun("What is 15 * 7?")
  science_task = science_agent.arun("What is H2O?")
  history_task = history_agent.arun("Who was the first US president?")

  # Wait for all to complete
  results = await asyncio.gather(math_task, science_task, history_task)

  print("Math Agent:", results[0].content)
  print("Science Agent:", results[1].content)
  print("History Agent:", results[2].content)
  print("-" * 40)


async def async_multi_turn():
  """Async multi-turn conversation."""
  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    instructions="You are a helpful assistant. Remember our conversation.",
  )

  print("\nAsync Multi-turn Conversation")
  print("-" * 40)

  # First turn
  output1 = await agent.arun("Hi, I'm learning Python.")
  print("User: Hi, I'm learning Python.")
  print(f"Agent: {output1.content}")

  # Second turn - pass previous messages
  output2 = await agent.arun(
    "What resources do you recommend?",
    messages=output1.messages,
  )
  print("\nUser: What resources do you recommend?")
  print(f"Agent: {output2.content}")

  print("-" * 40)


async def parallel_streaming():
  """Stream multiple agents in parallel (advanced)."""
  model = OpenAIChat(id="gpt-4o-mini")

  agent1 = Agent(model=model, instructions="You are agent 1. Be brief.")
  agent2 = Agent(model=model, instructions="You are agent 2. Be brief.")

  print("\nParallel Streaming (Advanced)")
  print("-" * 40)

  async def stream_agent(agent: Agent, prompt: str, name: str):
    """Helper to stream an agent and collect output."""
    chunks = []
    async for event in agent.arun_stream(prompt):
      if hasattr(event, "content") and event.content:
        chunks.append(event.content)
    return name, "".join(chunks)

  # Run both streams in parallel
  results = await asyncio.gather(
    stream_agent(agent1, "Say hello", "Agent 1"),
    stream_agent(agent2, "Say goodbye", "Agent 2"),
  )

  for name, response in results:
    print(f"{name}: {response}")

  print("-" * 40)


async def main():
  """Run all async examples."""
  await basic_async()
  await async_with_tools()
  await async_streaming()
  await parallel_agents()
  await async_multi_turn()
  await parallel_streaming()


if __name__ == "__main__":
  asyncio.run(main())
