"""
Asynchronous model invocation.

This example shows how to:
- Use async/await with model.ainvoke()
- Handle async operations properly with asyncio

Requirements:
    export OPENAI_API_KEY=sk-...
"""

import asyncio

from definable.models.message import Message
from definable.models.openai import OpenAIChat


async def main():
  # Initialize the model
  model = OpenAIChat(id="gpt-4o-mini")

  # Create the conversation messages
  messages = [
    Message(role="user", content="Write a haiku about programming."),
  ]

  # Invoke the model asynchronously
  response = await model.ainvoke(
    messages=messages,
    assistant_message=Message(role="assistant", content=""),
  )

  print("Response:")
  print(response.content)


async def parallel_requests():
  """Example of making multiple async requests in parallel."""
  model = OpenAIChat(id="gpt-4o-mini")

  # Define multiple prompts
  prompts = [
    "What is 2 + 2?",
    "What is the color of the sky?",
    "Name a famous scientist.",
  ]

  # Create tasks for parallel execution
  tasks = []
  for prompt in prompts:
    task = model.ainvoke(
      messages=[Message(role="user", content=prompt)],
      assistant_message=Message(role="assistant", content=""),
    )
    tasks.append(task)

  # Wait for all responses
  responses = await asyncio.gather(*tasks)

  print("\nParallel Responses:")
  for prompt, response in zip(prompts, responses):
    print(f"  Q: {prompt}")
    print(f"  A: {response.content}")
    print()


if __name__ == "__main__":
  # Run single request
  asyncio.run(main())

  # Run parallel requests
  asyncio.run(parallel_requests())
