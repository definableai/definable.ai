"""
Streaming model responses.

This example shows how to:
- Use invoke_stream() for sync streaming
- Use ainvoke_stream() for async streaming
- Process chunks as they arrive

Requirements:
    export OPENAI_API_KEY=sk-...
"""

import asyncio

from definable.model.message import Message
from definable.model.openai import OpenAIChat


def sync_streaming():
  """Synchronous streaming example."""
  model = OpenAIChat(id="gpt-4o-mini")

  messages = [
    Message(role="user", content="Write a short story about a robot in 3 sentences."),
  ]

  print("Streaming response (sync):")
  print("-" * 40)

  # Stream the response
  for chunk in model.invoke_stream(
    messages=messages,
    assistant_message=Message(role="assistant", content=""),
  ):
    # Print each chunk as it arrives
    if chunk.content:
      print(chunk.content, end="", flush=True)

  print("\n" + "-" * 40)


async def async_streaming():
  """Asynchronous streaming example."""
  model = OpenAIChat(id="gpt-4o-mini")

  messages = [
    Message(role="user", content="Count from 1 to 5, explaining each number."),
  ]

  print("\nStreaming response (async):")
  print("-" * 40)

  # Stream the response asynchronously
  async for chunk in model.ainvoke_stream(
    messages=messages,
    assistant_message=Message(role="assistant", content=""),
  ):
    if chunk.content:
      print(chunk.content, end="", flush=True)

  print("\n" + "-" * 40)


def collect_streamed_response():
  """Example of collecting streamed chunks into a complete response."""
  model = OpenAIChat(id="gpt-4o-mini")

  messages = [
    Message(role="user", content="What are the primary colors?"),
  ]

  print("\nCollecting streamed response:")
  print("-" * 40)

  # Collect all chunks
  full_content = []
  for chunk in model.invoke_stream(
    messages=messages,
    assistant_message=Message(role="assistant", content=""),
  ):
    if chunk.content:
      full_content.append(chunk.content)
      print(".", end="", flush=True)  # Progress indicator

  # Join all chunks
  complete_response = "".join(full_content)
  print(f"\n\nComplete response:\n{complete_response}")
  print("-" * 40)


if __name__ == "__main__":
  # Synchronous streaming
  sync_streaming()

  # Asynchronous streaming
  asyncio.run(async_streaming())

  # Collect streamed response
  collect_streamed_response()
