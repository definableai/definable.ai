"""
Streaming agent responses.

This example shows how to:
- Use run_stream() for synchronous streaming
- Process stream events as they arrive
- Handle different event types

Requirements:
    export OPENAI_API_KEY=sk-...
"""

from definable.agents import Agent
from definable.models.openai import OpenAIChat
from definable.tools.decorator import tool


@tool
def get_facts(topic: str) -> str:
  """Get interesting facts about a topic."""
  facts = {
    "python": "Python was created by Guido van Rossum in 1991.",
    "space": "The Sun is about 93 million miles from Earth.",
    "ocean": "The Pacific Ocean is the largest ocean on Earth.",
  }
  return facts.get(topic.lower(), f"No facts available for {topic}")


def basic_streaming():
  """Basic streaming example."""
  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    instructions="You are a helpful assistant. Give detailed responses.",
  )

  print("Streaming Response")
  print("-" * 40)

  # Stream the response
  for event in agent.run_stream("Explain the water cycle in 3 sentences."):
    # Print content as it arrives
    if hasattr(event, "content") and event.content:
      print(event.content, end="", flush=True)

  print("\n" + "-" * 40)


def streaming_with_progress():
  """Show streaming progress indicators."""
  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    instructions="You are a storyteller. Tell engaging stories.",
  )

  print("\nStreaming with Progress")
  print("-" * 40)

  char_count = 0
  for event in agent.run_stream("Tell me a very short story about a robot."):
    if hasattr(event, "content") and event.content:
      print(event.content, end="", flush=True)
      char_count += len(event.content)

  print(f"\n\n[Streamed {char_count} characters]")
  print("-" * 40)


def streaming_with_tools():
  """Streaming with tool usage."""
  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    tools=[get_facts],
    instructions="You are a knowledgeable assistant. Use the get_facts tool when asked about topics.",
  )

  print("\nStreaming with Tools")
  print("-" * 40)

  final_output = None
  for event in agent.run_stream("Tell me a fact about Python programming."):
    if hasattr(event, "content") and event.content:
      print(event.content, end="", flush=True)
    # Keep track of the final output
    final_output = event

  print("\n")

  # Access tool executions from the final output
  if final_output and hasattr(final_output, "tools") and final_output.tools:
    print("Tools used:")
    for execution in final_output.tools:
      print(f"  - {execution.tool_name}: {execution.result}")

  print("-" * 40)


def collect_streamed_response():
  """Collect streamed chunks into a complete response."""
  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    instructions="You are a helpful assistant.",
  )

  print("\nCollecting Streamed Response")
  print("-" * 40)

  # Collect all content chunks
  chunks = []
  final_output = None

  for event in agent.run_stream("What are the primary colors?"):
    if hasattr(event, "content") and event.content:
      chunks.append(event.content)
      print(".", end="", flush=True)  # Progress indicator
    final_output = event

  # Join chunks to get complete response
  complete_response = "".join(chunks)

  print(f"\n\nComplete response:\n{complete_response}")

  # Access final metrics if available
  if final_output and hasattr(final_output, "metrics") and final_output.metrics:
    print("\nMetrics:")
    print(f"  Input tokens: {final_output.metrics.input_tokens}")
    print(f"  Output tokens: {final_output.metrics.output_tokens}")

  print("-" * 40)


def streaming_multi_turn():
  """Streaming in a multi-turn conversation."""
  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    instructions="You are a helpful assistant. Remember our conversation.",
  )

  print("\nStreaming Multi-turn Conversation")
  print("-" * 40)

  messages = None

  prompts = [
    "My name is Charlie.",
    "What's my name?",
  ]

  for prompt in prompts:
    print(f"\nUser: {prompt}")
    print("Agent: ", end="")

    final_output = None
    for event in agent.run_stream(prompt, messages=messages):
      if hasattr(event, "content") and event.content:
        print(event.content, end="", flush=True)
      final_output = event

    print()

    # Update messages for next turn
    if final_output and hasattr(final_output, "messages"):
      messages = final_output.messages

  print("-" * 40)


if __name__ == "__main__":
  basic_streaming()
  streaming_with_progress()
  streaming_with_tools()
  collect_streamed_response()
  streaming_multi_turn()
