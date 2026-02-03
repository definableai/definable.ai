"""
Simple agent setup.

This example shows how to:
- Create a basic agent with a model
- Set agent instructions
- Run the agent and access outputs

Requirements:
    export OPENAI_API_KEY=sk-...
"""

from definable.agents import Agent
from definable.models.openai import OpenAIChat


def main():
  # Initialize the model
  model = OpenAIChat(id="gpt-4o-mini")

  # Create an agent with instructions
  agent = Agent(
    model=model,
    instructions="You are a helpful assistant that answers questions concisely.",
  )

  # Run the agent with a simple query
  output = agent.run("What is the capital of Japan?")

  # Access the response content
  print("Response:")
  print(output.content)
  print()

  # Access additional output information
  print("Output Details:")
  print(f"  Run ID: {output.run_id}")
  print(f"  Agent ID: {output.agent_id}")
  print(f"  Model: {output.model}")
  print(f"  Status: {output.status}")

  # Access metrics if available
  if output.metrics:
    print(f"  Input tokens: {output.metrics.input_tokens}")
    print(f"  Output tokens: {output.metrics.output_tokens}")


def agent_with_config():
  """Agent with custom configuration."""
  from definable.agents import Agent, AgentConfig

  model = OpenAIChat(id="gpt-4o-mini")

  # Create agent with configuration
  agent = Agent(
    model=model,
    instructions="You are a math tutor. Explain concepts clearly.",
    config=AgentConfig(
      agent_id="math-tutor",
      agent_name="Math Tutor",
      max_iterations=5,  # Limit tool call iterations
      max_retries=2,  # Retry on transient errors
    ),
  )

  output = agent.run("Explain what a derivative is in simple terms.")

  print("\nMath Tutor Response:")
  print(output.content)


def access_messages():
  """Access the full message history."""
  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    instructions="You are a helpful assistant.",
  )

  output = agent.run("Hello! My name is Alice.")

  print("\nMessage History:")
  if output.messages:
    for msg in output.messages:
      role = msg.role.upper()
      content = msg.content if isinstance(msg.content, str) else str(msg.content)
      # Truncate long content for display
      if len(content) > 100:
        content = content[:100] + "..."
      print(f"  [{role}]: {content}")


if __name__ == "__main__":
  main()
  agent_with_config()
  access_messages()
