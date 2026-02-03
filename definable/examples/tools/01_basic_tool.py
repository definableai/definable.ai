"""
Basic tool definition with @tool decorator.

This example shows how to:
- Create a simple tool using the @tool decorator
- Use type hints for parameters
- Write docstrings for tool descriptions
- Pass tools to an agent

Requirements:
    export OPENAI_API_KEY=sk-...
"""

from definable.agents import Agent
from definable.models.openai import OpenAIChat
from definable.tools.decorator import tool


# Basic tool with docstring as description
@tool
def greet(name: str) -> str:
  """Greet a person by name."""
  return f"Hello, {name}! Nice to meet you."


# Tool with return type annotation
@tool
def add_numbers(a: int, b: int) -> int:
  """Add two integers together and return the result."""
  return a + b


# Tool with more complex logic
@tool
def calculate_area(length: float, width: float) -> float:
  """Calculate the area of a rectangle given its length and width."""
  return length * width


# Tool with string processing
@tool
def reverse_string(text: str) -> str:
  """Reverse the characters in a string."""
  return text[::-1]


# Tool with conditional logic
@tool
def is_prime(n: int) -> bool:
  """Check if a number is prime."""
  if n < 2:
    return False
  for i in range(2, int(n**0.5) + 1):
    if n % i == 0:
      return False
  return True


def main():
  """Demonstrate basic tool usage."""
  model = OpenAIChat(id="gpt-4o-mini")

  # Create agent with all our tools
  agent = Agent(
    model=model,
    tools=[greet, add_numbers, calculate_area, reverse_string, is_prime],
    instructions="""You are a helpful assistant with access to various tools.
Use the appropriate tool to answer user questions.""",
  )

  # Test each tool
  print("Testing Basic Tools")
  print("=" * 50)

  # Test greet
  output = agent.run("Greet someone named Alice")
  print(f"\nGreeting: {output.content}")

  # Test add_numbers
  output = agent.run("What is 42 plus 58?")
  print(f"\nAddition: {output.content}")

  # Test calculate_area
  output = agent.run("What is the area of a rectangle that is 5.5 meters long and 3.2 meters wide?")
  print(f"\nArea calculation: {output.content}")

  # Test reverse_string
  output = agent.run("Reverse the word 'Python'")
  print(f"\nReversed string: {output.content}")

  # Test is_prime
  output = agent.run("Is 17 a prime number? What about 18?")
  print(f"\nPrime check: {output.content}")


def show_tool_metadata():
  """Show the metadata extracted from tool functions."""
  print("\n" + "=" * 50)
  print("Tool Metadata")
  print("=" * 50)

  # The @tool decorator creates a Function object with metadata
  print(f"\nTool name: {greet.name}")
  print(f"Tool description: {greet.description}")
  print(f"Tool parameters: {greet.parameters}")


if __name__ == "__main__":
  main()
  show_tool_metadata()
