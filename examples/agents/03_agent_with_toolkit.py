"""
Agent with custom Toolkit class.

This example shows how to:
- Create a custom Toolkit class
- Group related tools together
- Use multiple toolkits with an agent

Requirements:
    export OPENAI_API_KEY=sk-...
"""

from typing import List

from definable.agent import Agent, Toolkit
from definable.model.openai import OpenAIChat
from definable.tool.decorator import tool


# Define tools as standalone functions OUTSIDE the class
@tool
def add(a: float, b: float) -> float:
  """Add two numbers."""
  return a + b


@tool
def subtract(a: float, b: float) -> float:
  """Subtract b from a."""
  return a - b


@tool
def multiply(a: float, b: float) -> float:
  """Multiply two numbers."""
  return a * b


@tool
def divide(a: float, b: float) -> float:
  """Divide a by b."""
  if b == 0:
    raise ValueError("Cannot divide by zero")
  return a / b


class MathToolkit(Toolkit):
  """A toolkit for mathematical operations.

  Tools are defined outside the class and assigned as class attributes.
  """

  # Assign tools as class attributes
  add_tool = add
  subtract_tool = subtract
  multiply_tool = multiply
  divide_tool = divide

  @property
  def tools(self) -> List:
    """Return the list of tools in this toolkit."""
    return [self.add_tool, self.subtract_tool, self.multiply_tool, self.divide_tool]


# Define string tools
@tool
def uppercase(text: str) -> str:
  """Convert text to uppercase."""
  return text.upper()


@tool
def lowercase(text: str) -> str:
  """Convert text to lowercase."""
  return text.lower()


@tool
def reverse(text: str) -> str:
  """Reverse the text."""
  return text[::-1]


@tool
def word_count(text: str) -> int:
  """Count the number of words in the text."""
  return len(text.split())


class StringToolkit(Toolkit):
  """A toolkit for string manipulation."""

  upper = uppercase
  lower = lowercase
  rev = reverse
  count = word_count

  @property
  def tools(self) -> List:
    """Return all string tools."""
    return [self.upper, self.lower, self.rev, self.count]


def basic_toolkit_usage():
  """Basic example with a single toolkit."""
  model = OpenAIChat(id="gpt-4o-mini")

  # Create agent with a toolkit
  agent = Agent(
    model=model,
    toolkits=[MathToolkit()],
    instructions="You are a math assistant. Use the provided tools for calculations.",
  )

  output = agent.run("What is 10 + 5, then multiply the result by 3?")

  print("Math Toolkit Response:")
  print(output.content)

  if output.tools:
    print("\nOperations performed:")
    for execution in output.tools:
      print(f"  {execution.tool_name}({execution.tool_args}) = {execution.result}")


def multiple_toolkits():
  """Example with multiple toolkits."""
  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    toolkits=[MathToolkit(), StringToolkit()],
    instructions="""You are a versatile assistant with access to:
1. Math tools for calculations
2. String tools for text manipulation

Use the appropriate tools to help the user.""",
  )

  print("\n" + "=" * 50)
  print("Multiple Toolkits")
  print("=" * 50)

  output = agent.run("Convert 'Hello World' to uppercase and count its words")
  print(f"Result: {output.content}")


def toolkit_with_dependencies():
  """Toolkit with shared configuration."""
  print("\n" + "=" * 50)
  print("Toolkit with Dependencies")
  print("=" * 50)

  class ConfiguredToolkit(Toolkit):
    """A toolkit with configuration dependencies."""

    def __init__(self, prefix: str = "Result"):
      super().__init__(dependencies={"prefix": prefix})
      self._prefix = prefix

    @property
    def tools(self) -> List:
      return [self.format_result]

    # Note: For stateful tools that need access to self,
    # you would typically create the tool dynamically
    format_result = None

  # For this pattern, tools are assigned as class attributes
  # The dependencies are available via self._dependencies

  model = OpenAIChat(id="gpt-4o-mini")
  agent = Agent(
    model=model,
    toolkits=[MathToolkit()],
    instructions="You are a helpful math assistant.",
  )

  output = agent.run("What is 7 times 8?")
  print(f"Result: {output.content}")


if __name__ == "__main__":
  basic_toolkit_usage()
  multiple_toolkits()
  toolkit_with_dependencies()
