"""
Building a custom Toolkit class.

This example shows how to:
- Create a custom Toolkit subclass
- Define multiple related tools
- Expose tools via the tools property
- Use the toolkit with an agent

Requirements:
    export OPENAI_API_KEY=sk-...
"""

from typing import List

from definable.agents import Agent, Toolkit
from definable.models.openai import OpenAIChat
from definable.tools.decorator import tool

# ========================================
# Calculator Toolkit
# ========================================
# Define tools as standalone functions OUTSIDE the class


@tool
def calc_add(a: float, b: float) -> float:
  """Add two numbers."""
  return a + b


@tool
def calc_subtract(a: float, b: float) -> float:
  """Subtract b from a."""
  return a - b


@tool
def calc_multiply(a: float, b: float) -> float:
  """Multiply two numbers."""
  return a * b


@tool
def calc_divide(a: float, b: float) -> float:
  """Divide a by b."""
  if b == 0:
    raise ValueError("Cannot divide by zero")
  return a / b


@tool
def calc_power(base: float, exponent: float) -> float:
  """Raise base to the power of exponent."""
  return base**exponent


@tool
def calc_sqrt(number: float) -> float:
  """Calculate the square root of a number."""
  if number < 0:
    raise ValueError("Cannot calculate square root of negative number")
  return number**0.5


class CalculatorToolkit(Toolkit):
  """A toolkit for mathematical calculations.

  Pattern: Define @tool functions outside the class,
  then assign them as class attributes.
  """

  # Assign tools as class attributes
  add = calc_add
  subtract = calc_subtract
  multiply = calc_multiply
  divide = calc_divide
  power = calc_power
  sqrt = calc_sqrt

  @property
  def tools(self) -> List:
    """Return all calculator tools."""
    return [
      self.add,
      self.subtract,
      self.multiply,
      self.divide,
      self.power,
      self.sqrt,
    ]


# ========================================
# String Toolkit
# ========================================


@tool
def str_uppercase(text: str) -> str:
  """Convert text to uppercase."""
  return text.upper()


@tool
def str_lowercase(text: str) -> str:
  """Convert text to lowercase."""
  return text.lower()


@tool
def str_reverse(text: str) -> str:
  """Reverse the text."""
  return text[::-1]


@tool
def str_word_count(text: str) -> int:
  """Count the number of words in the text."""
  return len(text.split())


@tool
def str_char_count(text: str) -> int:
  """Count the number of characters in the text."""
  return len(text)


class StringToolkit(Toolkit):
  """A toolkit for string manipulation."""

  uppercase = str_uppercase
  lowercase = str_lowercase
  reverse = str_reverse
  word_count = str_word_count
  char_count = str_char_count

  @property
  def tools(self) -> List:
    """Return all string tools."""
    return [
      self.uppercase,
      self.lowercase,
      self.reverse,
      self.word_count,
      self.char_count,
    ]


def basic_toolkit_usage():
  """Basic usage of a custom toolkit."""
  model = OpenAIChat(id="gpt-4o-mini")

  # Create agent with calculator toolkit
  agent = Agent(
    model=model,
    toolkits=[CalculatorToolkit()],
    instructions="You are a math assistant. Use the calculator tools to solve problems.",
  )

  print("Calculator Toolkit")
  print("=" * 50)

  output = agent.run("Calculate (15 + 5) * 3")
  print(f"Result: {output.content}")

  output = agent.run("What is the square root of 144?")
  print(f"Result: {output.content}")


def multiple_toolkits():
  """Using multiple toolkits together."""
  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    toolkits=[CalculatorToolkit(), StringToolkit()],
    instructions="""You are a versatile assistant with access to:
1. Calculator tools for math operations
2. String tools for text manipulation

Use the appropriate tools to help the user.""",
  )

  print("\n" + "=" * 50)
  print("Multiple Toolkits")
  print("=" * 50)

  output = agent.run("Convert 'Hello World' to uppercase and count its characters")
  print(f"Result: {output.content}")

  output = agent.run("Calculate 2 to the power of 10")
  print(f"Result: {output.content}")


def toolkit_info():
  """Display toolkit information."""
  print("\n" + "=" * 50)
  print("Toolkit Information")
  print("=" * 50)

  calc = CalculatorToolkit()
  string = StringToolkit()

  print(f"\n{calc.name}:")
  print(f"  Tools: {len(calc.tools)}")
  for t in calc.tools:
    print(f"    - {t.name}: {t.description}")

  print(f"\n{string.name}:")
  print(f"  Tools: {len(string.tools)}")
  for t in string.tools:
    print(f"    - {t.name}: {t.description}")


if __name__ == "__main__":
  basic_toolkit_usage()
  multiple_toolkits()
  toolkit_info()
