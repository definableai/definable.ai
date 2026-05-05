"""
Complex parameter types for tools.

This example shows how to:
- Use optional parameters with defaults
- Use List, Dict, and other complex types
- Use Enum types for constrained choices
- Use nested types and Pydantic models

Requirements:
    export OPENAI_API_KEY=sk-...
"""

from enum import Enum
from typing import Dict, List, Optional

from definable.agent import Agent
from definable.model.openai import OpenAIChat
from definable.tool.decorator import tool
from pydantic import BaseModel, Field


# Tool with optional parameter
@tool
def search(query: str, limit: int = 10) -> str:
  """Search for items matching a query.

  Args:
      query: The search query string
      limit: Maximum number of results to return (default: 10)
  """
  return f"Found {limit} results for '{query}'"


# Tool with List parameter
@tool
def calculate_sum(numbers: List[int]) -> int:
  """Calculate the sum of a list of numbers."""
  return sum(numbers)


# Tool with Dict parameter
@tool
def format_user_info(user: Dict[str, str]) -> str:
  """Format user information into a readable string.

  Args:
      user: A dictionary with user info (name, email, etc.)
  """
  parts = [f"{key}: {value}" for key, value in user.items()]
  return "User Info:\n" + "\n".join(parts)


# Enum for constrained choices
class Priority(str, Enum):
  LOW = "low"
  MEDIUM = "medium"
  HIGH = "high"


# Tool with Enum parameter
@tool
def create_task(title: str, priority: Priority = Priority.MEDIUM) -> str:
  """Create a new task with a title and priority level.

  Args:
      title: The task title
      priority: Priority level (low, medium, or high)
  """
  return f"Created task '{title}' with {priority.value} priority"


# Enum for another example
class Operation(str, Enum):
  ADD = "add"
  SUBTRACT = "subtract"
  MULTIPLY = "multiply"
  DIVIDE = "divide"


@tool
def calculate(a: float, b: float, operation: Operation) -> float:
  """Perform a calculation with two numbers.

  Args:
      a: First number
      b: Second number
      operation: The operation to perform (add, subtract, multiply, divide)
  """
  if operation == Operation.ADD:
    return a + b
  elif operation == Operation.SUBTRACT:
    return a - b
  elif operation == Operation.MULTIPLY:
    return a * b
  elif operation == Operation.DIVIDE:
    if b == 0:
      raise ValueError("Cannot divide by zero")
    return a / b


# Tool with multiple optional parameters
@tool
def send_notification(
  message: str,
  recipient: Optional[str] = None,
  urgent: bool = False,
  channels: Optional[List[str]] = None,
) -> str:
  """Send a notification message.

  Args:
      message: The notification message
      recipient: Optional specific recipient (default: all)
      urgent: Whether this is urgent (default: False)
      channels: List of channels to send to (default: ["email"])
  """
  if channels is None:
    channels = ["email"]

  recipient_str = recipient or "all users"
  urgent_str = " [URGENT]" if urgent else ""
  return f"Sent{urgent_str} to {recipient_str} via {', '.join(channels)}: {message}"


# Pydantic model for complex nested types
class Address(BaseModel):
  street: str = Field(description="Street address")
  city: str = Field(description="City name")
  country: str = Field(description="Country name")
  postal_code: Optional[str] = Field(default=None, description="Postal/ZIP code")


class Person(BaseModel):
  name: str = Field(description="Person's full name")
  age: int = Field(description="Person's age")
  email: Optional[str] = Field(default=None, description="Email address")
  address: Optional[Address] = Field(default=None, description="Physical address")


@tool
def register_person(person: Person) -> str:
  """Register a new person in the system.

  Args:
      person: Person details including name, age, and optional contact info
  """
  result = f"Registered {person.name}, age {person.age}"
  if person.email:
    result += f", email: {person.email}"
  if person.address:
    result += f", location: {person.address.city}, {person.address.country}"
  return result


def main():
  """Demonstrate tools with complex parameters."""
  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    tools=[
      search,
      calculate_sum,
      format_user_info,
      create_task,
      calculate,
      send_notification,
      register_person,
    ],
    instructions="""You are a helpful assistant with access to various tools.
Use the appropriate tool and parameters to answer user questions.""",
  )

  print("Testing Complex Tool Parameters")
  print("=" * 50)

  # Test optional parameter
  output = agent.run("Search for 'python tutorials' and limit to 5 results")
  print(f"\nSearch with limit: {output.content}")

  # Test list parameter
  output = agent.run("Calculate the sum of 10, 20, 30, and 40")
  print(f"\nSum of list: {output.content}")

  # Test dict parameter
  output = agent.run("Format info for a user named John with email john@example.com")
  print(f"\nFormatted user: {output.content}")

  # Test enum parameter
  output = agent.run("Create a high priority task called 'Fix bug'")
  print(f"\nTask with enum: {output.content}")

  # Test calculation with enum
  output = agent.run("Multiply 15 by 4")
  print(f"\nCalculation: {output.content}")

  # Test multiple optional parameters
  output = agent.run("Send an urgent notification 'Server down!' to admin via email and slack")
  print(f"\nNotification: {output.content}")


if __name__ == "__main__":
  main()
