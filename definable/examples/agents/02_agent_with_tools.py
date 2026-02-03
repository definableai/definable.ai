"""
Agent with tools using @tool decorator.

This example shows how to:
- Define tools using the @tool decorator
- Pass tools to an agent
- Handle tool execution in responses

Requirements:
    export OPENAI_API_KEY=sk-...
"""

from datetime import datetime

from definable.agents import Agent
from definable.models.openai import OpenAIChat
from definable.tools.decorator import tool


# Define tools using the @tool decorator
@tool
def add(a: int, b: int) -> int:
  """Add two numbers together."""
  return a + b


@tool
def multiply(a: int, b: int) -> int:
  """Multiply two numbers together."""
  return a * b


@tool
def get_current_time() -> str:
  """Get the current date and time."""
  return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def get_weather(city: str) -> str:
  """Get the current weather for a city (mock implementation)."""
  # In a real application, this would call a weather API
  weather_data = {
    "new york": "Sunny, 72°F",
    "london": "Cloudy, 58°F",
    "tokyo": "Rainy, 65°F",
    "paris": "Partly cloudy, 68°F",
  }
  return weather_data.get(city.lower(), f"Weather data not available for {city}")


def basic_tool_usage():
  """Basic example of using tools with an agent."""
  model = OpenAIChat(id="gpt-4o-mini")

  # Create agent with tools
  agent = Agent(
    model=model,
    tools=[add, multiply, get_current_time],
    instructions="You are a helpful assistant with access to math and time tools.",
  )

  # Run a query that requires tool usage
  output = agent.run("What is 15 times 7, and what time is it now?")

  print("Response:")
  print(output.content)
  print()

  # Access tool execution details
  if output.tools:
    print("Tools Used:")
    for execution in output.tools:
      print(f"  - {execution.tool_name}({execution.tool_args})")
      print(f"    Result: {execution.result}")


def weather_agent():
  """Agent that can check weather."""
  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    tools=[get_weather],
    instructions="You are a weather assistant. Help users check the weather in different cities.",
  )

  output = agent.run("What's the weather like in Tokyo and London?")

  print("\nWeather Response:")
  print(output.content)


def calculator_agent():
  """Agent that performs calculations."""
  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    tools=[add, multiply],
    instructions="""You are a calculator assistant.
Use the add and multiply tools to solve math problems.
Always show your work by explaining which operations you performed.""",
  )

  output = agent.run("Calculate (5 + 3) * 4")

  print("\nCalculator Response:")
  print(output.content)

  # Show the tool calls made
  if output.tools:
    print("\nCalculation Steps:")
    for i, execution in enumerate(output.tools, 1):
      print(f"  {i}. {execution.tool_name}({execution.tool_args}) = {execution.result}")


@tool
def search_database(query: str, limit: int = 10) -> str:
  """Search a mock database for records matching the query."""
  # Mock database search
  results = [
    {"id": 1, "name": "Product A", "price": 29.99},
    {"id": 2, "name": "Product B", "price": 49.99},
    {"id": 3, "name": "Product C", "price": 19.99},
  ]
  return f"Found {len(results)} results for '{query}': {results[:limit]}"


def tool_with_optional_params():
  """Tool with optional parameters."""
  model = OpenAIChat(id="gpt-4o-mini")

  agent = Agent(
    model=model,
    tools=[search_database],
    instructions="You are a database assistant. Search for products when asked.",
  )

  output = agent.run("Search for electronics in the database, show me the top 2 results.")

  print("\nDatabase Search Response:")
  print(output.content)


if __name__ == "__main__":
  basic_tool_usage()
  weather_agent()
  calculator_agent()
  tool_with_optional_params()
