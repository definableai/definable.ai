"""
Example: Enabling the observability dashboard on an agent.

This example shows how to:
1. Enable observability with a single flag
2. Add tools for the agent to use
3. Serve the agent with the dashboard at /obs/

Run:
    .venv/bin/python definable/examples/observability/01_basic_dashboard.py

Then open http://localhost:8000/obs/ in your browser.
Run some agent calls (the serve endpoint handles them) and watch events
flow into the dashboard in real time.
"""

from definable.agent import Agent
from definable.agent.interface import CLIInterface
from definable.model.openai import OpenAIChat
from definable.tool.decorator import tool


@tool
def get_weather(city: str) -> str:
  """Get the current weather for a city."""
  return f"The weather in {city} is 72°F and sunny."


@tool
def calculate(expression: str) -> str:
  """Evaluate a math expression."""
  return str(eval(expression))  # noqa: S307


# Create an agent with observability enabled
agent = Agent(
  model=OpenAIChat(id="gpt-4o-mini"),
  tools=[get_weather, calculate],
  instructions="You are a helpful assistant with access to weather and calculator tools.",
  observability=True,  # <-- single flag enables the dashboard
)

# Serve the agent — dashboard is at /obs/
agent.serve(CLIInterface(mode="repl"), enable_server=True, port=8002)
