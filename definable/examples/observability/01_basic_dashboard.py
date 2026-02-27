from definable.agent import Agent
from definable.agent.interface import CLIInterface
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
  model="openai/gpt-5.2",
  tools=[get_weather, calculate],
  instructions="You are a helpful assistant with access to weather and calculator tools.",
  observability=True,  # <-- single flag enables the dashboard
)

# Serve the agent — dashboard is at /obs/
agent.serve(CLIInterface(mode="repl"), enable_server=True, port=8002)
