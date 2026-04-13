"""
Agent integration conftest — agent-specific fixtures.

Provides:
  - basic_agent: minimal agent with real OpenAI model
  - agent_with_tools: agent with sample tools
  - Common tool definitions reusable across agent tests
"""

import pytest

from definable.agent import Agent
from definable.tool.decorator import tool


# ---------------------------------------------------------------------------
# Sample tools for integration tests
# ---------------------------------------------------------------------------


@tool
def add(a: int, b: int) -> int:
  """Add two numbers together."""
  return a + b


@tool
def get_weather(city: str) -> str:
  """Get the current weather for a city."""
  return f"The weather in {city} is sunny and 72F."


@tool
def calculate(expression: str) -> str:
  """Evaluate a mathematical expression.

  Args:
    expression: A mathematical expression to evaluate (e.g. '2 + 3 * 4')
  """
  try:
    result = eval(expression)  # noqa: S307
    return str(result)
  except Exception as e:
    return f"Error: {e}"


# ---------------------------------------------------------------------------
# Agent fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def basic_agent(openai_model):
  """Minimal agent with a real OpenAI model (gpt-4o-mini), no tools."""
  return Agent(model=openai_model)


@pytest.fixture
def agent_with_tools(openai_model):
  """Agent with sample tools (add, get_weather, calculate)."""
  return Agent(model=openai_model, tools=[add, get_weather, calculate])
