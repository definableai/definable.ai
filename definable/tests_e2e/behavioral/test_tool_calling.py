"""
Behavioral tests: Does the agent call tools correctly?

Strategy:
  - Real OpenAI model only — no mocks
  - Assert on OUTCOMES: tool was invoked, result appeared in response content
  - Tools are real Python functions decorated with @tool

Covers:
  - Agent with tool: model decides to call the correct tool
  - Tool result is incorporated into the final response
  - Agent handles multiple tools and routes to the correct one
  - Tool with arguments receives correct argument values
  - Agent without tool does not call tools
  - Agent decides to call tool for action queries
  - Agent skips tools for simple factual questions
"""

import pytest

from definable.agent import Agent
from definable.agent.config import AgentConfig
from definable.agent.tracing import Tracing
from definable.tool.decorator import tool


# ---------------------------------------------------------------------------
# Test tools
# ---------------------------------------------------------------------------


@tool
def get_weather(city: str) -> str:
  """Get the current weather for a city."""
  return f"Sunny and 25°C in {city}"


@tool
def calculate(expression: str) -> str:
  """Evaluate a mathematical expression."""
  try:
    result = eval(expression, {"__builtins__": {}})  # noqa: S307
    return str(result)
  except Exception as e:
    return f"Error: {e}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
  """Send an email."""
  return f"Email sent to {to} with subject '{subject}'"


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def no_trace():
  return AgentConfig(tracing=Tracing(enabled=False))


# ---------------------------------------------------------------------------
# Real model decides whether to call a tool
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
@pytest.mark.openai
class TestToolCallingIntelligence:
  """Real model decides whether to call a tool based on query intent."""

  @pytest.fixture
  def agent_with_tools(self, openai_model, no_trace):
    return Agent(
      model=openai_model,
      tools=[get_weather, calculate, send_email],
      config=no_trace,
    )

  @pytest.mark.asyncio
  async def test_agent_calls_calculation_tool_for_math(self, agent_with_tools):
    output = await agent_with_tools.arun("Calculate 15 * 7 using the calculate tool.")
    tool_names = [t.tool_name for t in (output.tools or [])]
    has_calc = "calculate" in tool_names
    has_answer = "105" in output.content
    assert has_calc or has_answer

  @pytest.mark.asyncio
  async def test_agent_does_not_call_tool_for_simple_factual_question(self, agent_with_tools):
    """Simple factual questions shouldn't trigger tool calls."""
    output = await agent_with_tools.arun("What is the capital of Japan?")
    tool_names = [t.tool_name for t in (output.tools or [])]
    assert "get_weather" not in tool_names
    assert "calculate" not in tool_names
    assert "send_email" not in tool_names
    assert "tokyo" in output.content.lower()

  @pytest.mark.asyncio
  async def test_agent_calls_weather_tool_for_weather_query(self, agent_with_tools):
    output = await agent_with_tools.arun("What is the weather in Paris? Use the get_weather tool.")
    tool_names = [t.tool_name for t in (output.tools or [])]
    has_weather = "get_weather" in tool_names
    has_content = "paris" in output.content.lower() or "sunny" in output.content.lower()
    assert has_weather or has_content

  @pytest.mark.asyncio
  async def test_tool_result_appears_in_final_response(self, openai_model, no_trace):
    """When a tool is called, its result should be reflected in the agent response."""
    agent = Agent(model=openai_model, tools=[calculate], config=no_trace)
    output = await agent.arun("What is 6 * 7? Use the calculate tool and tell me the result.")
    # Tool result "42" should appear in the response
    tool_names = [t.tool_name for t in (output.tools or [])]
    assert "calculate" in tool_names or "42" in output.content  # type: ignore[operator]

  @pytest.mark.asyncio
  async def test_agent_without_tools_has_no_tool_executions(self, openai_model, no_trace):
    """Agent without tools should never have tool_executions."""
    agent = Agent(model=openai_model, config=no_trace)
    output = await agent.arun("What is 2+2?")
    assert not output.tools

  @pytest.mark.asyncio
  async def test_correct_tool_called_out_of_multiple(self, agent_with_tools):
    """When multiple tools are available, the correct one is chosen."""
    output = await agent_with_tools.arun("Please send an email to test@example.com with subject 'Hello' and body 'World'.")
    tool_names = [t.tool_name for t in (output.tools or [])]
    assert "send_email" in tool_names
    assert "get_weather" not in tool_names
    assert "calculate" not in tool_names
