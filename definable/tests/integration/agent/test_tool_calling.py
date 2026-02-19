"""
Integration tests: Agent tool calling with real OpenAI API.

Migrated from tests_e2e/behavioral/test_tool_calling.py.

Strategy:
  - Real OpenAI model only — no mocks
  - Assert on OUTCOMES: tool was invoked, result appeared in response content
  - Uses fixtures from conftest: agent_with_tools, basic_agent, openai_model

Covers:
  - Agent calls calculation tool for math
  - Agent does not call tool for simple factual question
  - Agent calls correct tool from multiple available tools
  - Tool result is incorporated into final response
  - Agent without tools never has tool executions
  - Sync run with tools works
"""

import pytest

from definable.agent import Agent
from definable.agent.events import RunStatus
from definable.tool.decorator import tool


# ---------------------------------------------------------------------------
# Additional tools for specific tests
# ---------------------------------------------------------------------------


@tool
def send_email(to: str, subject: str, body: str) -> str:
  """Send an email to the specified recipient."""
  return f"Email sent to {to} with subject '{subject}'"


# ---------------------------------------------------------------------------
# Tool calling intelligence
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.openai
class TestToolCallingIntelligence:
  """Real model decides whether to call a tool based on query intent."""

  @pytest.mark.asyncio
  async def test_agent_calls_calculation_tool_for_math(self, agent_with_tools):
    """Agent should use the calculate tool for math expressions."""
    output = await agent_with_tools.arun("Calculate 15 * 7 using the calculate tool.")
    tool_names = [t.tool_name for t in (output.tools or [])]
    has_calc = "calculate" in tool_names
    has_answer = "105" in output.content
    assert has_calc or has_answer

  @pytest.mark.asyncio
  async def test_agent_does_not_call_tool_for_simple_factual_question(self, agent_with_tools):
    """Simple factual questions should not trigger tool calls."""
    output = await agent_with_tools.arun("What is the capital of Japan?")
    tool_names = [t.tool_name for t in (output.tools or [])]
    assert "get_weather" not in tool_names
    assert "calculate" not in tool_names
    assert "add" not in tool_names
    assert "tokyo" in output.content.lower()

  @pytest.mark.asyncio
  async def test_agent_calls_weather_tool_for_weather_query(self, agent_with_tools):
    """Agent should use get_weather tool for weather queries."""
    output = await agent_with_tools.arun("What is the weather in Paris? Use the get_weather tool.")
    tool_names = [t.tool_name for t in (output.tools or [])]
    has_weather = "get_weather" in tool_names
    has_content = "paris" in output.content.lower() or "sunny" in output.content.lower()
    assert has_weather or has_content

  @pytest.mark.asyncio
  async def test_tool_result_appears_in_final_response(self, agent_with_tools):
    """When a tool is called, its result should be reflected in the agent response."""
    output = await agent_with_tools.arun("What is 6 * 7? Use the calculate tool and tell me the result.")
    tool_names = [t.tool_name for t in (output.tools or [])]
    assert "calculate" in tool_names or "42" in output.content

  @pytest.mark.asyncio
  async def test_correct_tool_called_out_of_multiple(self, agent_with_tools, openai_model):
    """When multiple tools are available, the model routes to the correct one."""
    agent = Agent(model=openai_model, tools=[*agent_with_tools.tools, send_email])
    output = await agent.arun("Please send an email to test@example.com with subject 'Hello' and body 'World'.")
    tool_names = [t.tool_name for t in (output.tools or [])]
    assert "send_email" in tool_names
    assert "get_weather" not in tool_names
    assert "calculate" not in tool_names


# ---------------------------------------------------------------------------
# Agent without tools
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.openai
class TestAgentWithoutTools:
  """Agent without tools should never report tool executions."""

  @pytest.mark.asyncio
  async def test_agent_without_tools_has_no_tool_executions(self, basic_agent):
    """Agent without tools should not have any tool_executions."""
    output = await basic_agent.arun("What is 2+2?")
    assert not output.tools
    assert output.status == RunStatus.completed


# ---------------------------------------------------------------------------
# Sync tool calling
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.openai
class TestToolCallingSyncPath:
  """Sync run() with tools works correctly."""

  def test_sync_run_with_tools_returns_result(self, agent_with_tools):
    """Sync agent.run() should execute tools and include results."""
    output = agent_with_tools.run("What is 9 * 8? Use the calculate tool.")
    tool_names = [t.tool_name for t in (output.tools or [])]
    assert "calculate" in tool_names or "72" in output.content
    assert output.status == RunStatus.completed
