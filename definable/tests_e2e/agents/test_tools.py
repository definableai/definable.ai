"""E2E tests for tool execution and tool-call handling."""

import pytest

from definable.agents.testing import AgentTestCase
from definable.tools.decorator import tool


@pytest.mark.e2e
class TestToolsE2E(AgentTestCase):
  """End-to-end tests for tools."""

  def test_tool_registration(self):
    """Tool decorated with @tool is available on agent."""

    @tool
    def greet(name: str) -> str:
      """Greet someone by name."""
      return f"Hello, {name}!"

    agent = self.create_agent(tools=[greet])
    tool_names = [t.name for t in agent.tools]

    assert "greet" in tool_names

  def test_tool_with_description(self):
    """Tool description is correctly set from docstring."""

    @tool
    def calculate(x: int, y: int) -> int:
      """Add two numbers together."""
      return x + y

    agent = self.create_agent(tools=[calculate])
    calc_tool = next(t for t in agent.tools if t.name == "calculate")

    assert "Add two numbers" in calc_tool.description

  def test_tool_with_custom_name(self):
    """Tool can have a custom name."""

    @tool(name="custom_add")
    def add(x: int, y: int) -> int:
      """Add two numbers."""
      return x + y

    agent = self.create_agent(tools=[add])
    tool_names = [t.name for t in agent.tools]

    assert "custom_add" in tool_names
    assert "add" not in tool_names

  def test_multiple_tools(self):
    """Multiple tools are all registered."""

    @tool
    def tool_a() -> str:
      """First tool."""
      return "A"

    @tool
    def tool_b() -> str:
      """Second tool."""
      return "B"

    @tool
    def tool_c() -> str:
      """Third tool."""
      return "C"

    agent = self.create_agent(tools=[tool_a, tool_b, tool_c])
    tool_names = [t.name for t in agent.tools]

    assert len(agent.tools) == 3
    assert "tool_a" in tool_names
    assert "tool_b" in tool_names
    assert "tool_c" in tool_names

  def test_agent_run_with_tool(self, mock_model):
    """Agent runs successfully with tools available."""

    @tool
    def get_weather(city: str) -> str:
      """Get weather for a city."""
      return f"Sunny in {city}"

    agent = self.create_agent(model=mock_model, tools=[get_weather])
    output = agent.run("What's the weather?")

    self.assert_has_content(output)

  def test_tool_parameters_extracted(self):
    """Tool parameters are correctly extracted from function signature."""

    @tool
    def search(query: str, limit: int = 10) -> str:
      """Search for something."""
      return f"Results for {query}"

    agent = self.create_agent(tools=[search])
    search_tool = next(t for t in agent.tools if t.name == "search")

    # Check parameters are defined
    assert search_tool.parameters is not None
    param_names = list(search_tool.parameters.get("properties", {}).keys())

    assert "query" in param_names
    assert "limit" in param_names

  def test_tool_direct_call(self):
    """Tool can be called directly (not through agent)."""

    @tool
    def multiply(a: int, b: int) -> int:
      """Multiply two numbers."""
      return a * b

    # Function object should be callable
    result = multiply.entrypoint(a=3, b=4)
    assert result == 12

  @pytest.mark.asyncio
  async def test_async_tool(self):
    """Async tool function is properly wrapped."""

    @tool
    async def async_fetch(url: str) -> str:
      """Fetch data from URL."""
      return f"Data from {url}"

    agent = self.create_agent(tools=[async_fetch])
    tool_names = [t.name for t in agent.tools]

    assert "async_fetch" in tool_names
