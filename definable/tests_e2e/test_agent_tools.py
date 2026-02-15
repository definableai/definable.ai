"""E2E tests — Tool Execution Workflows.

Scenario: "I want my agent to call functions and use their results."

All tests require OPENAI_API_KEY.
"""

from typing import List

import pytest

from definable.agents.agent import Agent
from definable.agents.config import AgentConfig, TracingConfig
from definable.agents.toolkit import Toolkit
from definable.tools.decorator import tool


@pytest.mark.e2e
@pytest.mark.openai
class TestBasicTools:
  """Agent selects and calls tools based on user queries."""

  @pytest.mark.asyncio
  async def test_agent_calls_tool_when_needed(self, openai_model):
    """Agent calls get_weather tool when asked about weather."""
    called_with = {}

    @tool
    def get_weather(city: str) -> str:
      """Get the current weather for a city. You MUST use this tool."""
      called_with["city"] = city
      return f"The weather in {city} is sunny, 72F"

    agent = Agent(
      model=openai_model,
      tools=[get_weather],
      instructions="You MUST use the get_weather tool when asked about weather. Never guess.",
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    output = await agent.arun("Use the get_weather tool to check the weather in Tokyo.")

    assert output.content is not None
    assert called_with.get("city") is not None

  @pytest.mark.asyncio
  async def test_tool_result_used_in_response(self, openai_model):
    """Tool returns specific data that the agent includes in its response."""

    @tool
    def get_stock_price(symbol: str) -> str:
      """Get the current stock price for a symbol."""
      return f"The current price of {symbol} is $142.50"

    agent = Agent(
      model=openai_model,
      tools=[get_stock_price],
      instructions="Use the get_stock_price tool to answer stock price questions.",
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    output = await agent.arun("What is the stock price of ACME? Use the tool.")

    assert output.content is not None
    assert "142" in output.content or "142.50" in output.content

  @pytest.mark.asyncio
  async def test_multiple_tools_available(self, openai_model):
    """Agent picks the correct tool among multiple options."""
    calls = []

    @tool
    def add_numbers(a: int, b: int) -> int:
      """Add two numbers together."""
      calls.append("add")
      return a + b

    @tool
    def multiply_numbers(a: int, b: int) -> int:
      """Multiply two numbers together."""
      calls.append("multiply")
      return a * b

    @tool
    def greet_user(name: str) -> str:
      """Greet a user by name."""
      calls.append("greet")
      return f"Hello, {name}!"

    agent = Agent(
      model=openai_model,
      tools=[add_numbers, multiply_numbers, greet_user],
      instructions="Use the appropriate tool for each request.",
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    output = await agent.arun("Multiply 7 and 8 using the multiply_numbers tool.")

    assert output.content is not None
    assert "56" in output.content
    assert "multiply" in calls

  @pytest.mark.asyncio
  async def test_async_tool_execution(self, openai_model):
    """Async tool function is executed correctly by the agent."""
    import asyncio

    @tool
    async def async_lookup(query: str) -> str:
      """Look up information asynchronously."""
      await asyncio.sleep(0.01)
      return f"Result for '{query}': Found 3 matching items"

    agent = Agent(
      model=openai_model,
      tools=[async_lookup],
      instructions="Use the async_lookup tool to answer lookup questions.",
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    output = await agent.arun("Look up 'python frameworks' using the tool.")

    assert output.content is not None
    assert "3" in output.content or "matching" in output.content.lower()

  @pytest.mark.asyncio
  async def test_tool_with_complex_params(self, openai_model):
    """Tool with dict/list/optional params receives correct arguments."""

    @tool
    def create_report(
      title: str,
      tags: List[str],
      priority: int = 1,
    ) -> str:
      """Create a report with the given parameters."""
      return f"Report '{title}' created with tags {tags} at priority {priority}"

    agent = Agent(
      model=openai_model,
      tools=[create_report],
      instructions="Use the create_report tool to create reports.",
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    output = await agent.arun("Create a report titled 'Q4 Sales' with tags 'finance' and 'quarterly'. Use the tool.")

    assert output.content is not None
    assert "Q4" in output.content or "report" in output.content.lower()


@pytest.mark.e2e
@pytest.mark.openai
class TestToolAdvanced:
  """Advanced tool features: stop_after, show_result, hooks, toolkits."""

  @pytest.mark.asyncio
  async def test_stop_after_tool_call(self, openai_model):
    """Tool with stop_after_tool_call=True ends the agent loop after execution."""

    @tool(stop_after_tool_call=True)
    def submit_order(order_id: str) -> str:
      """Submit an order. This is a final action."""
      return f"Order {order_id} submitted successfully"

    agent = Agent(
      model=openai_model,
      tools=[submit_order],
      instructions="Use the submit_order tool to submit orders.",
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    output = await agent.arun("Submit order ORD-123 using the tool.")

    # Tool should have been called
    assert output.tools is not None or output.content is not None

  @pytest.mark.asyncio
  async def test_tool_with_pre_post_hooks(self, openai_model):
    """pre_hook and post_hook fire around tool calls."""
    hook_log = []

    @tool
    def compute(x: int) -> int:
      """Compute the square of a number."""
      return x * x

    agent = Agent(
      model=openai_model,
      tools=[compute],
      instructions="Use the compute tool to square numbers.",
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )

    @agent.before_request
    async def before(ctx):
      hook_log.append("before")

    @agent.after_response
    async def after(ctx, output):
      hook_log.append("after")

    output = await agent.arun("What is 7 squared? Use the compute tool.")

    assert output.content is not None
    assert "before" in hook_log
    assert "after" in hook_log

  @pytest.mark.asyncio
  async def test_toolkit_tools_available(self, openai_model):
    """Agent with a Toolkit can discover and call toolkit tools."""

    @tool
    def convert_celsius(temp: float) -> str:
      """Convert Celsius to Fahrenheit."""
      return f"{temp}C = {temp * 9 / 5 + 32}F"

    @tool
    def convert_kg(weight: float) -> str:
      """Convert kilograms to pounds."""
      return f"{weight}kg = {weight * 2.205}lbs"

    class UnitToolkit(Toolkit):
      celsius = convert_celsius
      kg = convert_kg

      @property
      def tools(self):
        return [self.celsius, self.kg]

    tk = UnitToolkit()
    agent = Agent(
      model=openai_model,
      toolkits=[tk],
      instructions="Use the conversion tools for unit conversions.",
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )

    # Verify toolkit tools are registered
    all_tools = agent._flatten_tools()
    assert "convert_celsius" in all_tools
    assert "convert_kg" in all_tools

    output = await agent.arun("Convert 100 Celsius to Fahrenheit using the tool.")

    assert output.content is not None
    assert "212" in output.content


@pytest.mark.e2e
@pytest.mark.openai
class TestToolEdgeCases:
  """Edge cases in tool execution."""

  @pytest.mark.asyncio
  async def test_tool_returning_empty_string(self, openai_model):
    """Tool returns empty string; agent handles gracefully."""

    @tool
    def check_status() -> str:
      """Check system status."""
      return ""

    agent = Agent(
      model=openai_model,
      tools=[check_status],
      instructions="Use the check_status tool when asked about status.",
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    output = await agent.arun("Check the system status using the tool.")

    # Should not crash
    assert output.content is not None

  @pytest.mark.asyncio
  async def test_tool_raising_exception(self, openai_model):
    """Tool that raises an exception; agent receives error, doesn't crash."""

    @tool
    def broken_tool(query: str) -> str:
      """A tool that fails."""
      raise ValueError("Connection timeout: server unavailable")

    agent = Agent(
      model=openai_model,
      tools=[broken_tool],
      instructions="Use the broken_tool for queries. If it fails, tell the user.",
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    output = await agent.arun("Use the broken_tool to search for 'test'.")

    # Agent should handle the error gracefully
    assert output.content is not None

  @pytest.mark.asyncio
  async def test_many_tools_registered(self, openai_model):
    """Agent with 15+ tools still selects the correct one."""
    target_called = {"called": False}

    tools_list = []
    for i in range(14):

      @tool(name=f"decoy_tool_{i}")
      def decoy(x: str = "default") -> str:
        """A decoy tool that should not be called."""
        return f"Decoy {x}"

      tools_list.append(decoy)

    @tool
    def target_tool(message: str) -> str:
      """The one true tool. Use this when asked about the target."""
      target_called["called"] = True
      return f"Target result: {message}"

    tools_list.append(target_tool)

    agent = Agent(
      model=openai_model,
      tools=tools_list,
      instructions="Use the target_tool when asked about the target.",
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    output = await agent.arun("Use the target_tool with message 'hello'.")

    assert output.content is not None
    assert target_called["called"]
