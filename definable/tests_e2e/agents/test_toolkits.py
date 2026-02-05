"""E2E tests for toolkit registration and tool availability."""

import pytest

from definable.agents.testing import AgentTestCase
from definable.agents.toolkit import Toolkit
from definable.tools.decorator import tool


@pytest.mark.e2e
class TestToolkitsE2E(AgentTestCase):
  """End-to-end tests for toolkits."""

  def test_toolkit_tools_exposed(self):
    """Toolkit exposes its tools via .tools property."""

    @tool
    def add_func(a: int, b: int) -> int:
      """Add two numbers."""
      return a + b

    @tool
    def multiply_func(a: int, b: int) -> int:
      """Multiply two numbers."""
      return a * b

    class MathToolkit(Toolkit):
      add = add_func
      multiply = multiply_func

      @property
      def tools(self):
        return [self.add, self.multiply]

    tk = MathToolkit()
    assert len(tk.tools) == 2
    tool_names = [t.name for t in tk.tools]
    assert "add_func" in tool_names
    assert "multiply_func" in tool_names

  def test_toolkit_auto_discovery(self):
    """Toolkit auto-discovers Function attributes."""

    @tool
    def subtract(a: int, b: int) -> int:
      """Subtract two numbers."""
      return a - b

    @tool
    def divide(a: int, b: int) -> float:
      """Divide two numbers."""
      return a / b

    class AutoDiscoverToolkit(Toolkit):
      sub = subtract
      div = divide

    tk = AutoDiscoverToolkit()
    tools = tk.tools

    assert len(tools) == 2
    tool_names = [t.name for t in tools]
    assert "subtract" in tool_names
    assert "divide" in tool_names

  def test_toolkit_registration_on_agent(self, mock_model):
    """Toolkit tools are available on agent."""

    @tool
    def greet(name: str) -> str:
      """Greet by name."""
      return f"Hello, {name}!"

    @tool
    def farewell(name: str) -> str:
      """Say goodbye."""
      return f"Goodbye, {name}!"

    class GreetingToolkit(Toolkit):
      hello = greet
      bye = farewell

      @property
      def tools(self):
        return [self.hello, self.bye]

    tk = GreetingToolkit()
    agent = self.create_agent(model=mock_model, toolkits=[tk])

    # Flatten tools from toolkits
    all_tools = agent._flatten_tools()
    tool_names = list(all_tools.keys())

    assert "greet" in tool_names
    assert "farewell" in tool_names

  def test_toolkit_dependencies(self):
    """Toolkit dependencies are accessible."""

    class ConfiguredToolkit(Toolkit):
      def __init__(self, api_key: str):
        super().__init__(dependencies={"api_key": api_key})

    tk = ConfiguredToolkit(api_key="test-key-123")

    assert tk.dependencies["api_key"] == "test-key-123"

  def test_toolkit_name(self):
    """Toolkit name defaults to class name."""

    class MyCustomToolkit(Toolkit):
      pass

    tk = MyCustomToolkit()
    assert tk.name == "MyCustomToolkit"

  def test_multiple_toolkits(self, mock_model):
    """Multiple toolkits coexist on agent."""

    @tool
    def tool_a() -> str:
      """Tool A."""
      return "A"

    @tool
    def tool_b() -> str:
      """Tool B."""
      return "B"

    class ToolkitA(Toolkit):
      a = tool_a

      @property
      def tools(self):
        return [self.a]

    class ToolkitB(Toolkit):
      b = tool_b

      @property
      def tools(self):
        return [self.b]

    tk_a = ToolkitA()
    tk_b = ToolkitB()

    agent = self.create_agent(model=mock_model, toolkits=[tk_a, tk_b])
    all_tools = agent._flatten_tools()
    tool_names = list(all_tools.keys())

    assert "tool_a" in tool_names
    assert "tool_b" in tool_names

  def test_toolkit_and_direct_tools(self, mock_model):
    """Direct tools and toolkit tools can coexist."""

    @tool
    def direct_tool() -> str:
      """Direct tool."""
      return "direct"

    @tool
    def toolkit_tool() -> str:
      """Toolkit tool."""
      return "toolkit"

    class MyToolkit(Toolkit):
      tk_tool = toolkit_tool

      @property
      def tools(self):
        return [self.tk_tool]

    tk = MyToolkit()
    agent = self.create_agent(model=mock_model, tools=[direct_tool], toolkits=[tk])

    all_tools = agent._flatten_tools()
    tool_names = list(all_tools.keys())

    assert "direct_tool" in tool_names
    assert "toolkit_tool" in tool_names

  def test_toolkit_repr(self):
    """Toolkit has useful repr."""

    @tool
    def t1() -> str:
      """Tool 1."""
      return "1"

    @tool
    def t2() -> str:
      """Tool 2."""
      return "2"

    class TwoToolToolkit(Toolkit):
      tool1 = t1
      tool2 = t2

      @property
      def tools(self):
        return [self.tool1, self.tool2]

    tk = TwoToolToolkit()
    repr_str = repr(tk)

    assert "TwoToolToolkit" in repr_str
    assert "tools=2" in repr_str
