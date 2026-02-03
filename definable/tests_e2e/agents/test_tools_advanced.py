"""Advanced E2E tests for tools - edge cases and complex scenarios."""

import pytest

from definable.agents.testing import AgentTestCase, MockModel
from definable.tools.decorator import tool
from definable.tools.function import Function


@pytest.mark.e2e
class TestToolsAdvanced(AgentTestCase):
    """Advanced tool tests covering edge cases."""

    def test_tool_with_optional_params(self):
        """Tool with optional parameters."""

        @tool
        def search(query: str, limit: int = 10, offset: int = 0) -> str:
            """Search with pagination."""
            return f"Query: {query}, Limit: {limit}, Offset: {offset}"

        agent = self.create_agent(tools=[search])
        search_tool = next(t for t in agent.tools if t.name == "search")

        # Check required vs optional
        required = search_tool.parameters.get("required", [])
        assert "query" in required
        assert "limit" not in required
        assert "offset" not in required

    def test_tool_with_list_param(self):
        """Tool with list parameter."""
        from typing import List

        @tool
        def process_items(items: List[str]) -> str:
            """Process a list of items."""
            return f"Processed {len(items)} items"

        agent = self.create_agent(tools=[process_items])
        tool_obj = next(t for t in agent.tools if t.name == "process_items")

        assert tool_obj.parameters is not None
        items_schema = tool_obj.parameters.get("properties", {}).get("items", {})
        assert items_schema.get("type") == "array"

    def test_tool_with_dict_param(self):
        """Tool with dict parameter."""
        from typing import Dict

        @tool
        def update_config(config: Dict[str, str]) -> str:
            """Update configuration."""
            return f"Updated {len(config)} settings"

        agent = self.create_agent(tools=[update_config])
        tool_obj = next(t for t in agent.tools if t.name == "update_config")

        assert tool_obj.parameters is not None

    def test_tool_with_enum_param(self):
        """Tool with enum-like parameter."""
        from typing import Literal

        @tool
        def set_priority(level: Literal["low", "medium", "high"]) -> str:
            """Set priority level."""
            return f"Priority set to {level}"

        agent = self.create_agent(tools=[set_priority])
        tool_obj = next(t for t in agent.tools if t.name == "set_priority")

        assert tool_obj.parameters is not None

    def test_tool_with_bool_param(self):
        """Tool with boolean parameter."""

        @tool
        def toggle_feature(enabled: bool) -> str:
            """Toggle a feature."""
            return f"Feature {'enabled' if enabled else 'disabled'}"

        result = toggle_feature.entrypoint(enabled=True)
        assert "enabled" in result

        result = toggle_feature.entrypoint(enabled=False)
        assert "disabled" in result

    def test_tool_with_float_param(self):
        """Tool with float parameter."""

        @tool
        def set_temperature(temp: float) -> str:
            """Set temperature value."""
            return f"Temperature set to {temp}"

        result = set_temperature.entrypoint(temp=0.7)
        assert "0.7" in result

    def test_tool_returning_dict(self):
        """Tool that returns a dictionary."""
        from typing import Dict, Any

        @tool
        def get_user_info(user_id: str) -> Dict[str, Any]:
            """Get user information."""
            return {"id": user_id, "name": "Test User", "active": True}

        result = get_user_info.entrypoint(user_id="123")
        assert isinstance(result, dict)
        assert result["id"] == "123"

    def test_tool_returning_list(self):
        """Tool that returns a list."""
        from typing import List

        @tool
        def get_recent_items(count: int) -> List[str]:
            """Get recent items."""
            return [f"item_{i}" for i in range(count)]

        result = get_recent_items.entrypoint(count=3)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_tool_with_docstring_params(self):
        """Tool extracts parameter descriptions from docstring."""

        @tool
        def complex_operation(
            input_data: str,
            max_retries: int = 3,
        ) -> str:
            """Perform a complex operation.

            Args:
                input_data: The input data to process.
                max_retries: Maximum number of retry attempts.

            Returns:
                The processed result.
            """
            return f"Processed: {input_data}"

        agent = self.create_agent(tools=[complex_operation])
        tool_obj = next(t for t in agent.tools if t.name == "complex_operation")

        # Description should be extracted
        assert "complex operation" in tool_obj.description.lower()

    def test_tool_custom_description(self):
        """Tool with custom description override."""

        @tool(description="A custom description that overrides the docstring.")
        def my_tool() -> str:
            """Original docstring."""
            return "result"

        agent = self.create_agent(tools=[my_tool])
        tool_obj = next(t for t in agent.tools if t.name == "my_tool")

        assert "custom description" in tool_obj.description.lower()

    def test_tool_with_none_return(self):
        """Tool that returns None."""

        @tool
        def void_action(message: str) -> None:
            """Perform an action with no return value."""
            pass  # Side effect only

        result = void_action.entrypoint(message="test")
        assert result is None

    def test_tool_exception_handling(self):
        """Tool that raises an exception."""

        @tool
        def failing_tool() -> str:
            """A tool that always fails."""
            raise ValueError("Intentional failure")

        # Direct call should raise
        with pytest.raises(ValueError, match="Intentional failure"):
            failing_tool.entrypoint()

    def test_tool_with_complex_nested_type(self):
        """Tool with complex nested type hints."""
        from typing import Dict, List, Optional

        @tool
        def process_data(
            items: List[Dict[str, str]],
            options: Optional[Dict[str, int]] = None,
        ) -> Dict[str, List[str]]:
            """Process complex nested data."""
            return {"processed": [item.get("name", "") for item in items]}

        result = process_data.entrypoint(
            items=[{"name": "a"}, {"name": "b"}],
            options={"limit": 10},
        )
        assert result == {"processed": ["a", "b"]}

    @pytest.mark.asyncio
    async def test_async_tool_execution(self):
        """Async tool can be executed."""
        import asyncio

        @tool
        async def async_fetch(url: str) -> str:
            """Async fetch operation."""
            await asyncio.sleep(0.01)  # Simulate async work
            return f"Fetched: {url}"

        # Async tool entrypoint returns coroutine
        coro = async_fetch.entrypoint(url="https://example.com")
        result = await coro
        assert "Fetched" in result

    def test_tool_with_stop_after_call(self):
        """Tool with stop_after_tool_call flag."""

        @tool(stop_after_tool_call=True)
        def final_action() -> str:
            """Action that stops the agent."""
            return "Done"

        agent = self.create_agent(tools=[final_action])
        tool_obj = next(t for t in agent.tools if t.name == "final_action")

        assert tool_obj.stop_after_tool_call is True

    def test_many_tools(self):
        """Agent can handle many tools."""
        tools = []
        for i in range(20):
            @tool(name=f"tool_{i}")
            def dynamic_tool(x: int = i) -> int:
                """Dynamic tool."""
                return x

            tools.append(dynamic_tool)

        agent = self.create_agent(tools=tools)
        assert len(agent.tools) == 20

    def test_tool_name_uniqueness(self):
        """Tools with same implementation but different names."""

        @tool(name="add_v1")
        def add1(a: int, b: int) -> int:
            """Add version 1."""
            return a + b

        @tool(name="add_v2")
        def add2(a: int, b: int) -> int:
            """Add version 2."""
            return a + b

        agent = self.create_agent(tools=[add1, add2])
        tool_names = [t.name for t in agent.tools]

        assert "add_v1" in tool_names
        assert "add_v2" in tool_names
        assert len(agent.tools) == 2
