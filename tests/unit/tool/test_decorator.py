"""
Unit tests for the @tool decorator.

Tests pure decorator behavior: wrapping sync/async functions into Function
instances, forwarding kwargs, auto-setting linked fields, and rejecting
invalid arguments. No API calls. No external dependencies.

Covers:
  - @tool on sync function returns Function
  - @tool on async function returns Function
  - @tool() with no args behaves like @tool
  - @tool(name="custom") sets Function.name
  - @tool(description="desc") sets Function.description
  - @tool(stop_after_tool_call=True) auto-sets show_result=True
  - @tool(requires_confirmation=True) sets the flag
  - @tool(requires_user_input=True) sets user_input_fields=[]
  - Type hints produce correct parameters schema
  - Function.entrypoint is callable
  - Docstring becomes Function.description
  - @tool(strict=True) makes all fields required
  - @tool(cache_results=True) sets Function.cache_results
  - Invalid kwargs raises ValueError
"""

from typing import List, Optional

import pytest

from definable.tool.decorator import tool
from definable.tool.function import Function


@pytest.mark.unit
class TestToolDecoratorOnSyncFunction:
  """@tool applied to a sync function produces a Function instance."""

  def test_returns_function_instance(self):
    @tool
    def greet(name: str) -> str:
      """Say hello."""
      return f"Hello, {name}"

    assert isinstance(greet, Function)

  def test_entrypoint_is_callable(self):
    @tool
    def add(a: int, b: int) -> int:
      """Add two numbers."""
      return a + b

    assert callable(add.entrypoint)

  def test_name_defaults_to_function_name(self):
    @tool
    def my_tool() -> str:
      """A tool."""
      return "ok"

    assert my_tool.name == "my_tool"

  def test_docstring_becomes_description(self):
    @tool
    def search(query: str) -> str:
      """Search the web for results."""
      return query

    assert search.description == "Search the web for results."


@pytest.mark.unit
class TestToolDecoratorOnAsyncFunction:
  """@tool applied to an async function produces a Function instance."""

  def test_returns_function_instance(self):
    @tool
    async def fetch(url: str) -> str:
      """Fetch a URL."""
      return url

    assert isinstance(fetch, Function)

  def test_name_defaults_to_function_name(self):
    @tool
    async def async_search(query: str) -> str:
      """Async search."""
      return query

    assert async_search.name == "async_search"


@pytest.mark.unit
class TestToolDecoratorCallSyntax:
  """@tool() with no args behaves identically to @tool."""

  def test_bare_call_returns_function(self):
    @tool()
    def ping() -> str:
      """Ping."""
      return "pong"

    assert isinstance(ping, Function)
    assert ping.name == "ping"


@pytest.mark.unit
class TestToolDecoratorCustomName:
  """@tool(name="custom") overrides the function name."""

  def test_custom_name(self):
    @tool(name="custom_tool")
    def boring_name() -> str:
      """Do something."""
      return "done"

    assert boring_name.name == "custom_tool"


@pytest.mark.unit
class TestToolDecoratorCustomDescription:
  """@tool(description="desc") overrides the docstring-based description."""

  def test_custom_description(self):
    @tool(description="My custom description")
    def some_tool() -> str:
      """Original docstring."""
      return "result"

    assert some_tool.description == "My custom description"


@pytest.mark.unit
class TestToolDecoratorStopAfterToolCall:
  """@tool(stop_after_tool_call=True) auto-sets show_result=True."""

  def test_stop_after_tool_call_sets_show_result(self):
    @tool(stop_after_tool_call=True)
    def final_tool() -> str:
      """The last tool."""
      return "done"

    assert final_tool.stop_after_tool_call is True
    assert final_tool.show_result is True

  def test_show_result_not_overridden_when_explicit_false(self):
    @tool(stop_after_tool_call=True, show_result=False)
    def quiet_tool() -> str:
      """Quiet stop."""
      return "done"

    assert quiet_tool.stop_after_tool_call is True
    assert quiet_tool.show_result is False


@pytest.mark.unit
class TestToolDecoratorRequiresConfirmation:
  """@tool(requires_confirmation=True) sets the flag on Function."""

  def test_requires_confirmation_is_true(self):
    @tool(requires_confirmation=True)
    def danger(target: str) -> str:
      """Dangerous operation."""
      return target

    assert danger.requires_confirmation is True


@pytest.mark.unit
class TestToolDecoratorRequiresUserInput:
  """@tool(requires_user_input=True) sets user_input_fields to empty list."""

  def test_requires_user_input_sets_fields_list(self):
    @tool(requires_user_input=True)
    def interactive(prompt: str) -> str:
      """Interactive tool."""
      return prompt

    assert interactive.requires_user_input is True
    assert interactive.user_input_fields == []


@pytest.mark.unit
class TestToolDecoratorParametersSchema:
  """Type hints on the decorated function produce correct JSON schema parameters."""

  def test_string_param_in_schema(self):
    @tool
    def lookup(name: str) -> str:
      """Lookup."""
      return name

    props = lookup.parameters.get("properties", {})
    assert "name" in props
    assert props["name"]["type"] == "string"

  def test_int_param_in_schema(self):
    @tool
    def compute(count: int) -> int:
      """Compute."""
      return count

    props = compute.parameters.get("properties", {})
    assert "count" in props
    assert props["count"]["type"] == "number"

  def test_bool_param_in_schema(self):
    @tool
    def toggle(enabled: bool) -> bool:
      """Toggle."""
      return enabled

    props = toggle.parameters.get("properties", {})
    assert "enabled" in props
    assert props["enabled"]["type"] == "boolean"

  def test_list_param_in_schema(self):
    @tool
    def batch(items: List[str]) -> str:
      """Batch process."""
      return str(items)

    props = batch.parameters.get("properties", {})
    assert "items" in props
    assert props["items"]["type"] == "array"
    assert props["items"]["items"]["type"] == "string"

  def test_multiple_params_all_present(self):
    @tool
    def multi(name: str, count: int, verbose: bool) -> str:
      """Multi-param tool."""
      return "ok"

    props = multi.parameters.get("properties", {})
    assert "name" in props
    assert "count" in props
    assert "verbose" in props

  def test_required_includes_non_default_params(self):
    @tool
    def req(required_arg: str) -> str:
      """Has a required arg."""
      return required_arg

    required = req.parameters.get("required", [])
    assert "required_arg" in required

  def test_optional_param_not_required(self):
    @tool
    def opt(name: str, nickname: Optional[str] = None) -> str:
      """Optional arg."""
      return name

    required = opt.parameters.get("required", [])
    assert "name" in required
    assert "nickname" not in required


@pytest.mark.unit
class TestToolDecoratorStrict:
  """@tool(strict=True) marks all fields as required."""

  def test_strict_required_params_are_required(self):
    @tool(strict=True)
    def strict_tool(name: str, count: int) -> str:
      """Strict tool."""
      return name

    required = strict_tool.parameters.get("required", [])
    # In strict mode, all non-default params should be required
    assert "name" in required
    assert "count" in required


@pytest.mark.unit
class TestToolDecoratorCacheResults:
  """@tool(cache_results=True) sets Function.cache_results."""

  def test_cache_results_true(self):
    @tool(cache_results=True)
    def cached(query: str) -> str:
      """Cached tool."""
      return query

    assert cached.cache_results is True

  def test_cache_results_default_false(self):
    @tool
    def uncached(query: str) -> str:
      """Uncached tool."""
      return query

    assert uncached.cache_results is False


@pytest.mark.unit
class TestToolDecoratorInvalidKwargs:
  """Invalid kwargs to @tool raise ValueError."""

  def test_invalid_kwarg_raises(self):
    with pytest.raises(ValueError, match="Invalid tool configuration arguments"):

      @tool(nonexistent_param=True)  # type: ignore[call-overload]
      def bad_tool() -> str:
        """Bad."""
        return "oops"

  def test_multiple_invalid_kwargs(self):
    with pytest.raises(ValueError, match="Invalid tool configuration arguments"):

      @tool(foo="bar", baz=123)  # type: ignore[call-overload]
      def another_bad() -> str:
        """Also bad."""
        return "oops"


@pytest.mark.unit
class TestToolDecoratorMutuallyExclusiveFlags:
  """Only one of requires_user_input, requires_confirmation, external_execution can be True."""

  def test_confirmation_and_user_input_raises(self):
    with pytest.raises(ValueError, match="Only one of"):

      @tool(requires_confirmation=True, requires_user_input=True)
      def conflicting() -> str:
        """Conflicting flags."""
        return "oops"

  def test_confirmation_and_external_execution_raises(self):
    with pytest.raises(ValueError, match="Only one of"):

      @tool(requires_confirmation=True, external_execution=True)
      def conflicting() -> str:
        """Conflicting flags."""
        return "oops"
