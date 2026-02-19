"""
Unit tests for the Function and FunctionCall classes.

Tests the Function Pydantic model (construction, to_dict, from_callable,
process_entrypoint, model_copy) and FunctionCall (get_call_str, execute).
No API calls. No external dependencies.

Covers:
  - Function with name and description
  - Function.to_dict() includes name, description, parameters
  - Function.from_callable() creates Function from plain function
  - Function.from_callable() with type hints produces correct parameters schema
  - Function.parameters default is empty object schema
  - Function model_copy works for deep and shallow
  - FunctionCall.get_call_str() formats correctly
  - FunctionCall.execute() runs sync entrypoint
  - Function.process_entrypoint() updates parameters from type hints
  - Function.sequential defaults to False
  - Function.cache_results defaults to False
"""

from typing import List, Optional

import pytest

from definable.tool.function import Function, FunctionCall


@pytest.mark.unit
class TestFunctionConstruction:
  """Function model can be created with name and description."""

  def test_create_with_name_and_description(self):
    fn = Function(name="my_func", description="Does something useful")
    assert fn.name == "my_func"
    assert fn.description == "Does something useful"

  def test_create_with_name_only(self):
    fn = Function(name="bare_func")
    assert fn.name == "bare_func"
    assert fn.description is None

  def test_entrypoint_defaults_to_none(self):
    fn = Function(name="no_entry")
    assert fn.entrypoint is None


@pytest.mark.unit
class TestFunctionDefaultParameters:
  """Function.parameters defaults to an empty object schema."""

  def test_default_parameters_schema(self):
    fn = Function(name="empty")
    assert fn.parameters == {"type": "object", "properties": {}, "required": []}

  def test_default_parameters_is_object_type(self):
    fn = Function(name="empty")
    assert fn.parameters["type"] == "object"

  def test_default_parameters_has_empty_properties(self):
    fn = Function(name="empty")
    assert fn.parameters["properties"] == {}


@pytest.mark.unit
class TestFunctionToDict:
  """Function.to_dict() returns a dictionary with key fields."""

  def test_includes_name(self):
    fn = Function(name="test_fn", description="Test")
    d = fn.to_dict()
    assert d["name"] == "test_fn"

  def test_includes_description(self):
    fn = Function(name="test_fn", description="A description")
    d = fn.to_dict()
    assert d["description"] == "A description"

  def test_includes_parameters(self):
    fn = Function(name="test_fn", description="Test", parameters={"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]})
    d = fn.to_dict()
    assert "parameters" in d
    assert d["parameters"]["properties"]["x"]["type"] == "number"

  def test_excludes_none_fields(self):
    fn = Function(name="test_fn")
    d = fn.to_dict()
    assert "description" not in d

  def test_excludes_entrypoint(self):
    fn = Function(name="test_fn", entrypoint=lambda: None)
    d = fn.to_dict()
    assert "entrypoint" not in d

  def test_excludes_internal_fields(self):
    fn = Function(name="test_fn", description="Test")
    d = fn.to_dict()
    assert "show_result" not in d
    assert "stop_after_tool_call" not in d
    assert "cache_results" not in d


@pytest.mark.unit
class TestFunctionFromCallable:
  """Function.from_callable() creates a Function from a plain function."""

  def test_creates_function_from_plain_function(self):
    def greet(name: str) -> str:
      """Say hello."""
      return f"Hello, {name}"

    fn = Function.from_callable(greet)
    assert isinstance(fn, Function)
    assert fn.name == "greet"

  def test_description_from_docstring(self):
    def documented() -> str:
      """This is a documented function."""
      return "done"

    fn = Function.from_callable(documented)
    assert fn.description == "This is a documented function."

  def test_custom_name_override(self):
    def original() -> str:
      """Original."""
      return "ok"

    fn = Function.from_callable(original, name="custom_name")
    assert fn.name == "custom_name"

  def test_entrypoint_is_set(self):
    def simple() -> str:
      """Simple."""
      return "ok"

    fn = Function.from_callable(simple)
    assert fn.entrypoint is not None
    assert callable(fn.entrypoint)


@pytest.mark.unit
class TestFunctionFromCallableTypeHints:
  """Function.from_callable() with type hints produces correct parameters schema."""

  def test_string_param_schema(self):
    def search(query: str) -> str:
      """Search."""
      return query

    fn = Function.from_callable(search)
    props = fn.parameters.get("properties", {})
    assert "query" in props
    assert props["query"]["type"] == "string"

  def test_int_param_schema(self):
    def count(n: int) -> int:
      """Count."""
      return n

    fn = Function.from_callable(count)
    props = fn.parameters.get("properties", {})
    assert "n" in props
    assert props["n"]["type"] == "number"

  def test_list_param_schema(self):
    def batch(items: List[str]) -> str:
      """Batch."""
      return str(items)

    fn = Function.from_callable(batch)
    props = fn.parameters.get("properties", {})
    assert "items" in props
    assert props["items"]["type"] == "array"

  def test_required_params_detected(self):
    def require(a: str, b: int) -> str:
      """Require both."""
      return a

    fn = Function.from_callable(require)
    required = fn.parameters.get("required", [])
    assert "a" in required
    assert "b" in required

  def test_optional_params_not_required(self):
    def opt(a: str, b: Optional[str] = None) -> str:
      """One required, one optional."""
      return a

    fn = Function.from_callable(opt)
    required = fn.parameters.get("required", [])
    assert "a" in required
    assert "b" not in required

  def test_agent_param_excluded_from_schema(self):
    def with_agent(query: str, agent: object = None) -> str:
      """Has agent param."""
      return query

    fn = Function.from_callable(with_agent)
    props = fn.parameters.get("properties", {})
    assert "agent" not in props
    assert "query" in props

  def test_strict_makes_all_required(self):
    def relaxed(a: str, b: Optional[str] = None) -> str:
      """Relaxed function."""
      return a

    fn = Function.from_callable(relaxed, strict=True)
    required = fn.parameters.get("required", [])
    props = fn.parameters.get("properties", {})
    for prop_name in props:
      assert prop_name in required


@pytest.mark.unit
class TestFunctionModelCopy:
  """Function.model_copy() supports both shallow and deep copy."""

  def test_shallow_copy_produces_new_instance(self):
    fn = Function(name="original", description="Original description")
    copied = fn.model_copy()
    assert copied is not fn
    assert copied.name == "original"
    assert copied.description == "Original description"

  def test_shallow_copy_with_update(self):
    fn = Function(name="original", description="Original")
    copied = fn.model_copy(update={"name": "updated"})
    assert copied.name == "updated"
    assert fn.name == "original"

  def test_deep_copy_produces_new_instance(self):
    fn = Function(
      name="original",
      description="Original",
      parameters={"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]},
    )
    copied = fn.model_copy(deep=True)
    assert copied is not fn
    assert copied.name == "original"

  def test_deep_copy_parameters_are_independent(self):
    fn = Function(
      name="original",
      parameters={"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]},
    )
    copied = fn.model_copy(deep=True)
    copied.parameters["properties"]["y"] = {"type": "string"}
    assert "y" not in fn.parameters["properties"]

  def test_deep_copy_preserves_entrypoint_reference(self):
    def my_func():
      return "hello"

    fn = Function(name="original", entrypoint=my_func)
    copied = fn.model_copy(deep=True)
    # Entrypoint should be the same reference (shallow copied)
    assert copied.entrypoint is fn.entrypoint


@pytest.mark.unit
class TestFunctionCallGetCallStr:
  """FunctionCall.get_call_str() formats the call as a readable string."""

  def test_no_arguments(self):
    fn = Function(name="ping")
    fc = FunctionCall(function=fn, arguments=None)
    assert fc.get_call_str() == "ping()"

  def test_with_arguments(self):
    fn = Function(name="greet")
    fc = FunctionCall(function=fn, arguments={"name": "Alice"})
    call_str = fc.get_call_str()
    assert "greet(" in call_str
    assert "name=Alice" in call_str

  def test_with_multiple_arguments(self):
    fn = Function(name="compute")
    fc = FunctionCall(function=fn, arguments={"a": 1, "b": 2})
    call_str = fc.get_call_str()
    assert "compute(" in call_str
    assert "a=1" in call_str
    assert "b=2" in call_str


@pytest.mark.unit
class TestFunctionCallExecuteSync:
  """FunctionCall.execute() runs a sync entrypoint and returns result."""

  def test_execute_returns_success(self):
    def add(a: int, b: int) -> int:
      return a + b

    fn = Function(name="add", entrypoint=add)
    fc = FunctionCall(function=fn, arguments={"a": 3, "b": 4})
    result = fc.execute()
    assert result.status == "success"
    assert result.result == 7

  def test_execute_no_entrypoint_returns_failure(self):
    fn = Function(name="empty")
    fc = FunctionCall(function=fn, arguments={})
    result = fc.execute()
    assert result.status == "failure"
    assert "Entrypoint is not set" in result.error

  def test_execute_stores_result_on_function_call(self):
    def echo(msg: str) -> str:
      return msg

    fn = Function(name="echo", entrypoint=echo)
    fc = FunctionCall(function=fn, arguments={"msg": "hello"})
    fc.execute()
    assert fc.result == "hello"


@pytest.mark.unit
class TestFunctionProcessEntrypoint:
  """Function.process_entrypoint() updates parameters from type hints."""

  def test_updates_parameters_from_entrypoint(self):
    def my_func(query: str, limit: int) -> str:
      """Search with limit."""
      return query

    fn = Function(name="my_func", entrypoint=my_func)
    fn.process_entrypoint()

    props = fn.parameters.get("properties", {})
    assert "query" in props
    assert "limit" in props

  def test_sets_description_from_docstring(self):
    def documented(x: int) -> int:
      """A well-documented function."""
      return x

    fn = Function(name="documented", entrypoint=documented)
    fn.process_entrypoint()
    assert fn.description == "A well-documented function."

  def test_no_op_when_no_entrypoint(self):
    fn = Function(name="empty")
    fn.process_entrypoint()
    assert fn.parameters == {"type": "object", "properties": {}, "required": []}

  def test_skip_entrypoint_processing_flag(self):
    def my_func(x: int) -> int:
      """Skipped."""
      return x

    fn = Function(name="skipped", entrypoint=my_func, skip_entrypoint_processing=True)
    fn.process_entrypoint()
    # Parameters should remain as default since processing was skipped
    assert fn.parameters == {"type": "object", "properties": {}, "required": []}


@pytest.mark.unit
class TestFunctionDefaultFlags:
  """Function default field values are correct."""

  def test_sequential_defaults_to_false(self):
    fn = Function(name="test")
    assert fn.sequential is False

  def test_cache_results_defaults_to_false(self):
    fn = Function(name="test")
    assert fn.cache_results is False

  def test_show_result_defaults_to_false(self):
    fn = Function(name="test")
    assert fn.show_result is False

  def test_stop_after_tool_call_defaults_to_false(self):
    fn = Function(name="test")
    assert fn.stop_after_tool_call is False

  def test_add_instructions_defaults_to_true(self):
    fn = Function(name="test")
    assert fn.add_instructions is True

  def test_cache_ttl_defaults_to_3600(self):
    fn = Function(name="test")
    assert fn.cache_ttl == 3600
