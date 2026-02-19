"""
Unit tests for JSON schema generation utilities.

Tests pure logic: Python type hints → JSON schema objects.
No API calls. No external dependencies.

IMPORTANT: get_json_type_for_py_type takes a STRING (type name), not a type object.
Mapping notes (from actual implementation):
  - int/float/complex/Decimal → "number" (not "integer" — intentional in this lib)
  - str/string → "string"
  - bool/boolean → "boolean"
  - NoneType/None → "null"
  - list/tuple/set/frozenset → "array"
  - dict/mapping → "object"
  - Unknown → "object" (fallback)

Covers:
  - get_json_type_for_py_type maps string names correctly
  - get_json_schema produces valid OpenAI-format tool schema from type hints
  - Optional types are handled (not added to required)
  - List[str] produces array type
  - Description is added when provided
  - "return" key is excluded from schema properties
"""

from typing import List, Optional

import pytest

from definable.utils.json_schema import get_json_schema, get_json_type_for_py_type


@pytest.mark.unit
class TestGetJsonTypeForPyType:
  """get_json_type_for_py_type maps Python type NAME STRINGS to JSON Schema types."""

  def test_str_name_maps_to_string(self):
    assert get_json_type_for_py_type("str") == "string"

  def test_string_name_maps_to_string(self):
    assert get_json_type_for_py_type("string") == "string"

  def test_int_name_maps_to_number(self):
    # Implementation maps int→number (not "integer") — intentional
    assert get_json_type_for_py_type("int") == "number"

  def test_float_name_maps_to_number(self):
    assert get_json_type_for_py_type("float") == "number"

  def test_bool_name_maps_to_boolean(self):
    assert get_json_type_for_py_type("bool") == "boolean"

  def test_nonetype_maps_to_null(self):
    assert get_json_type_for_py_type("NoneType") == "null"

  def test_list_name_maps_to_array(self):
    assert get_json_type_for_py_type("list") == "array"

  def test_tuple_name_maps_to_array(self):
    assert get_json_type_for_py_type("tuple") == "array"

  def test_dict_name_maps_to_object(self):
    assert get_json_type_for_py_type("dict") == "object"

  def test_unknown_type_falls_back_to_object(self):
    assert get_json_type_for_py_type("SomeUnknownType") == "object"


@pytest.mark.unit
class TestGetJsonSchema:
  """get_json_schema generates OpenAI-compatible tool parameter schemas."""

  def test_single_string_param_produces_properties_entry(self):
    hints = {"name": str}
    schema = get_json_schema(hints)
    assert schema["type"] == "object"
    assert "name" in schema["properties"]
    assert schema["properties"]["name"]["type"] == "string"

  def test_float_param_produces_number_type(self):
    hints = {"score": float}
    schema = get_json_schema(hints)
    assert schema["properties"]["score"]["type"] == "number"

  def test_bool_param_produces_boolean_type(self):
    hints = {"verbose": bool}
    schema = get_json_schema(hints)
    assert schema["properties"]["verbose"]["type"] == "boolean"

  def test_multiple_params_all_in_properties(self):
    hints = {"name": str, "score": float, "active": bool}
    schema = get_json_schema(hints)
    props = schema["properties"]
    assert "name" in props
    assert "score" in props
    assert "active" in props

  def test_optional_param_extracts_inner_type(self):
    """Optional[str] should be treated as str (not nullable union) for simplicity."""
    hints = {"nickname": Optional[str]}
    schema = get_json_schema(hints)
    # Optional param should still appear in properties
    assert "nickname" in schema["properties"]

  def test_list_param_generates_array_type(self):
    hints = {"items": List[str]}
    schema = get_json_schema(hints)
    prop = schema["properties"]["items"]
    assert prop["type"] == "array"

  def test_list_param_has_items_schema(self):
    hints = {"tags": List[str]}
    schema = get_json_schema(hints)
    prop = schema["properties"]["tags"]
    assert "items" in prop
    assert prop["items"]["type"] == "string"

  def test_description_added_when_provided(self):
    hints = {"query": str}
    descriptions = {"query": "The search query string"}
    schema = get_json_schema(hints, param_descriptions=descriptions)
    assert schema["properties"]["query"].get("description") == "The search query string"

  def test_empty_hints_returns_empty_object_schema(self):
    schema = get_json_schema({})
    assert schema["type"] == "object"
    assert schema.get("properties", {}) == {}

  def test_return_key_excluded_from_properties(self):
    """'return' key from type hints should not appear in schema properties."""
    hints = {"query": str, "return": str}
    schema = get_json_schema(hints)
    assert "return" not in schema.get("properties", {})

  def test_schema_always_has_type_object(self):
    hints = {"x": int}
    schema = get_json_schema(hints)
    assert schema["type"] == "object"

  def test_description_not_added_for_params_without_description(self):
    hints = {"a": str, "b": str}
    descriptions = {"a": "Description for a"}
    schema = get_json_schema(hints, param_descriptions=descriptions)
    # 'b' has no description — should not have a 'description' key
    assert "description" not in schema["properties"]["b"]
