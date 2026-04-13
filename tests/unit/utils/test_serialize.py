"""
Unit tests for JSON serialization utilities.

Tests the json_serializer default handler for json.dumps, covering
datetime/date/time objects, Enum values, and the string fallback.
No API calls. No external dependencies.

Covers:
  - datetime objects serialize to ISO format
  - date objects serialize to ISO format
  - time objects serialize to ISO format
  - Enum with string value returns the value
  - Enum with int value returns the value
  - Enum with non-serializable value returns the name
  - Unknown objects fall back to str()
  - json_serializer works as json.dumps default handler
"""

import json
from datetime import date, datetime, time
from enum import Enum

import pytest

from definable.utils.serialize import json_serializer


@pytest.mark.unit
class TestJsonSerializerDatetime:
  """json_serializer handles datetime, date, and time objects."""

  def test_datetime_to_isoformat(self):
    dt = datetime(2026, 2, 19, 14, 30, 0)
    result = json_serializer(dt)
    assert result == "2026-02-19T14:30:00"

  def test_datetime_with_microseconds(self):
    dt = datetime(2026, 1, 15, 8, 0, 0, 123456)
    result = json_serializer(dt)
    assert result == "2026-01-15T08:00:00.123456"

  def test_date_to_isoformat(self):
    d = date(2026, 2, 19)
    result = json_serializer(d)
    assert result == "2026-02-19"

  def test_time_to_isoformat(self):
    t = time(14, 30, 0)
    result = json_serializer(t)
    assert result == "14:30:00"

  def test_time_with_microseconds(self):
    t = time(9, 15, 30, 500000)
    result = json_serializer(t)
    assert result == "09:15:30.500000"


@pytest.mark.unit
class TestJsonSerializerEnum:
  """json_serializer handles Enum objects by returning their values."""

  def test_string_enum_returns_value(self):
    class Color(Enum):
      RED = "red"
      GREEN = "green"

    result = json_serializer(Color.RED)
    assert result == "red"

  def test_int_enum_returns_value(self):
    class Priority(Enum):
      LOW = 1
      HIGH = 3

    result = json_serializer(Priority.HIGH)
    assert result == 3

  def test_float_enum_returns_value(self):
    class Weight(Enum):
      LIGHT = 0.5
      HEAVY = 9.8

    result = json_serializer(Weight.LIGHT)
    assert result == 0.5

  def test_bool_enum_returns_value(self):
    class Toggle(Enum):
      ON = True
      OFF = False

    result = json_serializer(Toggle.ON)
    assert result is True

  def test_none_enum_returns_value(self):
    class NullableStatus(Enum):
      UNKNOWN = None
      ACTIVE = "active"

    result = json_serializer(NullableStatus.UNKNOWN)
    assert result is None

  def test_non_serializable_enum_returns_name(self):
    class Complex(Enum):
      ITEM = {"nested": [1, 2, 3]}

    result = json_serializer(Complex.ITEM)
    assert result == "ITEM"


@pytest.mark.unit
class TestJsonSerializerFallback:
  """json_serializer falls back to str() for unknown types."""

  def test_custom_class_returns_str(self):
    class MyObj:
      def __str__(self):
        return "my-custom-object"

    result = json_serializer(MyObj())
    assert result == "my-custom-object"

  def test_set_returns_str(self):
    result = json_serializer({1, 2, 3})
    assert isinstance(result, str)

  def test_bytes_returns_str(self):
    result = json_serializer(b"hello")
    assert result == "b'hello'"

  def test_tuple_returns_str(self):
    result = json_serializer((1, 2, 3))
    assert result == "(1, 2, 3)"


@pytest.mark.unit
class TestJsonSerializerWithJsonDumps:
  """json_serializer works as the default handler in json.dumps."""

  def test_datetime_in_json_dumps(self):
    data = {"created": datetime(2026, 2, 19, 12, 0, 0)}
    result = json.dumps(data, default=json_serializer)
    parsed = json.loads(result)
    assert parsed["created"] == "2026-02-19T12:00:00"

  def test_enum_in_json_dumps(self):
    class Status(Enum):
      ACTIVE = "active"
      INACTIVE = "inactive"

    data = {"status": Status.ACTIVE}
    result = json.dumps(data, default=json_serializer)
    parsed = json.loads(result)
    assert parsed["status"] == "active"

  def test_mixed_types_in_json_dumps(self):
    class Level(Enum):
      INFO = "info"

    data = {
      "timestamp": datetime(2026, 2, 19, 10, 0, 0),
      "level": Level.INFO,
      "date": date(2026, 2, 19),
    }
    result = json.dumps(data, default=json_serializer)
    parsed = json.loads(result)
    assert parsed["timestamp"] == "2026-02-19T10:00:00"
    assert parsed["level"] == "info"
    assert parsed["date"] == "2026-02-19"

  def test_nested_datetime_in_json_dumps(self):
    data = {"events": [{"at": datetime(2026, 1, 1, 0, 0, 0)}]}
    result = json.dumps(data, default=json_serializer)
    parsed = json.loads(result)
    assert parsed["events"][0]["at"] == "2026-01-01T00:00:00"
