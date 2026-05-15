from __future__ import annotations

import datetime as _dt

from definable.db.types import decode_value, encode_value


def test_encode_passthrough_primitives() -> None:
  assert encode_value(None) is None
  assert encode_value(42) == 42
  assert encode_value(3.14) == 3.14
  assert encode_value("hi") == "hi"


def test_encode_dict_and_list_as_json() -> None:
  enc = encode_value({"a": 1, "b": [1, 2]})
  assert isinstance(enc, str)
  assert '"a": 1' in enc
  assert encode_value([1, 2, 3]) == "[1, 2, 3]"


def test_encode_datetime_iso() -> None:
  d = _dt.datetime(2026, 5, 15, 12, 30, 0)
  assert encode_value(d) == "2026-05-15T12:30:00"


def test_decode_roundtrip_dict() -> None:
  enc = encode_value({"a": 1})
  assert decode_value(enc, dict) == {"a": 1}


def test_decode_roundtrip_list() -> None:
  enc = encode_value([1, 2, 3])
  assert decode_value(enc, list) == [1, 2, 3]


def test_decode_datetime() -> None:
  enc = encode_value(_dt.datetime(2026, 5, 15, 12, 30, 0))
  out = decode_value(enc, _dt.datetime)
  assert isinstance(out, _dt.datetime)
  assert out.year == 2026 and out.month == 5 and out.day == 15


def test_decode_target_none_returns_value() -> None:
  assert decode_value("plain", None) == "plain"


def test_decode_invalid_json_returns_raw() -> None:
  assert decode_value("not json {{", dict) == "not json {{"
