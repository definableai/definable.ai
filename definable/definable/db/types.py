"""Column value adapters used by :class:`definable.db.repo.Repo`.

Keeps Repo small: when a dataclass field is a ``dict`` or ``list``, the
value is JSON-encoded on insert and decoded on fetch. Datetimes are
ISO strings. Everything else is passed through to the driver.
"""

from __future__ import annotations

import datetime as _dt
import json
from typing import Any


def encode_value(v: Any) -> Any:
  """Convert a Python value into something the sqlite driver accepts."""
  if v is None:
    return None
  if isinstance(v, (dict, list)):
    return json.dumps(v, default=str)
  if isinstance(v, _dt.datetime):
    return v.isoformat()
  return v


def decode_value(v: Any, target: type[Any] | None) -> Any:
  """Reverse :func:`encode_value` when the dataclass field type is known.

  ``target`` may be a generic alias like ``dict[str, int]`` — we only need
  the origin to decide whether to JSON-parse.
  """
  if v is None or target is None:
    return v
  origin = getattr(target, "__origin__", target)
  if origin in (dict, list) and isinstance(v, str):
    try:
      return json.loads(v)
    except json.JSONDecodeError:
      return v
  if target is _dt.datetime and isinstance(v, str):
    try:
      return _dt.datetime.fromisoformat(v)
    except ValueError:
      return v
  return v
