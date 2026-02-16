"""JSON operations skill — parse, query, transform, and format JSON data.

Gives the agent the ability to work with JSON data safely, including
parsing strings, querying with JSONPath-like expressions, formatting,
and structural transformations.

Example:
    from definable.skills.builtin import JSONOperations

    agent = Agent(
        model=model,
        skills=[JSONOperations()],
    )
    output = agent.run("Parse this JSON and extract all email addresses: ...")
"""

import json
from typing import Any, Dict, List

from definable.skills.base import Skill
from definable.tools.decorator import tool


def _query_path(data: Any, path: str) -> Any:
  """Simple dot-notation path query (e.g. 'users.0.name')."""
  parts = path.split(".")
  current = data
  for part in parts:
    if isinstance(current, dict):
      if part not in current:
        raise KeyError(f"Key '{part}' not found. Available keys: {list(current.keys())}")
      current = current[part]
    elif isinstance(current, list):
      try:
        idx = int(part)
        current = current[idx]
      except (ValueError, IndexError):
        if part == "*":
          # Wildcard: return list of all items
          return current
        raise KeyError(f"Invalid list index '{part}' for list of length {len(current)}")
    else:
      raise TypeError(f"Cannot traverse into {type(current).__name__} with key '{part}'")
  return current


@tool
def parse_json(text: str) -> str:
  """Parse a JSON string and return it formatted with indentation.

  Use this to validate JSON syntax and pretty-print it for readability.

  Args:
    text: A JSON string to parse and format.

  Returns:
    Pretty-printed JSON, or an error message with the parse error location.
  """
  try:
    data = json.loads(text)
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)
  except json.JSONDecodeError as e:
    return f"JSON parse error at line {e.lineno}, column {e.colno}: {e.msg}"


@tool
def query_json(json_text: str, path: str) -> str:
  """Extract a value from JSON data using a dot-notation path.

  Supports nested objects, arrays (by index), and wildcards (*).

  Args:
    json_text: The JSON string to query.
    path: Dot-notation path like "users.0.name", "config.database.host",
        or "items.*.id" for all items in an array.

  Returns:
    The extracted value as a formatted string, or an error message.
  """
  try:
    data = json.loads(json_text)
  except json.JSONDecodeError as e:
    return f"JSON parse error: {e.msg}"

  try:
    result = _query_path(data, path)
    if isinstance(result, (dict, list)):
      return json.dumps(result, indent=2, ensure_ascii=False, default=str)
    return str(result)
  except (KeyError, TypeError, IndexError) as e:
    return f"Query error: {e}"


@tool
def transform_json(json_text: str, operation: str) -> str:
  """Transform JSON data with common operations.

  Args:
    json_text: The JSON string to transform.
    operation: The transformation to apply. Options:
        - "keys" — list all top-level keys
        - "values" — list all top-level values
        - "flatten" — flatten nested objects with dot-notation keys
        - "length" — count items (array length or object key count)
        - "types" — show the type of each top-level field
        - "compact" — minify the JSON (remove whitespace)
        - "sort_keys" — sort object keys alphabetically

  Returns:
    The transformed result as a string.
  """
  try:
    data = json.loads(json_text)
  except json.JSONDecodeError as e:
    return f"JSON parse error: {e.msg}"

  op = operation.strip().lower()

  if op == "keys":
    if isinstance(data, dict):
      return json.dumps(list(data.keys()), indent=2)
    return f"Error: 'keys' requires a JSON object, got {type(data).__name__}"

  elif op == "values":
    if isinstance(data, dict):
      return json.dumps(list(data.values()), indent=2, default=str)
    return f"Error: 'values' requires a JSON object, got {type(data).__name__}"

  elif op == "flatten":
    if not isinstance(data, dict):
      return f"Error: 'flatten' requires a JSON object, got {type(data).__name__}"

    flat: Dict[str, Any] = {}

    def _flatten(obj: Any, prefix: str = "") -> None:
      if isinstance(obj, dict):
        for k, v in obj.items():
          new_key = f"{prefix}.{k}" if prefix else k
          _flatten(v, new_key)
      elif isinstance(obj, list):
        for i, v in enumerate(obj):
          _flatten(v, f"{prefix}.{i}")
      else:
        flat[prefix] = obj

    _flatten(data)
    return json.dumps(flat, indent=2, default=str)

  elif op == "length":
    if isinstance(data, (list, dict)):
      return str(len(data))
    return f"Error: 'length' requires an array or object, got {type(data).__name__}"

  elif op == "types":
    if isinstance(data, dict):
      types = {k: type(v).__name__ for k, v in data.items()}
      return json.dumps(types, indent=2)
    elif isinstance(data, list):
      types_list = [type(v).__name__ for v in data]
      return json.dumps(types_list, indent=2)
    return type(data).__name__

  elif op == "compact":
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False, default=str)

  elif op == "sort_keys":
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, default=str)

  else:
    return f"Unknown operation: '{operation}'. Options: keys, values, flatten, length, types, compact, sort_keys"


@tool
def compare_json(json_a: str, json_b: str) -> str:
  """Compare two JSON values and report differences.

  Args:
    json_a: First JSON string.
    json_b: Second JSON string.

  Returns:
    A summary of differences between the two JSON values.
  """
  try:
    a = json.loads(json_a)
  except json.JSONDecodeError as e:
    return f"Error parsing first JSON: {e.msg}"
  try:
    b = json.loads(json_b)
  except json.JSONDecodeError as e:
    return f"Error parsing second JSON: {e.msg}"

  if a == b:
    return "The two JSON values are identical."

  diffs: List[str] = []

  if type(a) is not type(b):
    diffs.append(f"Type mismatch: {type(a).__name__} vs {type(b).__name__}")
  elif isinstance(a, dict) and isinstance(b, dict):
    keys_a = set(a.keys())
    keys_b = set(b.keys())
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    common = keys_a & keys_b

    if only_a:
      diffs.append(f"Keys only in first: {sorted(only_a)}")
    if only_b:
      diffs.append(f"Keys only in second: {sorted(only_b)}")
    for key in sorted(common):
      if a[key] != b[key]:
        diffs.append(f"  '{key}': {json.dumps(a[key], default=str)} → {json.dumps(b[key], default=str)}")
  elif isinstance(a, list) and isinstance(b, list):
    diffs.append(f"Array lengths: {len(a)} vs {len(b)}")
    for i in range(min(len(a), len(b))):
      if a[i] != b[i]:
        diffs.append(f"  [{i}]: {json.dumps(a[i], default=str)} → {json.dumps(b[i], default=str)}")
  else:
    diffs.append(f"Values differ: {json.dumps(a, default=str)} vs {json.dumps(b, default=str)}")

  return "\n".join(diffs)


class JSONOperations(Skill):
  """Skill for parsing, querying, and transforming JSON data.

  Adds tools for safely working with JSON: parsing, path queries,
  transformations (flatten, sort, filter), and comparison.

  Example:
    agent = Agent(model=model, skills=[JSONOperations()])
    output = agent.run("Parse this JSON and extract the user emails: {...}")
  """

  name = "json_operations"
  instructions = (
    "You have access to JSON tools for working with structured data. "
    "Use parse_json to validate and pretty-print JSON strings. "
    "Use query_json with dot-notation paths (e.g. 'users.0.email') to extract values. "
    "Use transform_json for operations like flattening, sorting keys, or getting structure info. "
    "Use compare_json to find differences between two JSON documents."
  )

  def __init__(self):
    super().__init__()

  @property
  def tools(self) -> list:
    return [parse_json, query_json, transform_json, compare_json]
