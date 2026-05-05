from definable.tool import Function, tool


@tool
def add(a: int, b: int) -> int:
  """Add two integers."""
  return a + b


assert isinstance(add, Function)
assert add.parameters["properties"]["a"]["type"] == "number"
assert add.entrypoint(a=3, b=4) == 7
