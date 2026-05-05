from definable.tool import tool
from definable.tool.function import FunctionCall


events: list[tuple[str, object]] = []


def before(fc) -> None:
  events.append(("before", dict(fc.arguments or {})))


def after(fc) -> None:
  events.append(("after", fc.result))


@tool(pre_hook=before, post_hook=after)
def greet(name: str) -> str:
  """Return a greeting."""
  return f"Hello {name}"


result = FunctionCall(function=greet, arguments={"name": "Ada"}).execute()

assert result.result == "Hello Ada"
assert events == [
  ("before", {"name": "Ada"}),
  ("after", "Hello Ada"),
]
