from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from definable.tool import tool


class SearchFilter(BaseModel):
  category: Literal["docs", "blog"]
  limit: int = Field(default=5, ge=1, le=20)


class Priority(str, Enum):
  low = "low"
  medium = "medium"
  high = "high"


@tool
def search(query: str, filters: SearchFilter, priority: Priority = Priority.medium) -> str:
  """Search indexed content.

  Args:
    query: Search text.
    filters: Search filter configuration.
    priority: Execution priority.
  """
  return f"{query}:{filters.category}:{priority.value}:{filters.limit}"


assert search.parameters["properties"]["query"]["type"] == "string"
assert search.parameters["properties"]["filters"]["type"] == "object"
assert search.parameters["properties"]["priority"]["enum"] == ["low", "medium", "high"]
assert search.entrypoint(query="agents", filters=SearchFilter(category="docs"), priority=Priority.high) == "agents:docs:high:5"
