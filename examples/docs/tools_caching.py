from pathlib import Path
from tempfile import TemporaryDirectory

from definable.tool import tool
from definable.tool.function import FunctionCall


calls = {"count": 0}

with TemporaryDirectory() as tmpdir:

  @tool(cache_results=True, cache_dir=tmpdir, cache_ttl=60)
  def expensive_lookup(query: str) -> str:
    """Return a cached result for repeated queries."""
    calls["count"] += 1
    return f"{query}:{calls['count']}"

  first = FunctionCall(function=expensive_lookup, arguments={"query": "agents"}).execute()
  second = FunctionCall(function=expensive_lookup, arguments={"query": "agents"}).execute()

  assert first.result == "agents:1"
  assert second.result == "agents:1"
  assert calls["count"] == 1
  assert list(Path(tmpdir).rglob("*.json"))
