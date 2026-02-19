#!/usr/bin/env python3
"""Review test: @tool decorator + Agent tool execution.

Tests tool creation, schema generation, direct invocation,
and Agent composition with tools.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

passed, failed, skipped = 0, 0, 0


def check(name, condition, error=""):
  global passed, failed
  if condition:
    print(f"✅ PASS: {name}")
    passed += 1
  else:
    print(f"❌ FAIL: {name} — {error}")
    failed += 1


# ── Imports ─────────────────────────────────────────────────────
try:
  from definable.tool import tool, Function
  from definable.agent import Agent, MockModel

  check("Import tool + agent", True)
except Exception as e:
  check("Import tool + agent", False, str(e))
  print(f"\n{'=' * 60}\nRESULT: {passed} passed | {failed} failed | 0 skipped")
  sys.exit(1)


# ── @tool decorator variants ───────────────────────────────────
@tool
def add(a: int, b: int) -> int:
  """Add two numbers."""
  return a + b


check("@tool bare decorator", isinstance(add, Function), type(add).__name__)
check("@tool .name matches function", add.name == "add", f"got: {add.name}")
check("@tool .description from docstring", "Add" in (add.description or ""), f"got: {add.description}")


@tool(name="custom_name", description="Custom desc")
def my_func(x: str) -> str:
  """Original docstring."""
  return x.upper()


check("@tool(name=...) custom name", my_func.name == "custom_name", f"got: {my_func.name}")
check("@tool(description=...) custom desc", my_func.description == "Custom desc", f"got: {my_func.description}")


@tool
def no_docstring(x: int) -> int:
  return x * 2


check("@tool without docstring creates Function", isinstance(no_docstring, Function), type(no_docstring).__name__)


@tool
def no_args() -> str:
  """No arguments tool."""
  return "hello"


check("@tool zero-arg function", isinstance(no_args, Function), type(no_args).__name__)


@tool
async def async_tool(x: int) -> int:
  """Async tool."""
  return x + 1


check("@tool async function", isinstance(async_tool, Function), type(async_tool).__name__)


@tool
def returns_none() -> None:
  """Returns nothing."""
  pass


check("@tool returning None", isinstance(returns_none, Function), type(returns_none).__name__)


# ── Tool direct invocation ──────────────────────────────────────
try:
  result = add.entrypoint(3, 4)
  check("Tool direct call: add(3, 4) == 7", result == 7, f"got: {result}")
except Exception as e:
  check("Tool direct call", False, str(e))

try:
  result = my_func.entrypoint("hello")
  check("Tool direct call: custom_name('hello') == 'HELLO'", result == "HELLO", f"got: {result}")
except Exception as e:
  check("Tool direct custom", False, str(e))


# ── Tool schema ─────────────────────────────────────────────────
try:
  params = add.parameters
  check("Tool .parameters exists", params is not None, "None")
  check("Tool .parameters has 'a' param", "a" in str(params), f"params: {params}")
except Exception as e:
  check("Tool schema", False, str(e))


# ── Agent + Tools composition ───────────────────────────────────
try:
  agent = Agent(model=MockModel(), tools=[add, my_func, no_args])
  check("Agent constructs with 3 tools", len(agent.tools) == 3, f"got: {len(agent.tools)}")
except Exception as e:
  check("Agent + tools construction", False, str(e))

try:
  agent = Agent(model=MockModel(), tools=[add])
  output = agent.run("test")
  check("Agent + tool runs without error", output.content is not None, "no content")
except Exception as e:
  check("Agent + tool run", False, str(e))


# ── Tool that raises ───────────────────────────────────────────
@tool
def exploding() -> str:
  """This always fails."""
  raise ValueError("Boom!")


try:
  agent = Agent(model=MockModel(), tools=[exploding])
  check("Agent accepts tool that can raise", True)
except Exception as e:
  check("Agent with raising tool", False, str(e))


# ── @tool with stop_after_tool_call ────────────────────────────
@tool(stop_after_tool_call=True)
def stop_tool(x: str) -> str:
  """Tool that stops."""
  return x


check("@tool(stop_after_tool_call=True)", isinstance(stop_tool, Function), type(stop_tool).__name__)
check(
  "stop_after_tool_call sets show_result", getattr(stop_tool, "show_result", None) is True, f"show_result={getattr(stop_tool, 'show_result', 'N/A')}"
)


# ── @tool with cache ───────────────────────────────────────────
@tool(cache_results=True, cache_ttl=60)
def cached_tool(q: str) -> str:
  """Cached tool."""
  return f"result for {q}"


check("@tool(cache_results=True) constructs", isinstance(cached_tool, Function), type(cached_tool).__name__)


# ── Result ──────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"RESULT: {passed} passed | {failed} failed | 0 skipped")
sys.exit(1 if failed else 0)
