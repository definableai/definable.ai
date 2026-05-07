"""ToolRegistry unit tests — flatten, dispatch, error capture."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pytest

from definable.agent.core.events import (
  Event,
  EventBus,
  ToolCall,
  ToolCallCompleted,
  ToolCallFailed,
  ToolCallStarted,
)
from definable.agent.core.tools import ToolRegistry
from definable.tool.decorator import tool


@tool
def add(x: int, y: int) -> int:
  """Add two numbers."""
  return x + y


@tool
def boom() -> str:
  """Always fails."""
  raise RuntimeError("kaboom")


@tool
async def slow_double(n: int) -> int:
  """Double n after a tiny async sleep — used to verify parallel dispatch."""
  import asyncio as _a

  await _a.sleep(0.05)
  return n * 2


@dataclass
class _FakeToolkit:
  """Minimal duck-typed toolkit for registry tests."""

  tools: list[Any]


@pytest.mark.unit
def test_registry_flattens_tools_and_toolkits() -> None:
  reg = ToolRegistry(tools=[add], toolkits=[_FakeToolkit(tools=[boom])])

  names = reg.names()
  assert "add" in names
  assert "boom" in names
  assert len(reg.all()) == 2


@pytest.mark.unit
def test_registry_rejects_duplicate_names() -> None:
  with pytest.raises(ValueError, match="Duplicate tool name"):
    ToolRegistry(tools=[add], toolkits=[_FakeToolkit(tools=[add])])


@pytest.mark.unit
async def test_dispatch_parallel_happy_path() -> None:
  reg = ToolRegistry(tools=[add])
  bus = EventBus()
  fired: list[Event] = []
  bus.subscribe(fired.append)

  calls = [ToolCall(id="c1", name="add", args={"x": 2, "y": 3})]
  results = await reg.dispatch_parallel(calls, events=bus, run_id="r1")

  assert len(results) == 1
  assert results[0].success is True
  assert results[0].output == 5

  starts = [e for e in fired if isinstance(e, ToolCallStarted)]
  completes = [e for e in fired if isinstance(e, ToolCallCompleted)]
  fails = [e for e in fired if isinstance(e, ToolCallFailed)]
  assert len(starts) == 1
  assert len(completes) == 1
  assert len(fails) == 0


@pytest.mark.unit
async def test_dispatch_captures_errors_per_call() -> None:
  reg = ToolRegistry(tools=[add, boom])
  bus = EventBus()
  fired: list[Event] = []
  bus.subscribe(fired.append)

  calls = [
    ToolCall(id="c1", name="add", args={"x": 1, "y": 1}),
    ToolCall(id="c2", name="boom", args={}),
  ]
  results = await reg.dispatch_parallel(calls, events=bus, run_id="r1")

  assert results[0].success is True
  assert results[0].output == 2
  assert results[1].success is False
  assert results[1].error and "kaboom" in results[1].error

  fails = [e for e in fired if isinstance(e, ToolCallFailed)]
  assert len(fails) == 1
  assert fails[0].call.name == "boom"


@pytest.mark.unit
async def test_dispatch_unknown_tool_returns_failed_result() -> None:
  reg = ToolRegistry(tools=[add])
  bus = EventBus()
  fired: list[Event] = []
  bus.subscribe(fired.append)

  calls = [ToolCall(id="c1", name="ghost", args={})]
  results = await reg.dispatch_parallel(calls, events=bus, run_id="r1")

  assert len(results) == 1
  assert results[0].success is False
  assert results[0].error and "Unknown tool" in results[0].error

  fails = [e for e in fired if isinstance(e, ToolCallFailed)]
  assert len(fails) == 1


@pytest.mark.unit
async def test_dispatch_runs_calls_concurrently() -> None:
  reg = ToolRegistry(tools=[slow_double])
  bus = EventBus()

  calls = [ToolCall(id=f"c{i}", name="slow_double", args={"n": i}) for i in range(5)]

  start = time.perf_counter()
  results = await reg.dispatch_parallel(calls, events=bus, run_id="r1")
  elapsed = time.perf_counter() - start

  assert len(results) == 5
  assert [r.output for r in results] == [0, 2, 4, 6, 8]
  # Sequential would be ~250ms; parallel finishes in ~50ms (one sleep duration).
  assert elapsed < 0.15, f"dispatch was not parallel: took {elapsed:.3f}s"
