"""Smoke tests for the minimal harness-v2 Agent class."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from definable.agent.agent import Agent
from definable.agent.core.events import (
  Event,
  RunCompleted,
  StreamChunkEvent,
  TurnStarted,
)
from definable.model.base import Model
from definable.model.response import ModelResponse, ToolExecution
from definable.agent.toolkit.decorator import tool


@dataclass
class MockModel(Model):
  """Minimal Model mirroring the test_loop.py shape."""

  id: str = "mock"
  name: str = "Mock"  # type: ignore[assignment]
  provider: str = "mock"  # type: ignore[assignment]
  api_key: str = ""
  client: Any = None
  async_client: Any = None

  scripted: list[ModelResponse] = field(default_factory=list)
  stream_scripts: list[list[ModelResponse]] = field(default_factory=list)

  _ainvoke_count: int = 0
  _ainvoke_stream_count: int = 0

  def invoke(self, *a: Any, **kw: Any) -> ModelResponse:
    raise NotImplementedError

  async def ainvoke(self, *a: Any, assistant_message: Any = None, **kw: Any) -> ModelResponse:
    idx = self._ainvoke_count
    self._ainvoke_count += 1
    if idx >= len(self.scripted):
      raise IndexError(f"out of scripted at {idx}")
    resp = self.scripted[idx]
    if assistant_message is not None:
      assistant_message.content = resp.content
    return resp

  def invoke_stream(self, *a: Any, **kw: Any) -> Iterator[ModelResponse]:
    raise NotImplementedError

  async def ainvoke_stream(self, *a: Any, assistant_message: Any = None, **kw: Any) -> AsyncIterator[ModelResponse]:
    idx = self._ainvoke_stream_count
    self._ainvoke_stream_count += 1
    chunks = self.stream_scripts[idx] if idx < len(self.stream_scripts) else []
    full = "".join(c.content or "" for c in chunks if isinstance(c.content, str))
    if assistant_message is not None:
      assistant_message.content = full
    for c in chunks:
      yield c

  def _parse_provider_response(self, response: Any, **kwargs: Any) -> ModelResponse:
    return ModelResponse()

  def _parse_provider_response_delta(self, response: Any, **kwargs: Any) -> ModelResponse:
    return ModelResponse()


@tool
def echo(s: str) -> str:
  """Return its argument."""
  return s


@pytest.mark.unit
async def test_agent_minimal_basic_arun() -> None:
  llm = MockModel(scripted=[ModelResponse(content="hi back")])
  agent = Agent(name="t1", model=llm, instructions="be brief")
  result = await agent.arun("hello")
  assert result.content == "hi back"
  assert result.exit_reason == "natural"


@pytest.mark.unit
async def test_agent_minimal_streaming() -> None:
  llm = MockModel(stream_scripts=[[ModelResponse(content="he"), ModelResponse(content="llo")]])
  agent = Agent(name="t2", model=llm)

  fired: list[Event] = []
  agent.events.subscribe(fired.append)

  events = []
  async for event in await agent.arun("hi", stream=True):
    events.append(event)

  contents = [e.data for e in events if isinstance(e, StreamChunkEvent)]
  assert contents == ["he", "llo"]
  assert any(isinstance(e, TurnStarted) for e in events)
  assert any(isinstance(e, RunCompleted) for e in events)


@pytest.mark.unit
async def test_agent_minimal_memory_auto_injects_tools(tmp_path: Path) -> None:
  from definable.agent.memory import FileMemory

  mem = FileMemory(tmp_path)
  llm = MockModel(scripted=[ModelResponse(content="ok")])
  agent = Agent(name="t3", model=llm, memory=mem)

  names = agent.tools.names()
  assert "read_memory" in names
  assert "write_memory" in names
  assert "list_memories" in names
  assert "search_memory" in names


@pytest.mark.unit
async def test_agent_minimal_with_tools_in_loop() -> None:
  llm = MockModel(
    scripted=[
      ModelResponse(
        content=None,
        tool_executions=[ToolExecution(tool_call_id="c1", tool_name="echo", tool_args={"s": "hello"})],
      ),
      ModelResponse(content="echo'd: hello"),
    ]
  )
  agent = Agent(name="t4", model=llm, tools=[echo])
  result = await agent.arun("say hi")
  assert result.content == "echo'd: hello"
  assert result.turns == 2
