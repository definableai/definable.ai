"""Loop unit tests with a programmable MockModel."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from definable.agent.core.events import (
  Event,
  EventBus,
  ModelResponded,
  RunCompleted,
  RunErrored,
  StreamChunkEvent,
  ToolCallStarted,
  TurnStarted,
)
from definable.agent.core.loop import run
from definable.agent.core.tools import ToolRegistry
from definable.model.base import Model
from definable.model.response import ModelResponse, ToolExecution
from definable.tool.decorator import tool


# ---- test tools ----------------------------------------------------------


@tool
def add(x: int, y: int) -> int:
  """Add two numbers."""
  return x + y


@tool
def boom() -> str:
  """Always fails."""
  raise RuntimeError("kaboom")


# ---- programmable MockModel ----------------------------------------------


@dataclass
class MockModel(Model):
  """Minimal Model that returns scripted responses on each ainvoke call."""

  id: str = "mock"
  name: str = "Mock"  # type: ignore[assignment]
  provider: str = "mock"  # type: ignore[assignment]
  api_key: str = ""
  client: Any = None
  async_client: Any = None

  scripted: list[ModelResponse] = field(default_factory=list)
  stream_scripts: list[list[ModelResponse]] = field(default_factory=list)
  raise_on_call: Exception | None = None

  _ainvoke_count: int = 0
  _ainvoke_stream_count: int = 0

  def invoke(self, *a: Any, **kw: Any) -> ModelResponse:
    raise NotImplementedError

  async def ainvoke(self, *a: Any, assistant_message: Any = None, **kw: Any) -> ModelResponse:
    if self.raise_on_call is not None:
      raise self.raise_on_call
    idx = self._ainvoke_count
    self._ainvoke_count += 1
    if idx >= len(self.scripted):
      raise IndexError(f"MockModel out of scripted responses (asked for {idx})")
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
    full_content = "".join(c.content or "" for c in chunks if isinstance(c.content, str))
    if assistant_message is not None:
      assistant_message.content = full_content
    for c in chunks:
      yield c

  def _parse_provider_response(self, response: Any, **kwargs: Any) -> ModelResponse:
    return ModelResponse()

  def _parse_provider_response_delta(self, response: Any, **kwargs: Any) -> ModelResponse:
    return ModelResponse()


# ---- helpers -------------------------------------------------------------


def _make_messages(user: str = "hi") -> list[Any]:
  from definable.model.message import Message

  return [Message(role="user", content=user)]


# ---- tests ---------------------------------------------------------------


@pytest.mark.unit
async def test_single_turn_no_tools_returns_content() -> None:
  llm = MockModel(scripted=[ModelResponse(content="hello back")])
  result = await run(
    llm=llm,
    messages=_make_messages(),
    tools=ToolRegistry(),
    events=EventBus(),
    run_id="r1",
  )
  assert result.content == "hello back"
  assert result.turns == 1
  assert result.exit_reason == "natural"


@pytest.mark.unit
async def test_two_turns_with_tool_call() -> None:
  llm = MockModel(
    scripted=[
      ModelResponse(
        content=None,
        tool_executions=[ToolExecution(tool_call_id="c1", tool_name="add", tool_args={"x": 2, "y": 3})],
      ),
      ModelResponse(content="The sum is 5"),
    ]
  )
  reg = ToolRegistry(tools=[add])
  bus = EventBus()
  fired: list[Event] = []
  bus.subscribe(fired.append)

  result = await run(llm=llm, messages=_make_messages(), tools=reg, events=bus, run_id="r1")

  assert result.content == "The sum is 5"
  assert result.turns == 2
  assert result.exit_reason == "natural"

  tool_starts = [e for e in fired if isinstance(e, ToolCallStarted)]
  assert len(tool_starts) == 1
  assert tool_starts[0].call.name == "add"


@pytest.mark.unit
async def test_max_turns_exit_when_model_keeps_calling_tools() -> None:
  llm = MockModel(
    scripted=[
      ModelResponse(
        content=None,
        tool_executions=[ToolExecution(tool_call_id=f"c{i}", tool_name="add", tool_args={"x": 1, "y": 1})],
      )
      for i in range(5)
    ]
  )
  result = await run(
    llm=llm,
    messages=_make_messages(),
    tools=ToolRegistry(tools=[add]),
    events=EventBus(),
    max_turns=3,
    run_id="r1",
  )
  assert result.content is None
  assert result.turns == 3
  assert result.exit_reason == "max_turns"


@pytest.mark.unit
async def test_streaming_emits_chunk_events() -> None:
  llm = MockModel(
    stream_scripts=[
      [
        ModelResponse(content="hello "),
        ModelResponse(content="world"),
      ],
    ]
  )
  bus = EventBus()
  chunks: list[StreamChunkEvent] = []
  bus.subscribe(lambda e: chunks.append(e) if isinstance(e, StreamChunkEvent) else None)

  fired: list[Event] = []
  bus.subscribe(fired.append)

  result = await run(
    llm=llm,
    messages=_make_messages(),
    tools=ToolRegistry(),
    events=bus,
    stream=True,
    run_id="r1",
  )

  assert result.content == "hello world"
  assert [c.data for c in chunks] == ["hello ", "world"]
  assert all(c.kind == "content" for c in chunks)
  # Full event sequence: TurnStarted, chunks..., ModelResponded, RunCompleted
  assert any(isinstance(e, TurnStarted) for e in fired)
  assert any(isinstance(e, ModelResponded) for e in fired)
  assert any(isinstance(e, RunCompleted) for e in fired)


@pytest.mark.unit
async def test_tool_failure_is_passed_back_and_loop_continues() -> None:
  llm = MockModel(
    scripted=[
      ModelResponse(
        content=None,
        tool_executions=[ToolExecution(tool_call_id="c1", tool_name="boom", tool_args={})],
      ),
      ModelResponse(content="caught the error and moved on"),
    ]
  )
  result = await run(
    llm=llm,
    messages=_make_messages(),
    tools=ToolRegistry(tools=[boom]),
    events=EventBus(),
    run_id="r1",
  )
  assert result.content == "caught the error and moved on"
  assert result.turns == 2
  # The tool error message must be in the transcript so the model can react.
  tool_msgs = [m for m in result.messages if m.role == "tool"]
  assert len(tool_msgs) == 1
  assert "Error:" in (tool_msgs[0].content or "")
  assert "kaboom" in (tool_msgs[0].content or "")


@pytest.mark.unit
async def test_run_errored_when_model_raises() -> None:
  llm = MockModel(raise_on_call=RuntimeError("provider down"))
  bus = EventBus()
  fired: list[Event] = []
  bus.subscribe(fired.append)

  result = await run(llm=llm, messages=_make_messages(), tools=ToolRegistry(), events=bus, run_id="r1")

  assert result.content is None
  assert result.exit_reason == "error"
  errs = [e for e in fired if isinstance(e, RunErrored)]
  assert len(errs) == 1
  assert errs[0].error == "provider down"
  # Should not have emitted RunCompleted on this path
  completes = [e for e in fired if isinstance(e, RunCompleted)]
  assert len(completes) == 0
  # But TurnStarted + ModelResponded? — only TurnStarted since model failed before responding
  starts = [e for e in fired if isinstance(e, TurnStarted)]
  assert len(starts) == 1
  responds = [e for e in fired if isinstance(e, ModelResponded)]
  assert len(responds) == 0
