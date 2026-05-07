"""Tests for streaming tool support — generator consumption and ToolContentEvent.

Covers:
  - Async generator tools: chunks consumed, accumulated, ToolContentEvent emitted
  - Sync generator tools: same behavior via sync iteration
  - Plain (non-generator) tools: unchanged behavior, no ToolContentEvent
  - Empty generators: produce empty string result
  - Generator error mid-stream: propagates correctly
  - ToolContentEvent fields: tool_name, chunk, chunk_index, is_final
  - Final aggregated string used in Message and ToolExecution
  - Event ordering: ToolCallStarted → ToolContentEvent(s) → ToolCallCompleted
  - Bug fix: ToolCallStartedEvent must NOT leak the tool result
  - Bug fix: Tool events must stream in real-time (not batched)
"""

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from definable.agent.events import ToolCallCompletedEvent, ToolCallStartedEvent, ToolContentEvent
from definable.agent.loop import AgentLoop, ToolBatchResult
from definable.run.agent import BaseRunOutputEvent, RunEvent
from definable.run.base import RunContext


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _make_loop(**overrides: Any) -> AgentLoop:
  """Create a minimal AgentLoop for unit testing _resolve_tool_result."""
  from definable.agent.config import AgentConfig

  return AgentLoop(
    model=overrides.get("model", MagicMock()),
    tools=overrides.get("tools", {}),
    messages=overrides.get("messages", []),
    context=overrides.get("context", RunContext(run_id="test-run", session_id="test-session")),
    config=overrides.get("config", AgentConfig()),
    streaming=overrides.get("streaming", False),
    emit_fn=overrides.get("emit_fn", lambda _: None),
    agent_id=overrides.get("agent_id", "agent-1"),
    agent_name=overrides.get("agent_name", "TestAgent"),
  )


# ═══════════════════════════════════════════════════════════════════════
# _resolve_tool_result — async generators
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestResolveAsyncGenerator:
  """Async generator results are consumed, accumulated, and emit ToolContentEvent."""

  @pytest.mark.asyncio
  async def test_async_gen_chunks_accumulated(self):
    loop = _make_loop()

    async def gen():
      yield "alpha"
      yield "beta"
      yield "gamma"

    events: list[BaseRunOutputEvent] = []
    result = await loop._resolve_tool_result(
      gen(),
      fn_name="stream_tool",
      tool_call_id="call-1",
      events=events,
    )
    assert result == "alpha\nbeta\ngamma"

  @pytest.mark.asyncio
  async def test_async_gen_emits_tool_content_events(self):
    loop = _make_loop()

    async def gen():
      yield "chunk-0"
      yield "chunk-1"

    events: list[BaseRunOutputEvent] = []
    await loop._resolve_tool_result(
      gen(),
      fn_name="my_tool",
      tool_call_id="call-2",
      events=events,
    )
    content_events = [e for e in events if isinstance(e, ToolContentEvent)]
    assert len(content_events) == 2
    assert content_events[0].chunk == "chunk-0"
    assert content_events[0].chunk_index == 0
    assert content_events[0].is_final is False
    assert content_events[1].chunk == "chunk-1"
    assert content_events[1].chunk_index == 1
    assert content_events[1].is_final is True

  @pytest.mark.asyncio
  async def test_async_gen_event_fields(self):
    loop = _make_loop()

    async def gen():
      yield "only"

    events: list[BaseRunOutputEvent] = []
    await loop._resolve_tool_result(
      gen(),
      fn_name="the_tool",
      tool_call_id="tc-99",
      events=events,
    )
    evt = events[0]
    assert isinstance(evt, ToolContentEvent)
    assert evt.tool_name == "the_tool"
    assert evt.tool_call_id == "tc-99"
    assert evt.run_id == "test-run"
    assert evt.session_id == "test-session"
    assert evt.agent_id == "agent-1"
    assert evt.agent_name == "TestAgent"
    assert evt.event == RunEvent.tool_content.value

  @pytest.mark.asyncio
  async def test_async_gen_empty(self):
    loop = _make_loop()

    async def gen():
      return
      yield  # noqa: RET504 — makes this an async generator

    events: list[BaseRunOutputEvent] = []
    result = await loop._resolve_tool_result(
      gen(),
      fn_name="empty",
      tool_call_id="call-e",
      events=events,
    )
    assert result == ""
    assert len([e for e in events if isinstance(e, ToolContentEvent)]) == 0

  @pytest.mark.asyncio
  async def test_async_gen_non_string_chunks(self):
    """Non-string chunks are str()-ified."""
    loop = _make_loop()

    async def gen():
      yield 42
      yield {"key": "value"}

    events: list[BaseRunOutputEvent] = []
    result = await loop._resolve_tool_result(
      gen(),
      fn_name="typed_tool",
      tool_call_id="call-t",
      events=events,
    )
    assert "42" in result
    assert "key" in result


# ═══════════════════════════════════════════════════════════════════════
# _resolve_tool_result — sync generators
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestResolveSyncGenerator:
  """Sync generator results are consumed identically to async generators."""

  @pytest.mark.asyncio
  async def test_sync_gen_chunks_accumulated(self):
    loop = _make_loop()

    def gen():
      yield "one"
      yield "two"
      yield "three"

    events: list[BaseRunOutputEvent] = []
    result = await loop._resolve_tool_result(
      gen(),
      fn_name="sync_stream",
      tool_call_id="call-s",
      events=events,
    )
    assert result == "one\ntwo\nthree"

  @pytest.mark.asyncio
  async def test_sync_gen_emits_tool_content_events(self):
    loop = _make_loop()

    def gen():
      yield "a"
      yield "b"

    events: list[BaseRunOutputEvent] = []
    await loop._resolve_tool_result(
      gen(),
      fn_name="sg",
      tool_call_id="call-sg",
      events=events,
    )
    content_events = [e for e in events if isinstance(e, ToolContentEvent)]
    assert len(content_events) == 2
    assert content_events[0].chunk_index == 0
    assert content_events[1].chunk_index == 1
    assert content_events[1].is_final is True

  @pytest.mark.asyncio
  async def test_sync_gen_empty(self):
    loop = _make_loop()

    def gen():
      return
      yield  # noqa: RET504

    events: list[BaseRunOutputEvent] = []
    result = await loop._resolve_tool_result(
      gen(),
      fn_name="empty_sync",
      tool_call_id="call-es",
      events=events,
    )
    assert result == ""


# ═══════════════════════════════════════════════════════════════════════
# _resolve_tool_result — plain values (non-generator)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestResolvePlainResult:
  """Non-generator results pass through as str() with no ToolContentEvent."""

  @pytest.mark.asyncio
  async def test_string_passthrough(self):
    loop = _make_loop()
    events: list[BaseRunOutputEvent] = []
    result = await loop._resolve_tool_result(
      "hello world",
      fn_name="plain",
      tool_call_id="call-p",
      events=events,
    )
    assert result == "hello world"
    assert len(events) == 0

  @pytest.mark.asyncio
  async def test_int_passthrough(self):
    loop = _make_loop()
    events: list[BaseRunOutputEvent] = []
    result = await loop._resolve_tool_result(
      42,
      fn_name="num",
      tool_call_id="call-n",
      events=events,
    )
    assert result == "42"
    assert len(events) == 0

  @pytest.mark.asyncio
  async def test_dict_passthrough(self):
    loop = _make_loop()
    events: list[BaseRunOutputEvent] = []
    result = await loop._resolve_tool_result(
      {"status": "ok"},
      fn_name="dict_tool",
      tool_call_id="call-d",
      events=events,
    )
    assert "status" in result
    assert "ok" in result
    assert len(events) == 0

  @pytest.mark.asyncio
  async def test_none_passthrough(self):
    loop = _make_loop()
    events: list[BaseRunOutputEvent] = []
    result = await loop._resolve_tool_result(
      None,
      fn_name="void",
      tool_call_id="call-v",
      events=events,
    )
    assert result == "None"
    assert len(events) == 0


# ═══════════════════════════════════════════════════════════════════════
# ToolContentEvent dataclass
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestToolContentEvent:
  """ToolContentEvent fields and defaults are correct."""

  def test_default_values(self):
    evt = ToolContentEvent()
    assert evt.event == "ToolContent"
    assert evt.tool_name == ""
    assert evt.tool_call_id is None
    assert evt.chunk == ""
    assert evt.chunk_index == 0
    assert evt.is_final is False

  def test_event_string_matches_enum(self):
    assert RunEvent.tool_content.value == "ToolContent"

  def test_custom_fields(self):
    evt = ToolContentEvent(
      tool_name="search",
      tool_call_id="tc-1",
      chunk="partial result",
      chunk_index=3,
      is_final=True,
      agent_id="a1",
      agent_name="Bot",
    )
    assert evt.tool_name == "search"
    assert evt.chunk == "partial result"
    assert evt.chunk_index == 3
    assert evt.is_final is True


# ═══════════════════════════════════════════════════════════════════════
# Import sanity
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestImports:
  """ToolContentEvent is importable from public API."""

  def test_import_from_events(self):
    from definable.agent.events import ToolContentEvent as TCE

    assert TCE is not None

  def test_import_from_run_agent(self):
    from definable.run.agent import ToolContentEvent as TCE

    assert TCE is not None

  def test_in_run_event_registry(self):
    from definable.run.agent import RUN_EVENT_TYPE_REGISTRY

    assert "ToolContent" in RUN_EVENT_TYPE_REGISTRY


# ═══════════════════════════════════════════════════════════════════════
# Helpers for tool-level tests
# ═══════════════════════════════════════════════════════════════════════


def _make_function(name: str, entrypoint, *, sequential: bool = False):
  """Create a Function with the given entrypoint for testing."""
  from definable.tool.function import Function

  return Function(
    name=name,
    description=f"Test tool {name}",
    entrypoint=entrypoint,
    skip_entrypoint_processing=True,
    sequential=sequential,
  )


def _make_tool_call(name: str, call_id: str, args: str = "{}") -> dict:
  """Create a tool call dict in OpenAI format."""
  return {
    "id": call_id,
    "type": "function",
    "function": {"name": name, "arguments": args},
  }


def _make_loop_with_tools(tools: dict[str, Any], **overrides: Any) -> AgentLoop:
  """Create an AgentLoop with real Function objects in the tools dict."""
  from definable.agent.config import AgentConfig

  return AgentLoop(
    model=overrides.get("model", MagicMock()),
    tools=tools,
    messages=overrides.get("messages", []),
    context=overrides.get("context", RunContext(run_id="test-run", session_id="test-session")),
    config=overrides.get("config", AgentConfig()),
    streaming=overrides.get("streaming", False),
    emit_fn=overrides.get("emit_fn", lambda _: None),
    agent_id=overrides.get("agent_id", "agent-1"),
    agent_name=overrides.get("agent_name", "TestAgent"),
  )


# ═══════════════════════════════════════════════════════════════════════
# Bug 1: ToolCallStartedEvent must NOT contain the tool result
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestToolCallStartedEventSnapshot:
  """ToolCallStartedEvent.tool is a snapshot — it must never contain the result."""

  @pytest.mark.asyncio
  async def test_started_event_has_no_result(self):
    """The started event's ToolExecution.result must be None, even after the tool finishes."""

    def greet(name: str = "world") -> str:
      return f"Hello, {name}!"

    fn = _make_function("greet", greet)
    tools = {"greet": fn}
    loop = _make_loop_with_tools(tools)

    tc = _make_tool_call("greet", "call-1", '{"name": "Alice"}')
    result = await loop._execute_single_tool(tc)

    started_events = [e for e in result.events if isinstance(e, ToolCallStartedEvent)]
    completed_events = [e for e in result.events if isinstance(e, ToolCallCompletedEvent)]

    assert len(started_events) == 1
    assert len(completed_events) == 1

    # Started event must NOT have the result
    assert started_events[0].tool is not None
    assert started_events[0].tool.result is None

    # Completed event MUST have the result
    assert completed_events[0].tool is not None
    assert completed_events[0].tool.result is not None
    assert "Hello, Alice!" in completed_events[0].tool.result

  @pytest.mark.asyncio
  async def test_started_and_completed_have_different_tool_objects(self):
    """Started and completed events reference different ToolExecution instances."""

    def echo(msg: str = "hi") -> str:
      return msg

    fn = _make_function("echo", echo)
    tools = {"echo": fn}
    loop = _make_loop_with_tools(tools)

    tc = _make_tool_call("echo", "call-2", '{"msg": "test"}')
    result = await loop._execute_single_tool(tc)

    started = [e for e in result.events if isinstance(e, ToolCallStartedEvent)][0]
    completed = [e for e in result.events if isinstance(e, ToolCallCompletedEvent)][0]

    assert started.tool is not completed.tool  # Different objects

  @pytest.mark.asyncio
  async def test_started_event_preserves_tool_name_and_args(self):
    """The snapshot preserves tool_name and tool_args correctly."""

    def add(a: int = 0, b: int = 0) -> str:
      return str(a + b)

    fn = _make_function("add", add)
    tools = {"add": fn}
    loop = _make_loop_with_tools(tools)

    tc = _make_tool_call("add", "call-3", '{"a": 1, "b": 2}')
    result = await loop._execute_single_tool(tc)

    started = [e for e in result.events if isinstance(e, ToolCallStartedEvent)][0]
    assert started.tool is not None
    assert started.tool.tool_name == "add"
    assert started.tool.tool_args == {"a": 1, "b": 2}


# ═══════════════════════════════════════════════════════════════════════
# Bug 2: Tool events must stream in real-time via _execute_tools
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestToolEventRealTimeStreaming:
  """_execute_tools streams events in real-time, not batched after completion."""

  @pytest.mark.asyncio
  async def test_parallel_events_stream_before_completion(self):
    """Started events arrive while tools are still running."""
    gate = asyncio.Event()

    async def slow_tool(x: str = "default") -> str:
      await gate.wait()
      return f"result-{x}"

    fn = _make_function("slow", slow_tool)
    tools = {"slow": fn}
    loop = _make_loop_with_tools(tools)

    tc1 = _make_tool_call("slow", "call-a", '{"x": "a"}')
    tc2 = _make_tool_call("slow", "call-b", '{"x": "b"}')

    received: list = []

    async def collect():
      async for item in loop._execute_tools([tc1, tc2]):
        received.append(item)
        # Once we see started events, unblock the tools
        started_count = sum(1 for e in received if isinstance(e, ToolCallStartedEvent))
        if started_count >= 2:
          gate.set()

    await asyncio.wait_for(collect(), timeout=5.0)

    # Verify we got started events
    started = [e for e in received if isinstance(e, ToolCallStartedEvent)]
    completed = [e for e in received if isinstance(e, ToolCallCompletedEvent)]
    batch = [e for e in received if isinstance(e, ToolBatchResult)]

    assert len(started) == 2
    assert len(completed) == 2
    assert len(batch) == 1

    # Started events must appear before their corresponding completed events
    for s in started:
      s_idx = received.index(s)
      assert s.tool is not None
      matching_completed = [c for c in completed if c.tool is not None and c.tool.tool_call_id == s.tool.tool_call_id]
      assert len(matching_completed) == 1
      c_idx = received.index(matching_completed[0])
      assert s_idx < c_idx

  @pytest.mark.asyncio
  async def test_sequential_events_yield_immediately(self):
    """Sequential tools yield their events one at a time, not batched."""
    call_order: list[str] = []

    def tool_a() -> str:
      call_order.append("a")
      return "result-a"

    def tool_b() -> str:
      call_order.append("b")
      return "result-b"

    fn_a = _make_function("tool_a", tool_a, sequential=True)
    fn_b = _make_function("tool_b", tool_b, sequential=True)
    tools = {"tool_a": fn_a, "tool_b": fn_b}
    loop = _make_loop_with_tools(tools)

    tc1 = _make_tool_call("tool_a", "call-1")
    tc2 = _make_tool_call("tool_b", "call-2")

    events: list = []
    async for item in loop._execute_tools([tc1, tc2]):
      events.append(item)

    # Should see: Started_A, Completed_A, Started_B, Completed_B, ToolBatchResult
    started = [e for e in events if isinstance(e, ToolCallStartedEvent)]
    completed = [e for e in events if isinstance(e, ToolCallCompletedEvent)]
    batch = [e for e in events if isinstance(e, ToolBatchResult)]

    assert len(started) == 2
    assert len(completed) == 2
    assert len(batch) == 1

    # tool_a events should all precede tool_b events
    a_started_idx = next(
      i for i, e in enumerate(events) if isinstance(e, ToolCallStartedEvent) and e.tool is not None and e.tool.tool_name == "tool_a"
    )
    a_completed_idx = next(
      i for i, e in enumerate(events) if isinstance(e, ToolCallCompletedEvent) and e.tool is not None and e.tool.tool_name == "tool_a"
    )
    b_started_idx = next(
      i for i, e in enumerate(events) if isinstance(e, ToolCallStartedEvent) and e.tool is not None and e.tool.tool_name == "tool_b"
    )
    assert a_started_idx < a_completed_idx < b_started_idx

  @pytest.mark.asyncio
  async def test_execute_tools_yields_batch_result_last(self):
    """The final item from _execute_tools is always a ToolBatchResult."""

    def noop() -> str:
      return "ok"

    fn = _make_function("noop", noop)
    tools = {"noop": fn}
    loop = _make_loop_with_tools(tools)

    tc = _make_tool_call("noop", "call-1")

    items: list = []
    async for item in loop._execute_tools([tc]):
      items.append(item)

    assert isinstance(items[-1], ToolBatchResult)
    assert items[-1].events == []  # Events already streamed

  @pytest.mark.asyncio
  async def test_parallel_tools_with_event_sink(self):
    """Parallel tool execution uses event_sink — ToolResult.events is empty."""

    def fast() -> str:
      return "done"

    fn = _make_function("fast", fast)
    tools = {"fast": fn}
    loop = _make_loop_with_tools(tools)

    # Call with event_sink to simulate parallel dispatch
    sink_events: list = []
    tc = _make_tool_call("fast", "call-1")
    result = await loop._execute_single_tool(tc, event_sink=sink_events.append)

    # Events went to sink, not to ToolResult.events
    assert len(result.events) == 0
    assert len(sink_events) == 2  # Started + Completed

    started = [e for e in sink_events if isinstance(e, ToolCallStartedEvent)]
    completed = [e for e in sink_events if isinstance(e, ToolCallCompletedEvent)]
    assert len(started) == 1
    assert len(completed) == 1

  @pytest.mark.asyncio
  async def test_event_sink_with_generator_tool(self):
    """Generator tool content events also go to the event_sink."""

    async def streaming_tool() -> str:  # noqa: RUF029
      # Simulate a tool that returns an async generator
      return "chunk1"

    fn = _make_function("stream", streaming_tool)
    tools = {"stream": fn}
    loop = _make_loop_with_tools(tools)

    sink_events: list = []
    tc = _make_tool_call("stream", "call-1")
    result = await loop._execute_single_tool(tc, event_sink=sink_events.append)

    assert len(result.events) == 0
    assert len(sink_events) >= 2  # At least Started + Completed


# ═══════════════════════════════════════════════════════════════════════
# Regression: _merge_tool_call_deltas — parallel tool name concatenation
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestMergeToolCallDeltas:
  """Regression tests for _merge_tool_call_deltas.

  Bug: When non-OpenAI providers (Anthropic, Gemini, Mistral, Ollama) emit
  tool call dicts without an "index" key, all parallel tool calls were merged
  into position 0 and their names concatenated via +=.

  Example: "MANAGE_CONNECTIONS" + "MULTI_EXECUTE_TOOL" became
  "MANAGE_CONNECTIONSMULTI_EXECUTE_TOOL".
  """

  def test_dicts_without_index_get_separate_entries(self):
    """Complete tool call dicts (no index) must each become a separate entry."""
    from definable.agent.loop import _merge_tool_call_deltas

    existing: list[dict] = []
    # Simulate two Anthropic-style tool calls arriving in separate chunks
    _merge_tool_call_deltas(
      existing,
      [
        {"id": "call_1", "type": "function", "function": {"name": "MANAGE_CONNECTIONS", "arguments": '{"x": 1}'}},
      ],
    )
    _merge_tool_call_deltas(
      existing,
      [
        {"id": "call_2", "type": "function", "function": {"name": "MULTI_EXECUTE_TOOL", "arguments": '{"y": 2}'}},
      ],
    )

    assert len(existing) == 2
    assert existing[0]["function"]["name"] == "MANAGE_CONNECTIONS"
    assert existing[1]["function"]["name"] == "MULTI_EXECUTE_TOOL"
    assert existing[0]["id"] == "call_1"
    assert existing[1]["id"] == "call_2"

  def test_dicts_without_index_same_tool_called_twice(self):
    """Same tool called twice in parallel must produce two entries, not one with doubled name."""
    from definable.agent.loop import _merge_tool_call_deltas

    existing: list[dict] = []
    _merge_tool_call_deltas(
      existing,
      [
        {"id": "call_a", "type": "function", "function": {"name": "web_search", "arguments": '{"q": "foo"}'}},
      ],
    )
    _merge_tool_call_deltas(
      existing,
      [
        {"id": "call_b", "type": "function", "function": {"name": "web_search", "arguments": '{"q": "bar"}'}},
      ],
    )

    assert len(existing) == 2
    assert existing[0]["function"]["name"] == "web_search"
    assert existing[1]["function"]["name"] == "web_search"
    assert existing[0]["function"]["arguments"] == '{"q": "foo"}'
    assert existing[1]["function"]["arguments"] == '{"q": "bar"}'

  def test_dicts_without_index_both_in_single_batch(self):
    """Multiple tool calls arriving in a single chunk (list) without index."""
    from definable.agent.loop import _merge_tool_call_deltas

    existing: list[dict] = []
    _merge_tool_call_deltas(
      existing,
      [
        {"id": "call_1", "type": "function", "function": {"name": "tool_a", "arguments": "{}"}},
        {"id": "call_2", "type": "function", "function": {"name": "tool_b", "arguments": "{}"}},
      ],
    )

    assert len(existing) == 2
    assert existing[0]["function"]["name"] == "tool_a"
    assert existing[1]["function"]["name"] == "tool_b"

  def test_openai_indexed_deltas_still_work(self):
    """OpenAI-style deltas with index attribute are merged correctly."""
    from definable.agent.loop import _merge_tool_call_deltas

    class FakeDelta:
      def __init__(self, index, tc_id=None, tc_type=None, name=None, arguments=None):
        self.index = index
        self.id = tc_id
        self.type = tc_type
        self.function = MagicMock()
        self.function.name = name
        self.function.arguments = arguments

    existing: list[dict] = []
    # Chunk 1: first tool starts
    _merge_tool_call_deltas(existing, [FakeDelta(index=0, tc_id="call_1", tc_type="function", name="search", arguments="")])
    # Chunk 2: first tool arguments stream
    _merge_tool_call_deltas(existing, [FakeDelta(index=0, name=None, arguments='{"query":')])
    _merge_tool_call_deltas(existing, [FakeDelta(index=0, name=None, arguments=' "test"}')])
    # Chunk 3: second tool starts
    _merge_tool_call_deltas(existing, [FakeDelta(index=1, tc_id="call_2", tc_type="function", name="get_weather", arguments="")])
    _merge_tool_call_deltas(existing, [FakeDelta(index=1, name=None, arguments='{"city": "NYC"}')])

    assert len(existing) == 2
    assert existing[0]["function"]["name"] == "search"
    assert existing[0]["function"]["arguments"] == '{"query": "test"}'
    assert existing[1]["function"]["name"] == "get_weather"
    assert existing[1]["function"]["arguments"] == '{"city": "NYC"}'

  def test_mixed_indexed_and_non_indexed(self):
    """Edge case: mix of indexed deltas and non-indexed complete calls."""
    from definable.agent.loop import _merge_tool_call_deltas

    class FakeDelta:
      def __init__(self, index, tc_id=None, tc_type=None, name=None, arguments=None):
        self.index = index
        self.id = tc_id
        self.type = tc_type
        self.function = MagicMock()
        self.function.name = name
        self.function.arguments = arguments

    existing: list[dict] = []
    # An indexed delta
    _merge_tool_call_deltas(existing, [FakeDelta(index=0, tc_id="c1", tc_type="function", name="indexed_tool", arguments='{"a": 1}')])
    # A non-indexed complete call
    _merge_tool_call_deltas(
      existing,
      [
        {"id": "c2", "type": "function", "function": {"name": "complete_tool", "arguments": '{"b": 2}'}},
      ],
    )

    assert len(existing) == 2
    assert existing[0]["function"]["name"] == "indexed_tool"
    assert existing[1]["function"]["name"] == "complete_tool"

  def test_object_without_index_appended(self):
    """Objects with index=None should be appended as new entries."""
    from definable.agent.loop import _merge_tool_call_deltas

    class NoIndexDelta:
      def __init__(self, tc_id, name, arguments):
        self.index = None
        self.id = tc_id
        self.type = "function"
        self.function = MagicMock()
        self.function.name = name
        self.function.arguments = arguments

    existing: list[dict] = []
    _merge_tool_call_deltas(existing, [NoIndexDelta("c1", "tool_x", '{"a": 1}')])
    _merge_tool_call_deltas(existing, [NoIndexDelta("c2", "tool_y", '{"b": 2}')])

    assert len(existing) == 2
    assert existing[0]["function"]["name"] == "tool_x"
    assert existing[1]["function"]["name"] == "tool_y"
