"""
Unit tests for ToolExecution, ModelResponse, ModelResponseEvent, and FileType.

Tests pure construction, field storage, serialization, deserialization,
and the is_paused property. No API calls. No external dependencies.

Covers:
  - ToolExecution creation with defaults
  - ToolExecution fields: tool_call_id, tool_name, tool_args, result
  - ToolExecution.is_paused property (requires_confirmation / requires_user_input / external)
  - ToolExecution.to_dict() serialization
  - ToolExecution.from_dict() round-trip
  - ToolExecution HITL fields (confirmed, confirmation_note, answered)
  - ModelResponse creation with defaults
  - ModelResponse.tool_calls and tool_executions
  - ModelResponse.to_dict() / from_dict() round-trip
  - ModelResponseEvent enum values
  - FileType enum values
"""

import pytest

from definable.model.response import (
  FileType,
  ModelResponse,
  ModelResponseEvent,
  ToolExecution,
)
from definable.model.metrics import Metrics


# ---------------------------------------------------------------------------
# ToolExecution: basic creation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToolExecutionCreation:
  """ToolExecution creates with sensible defaults."""

  def test_default_creation(self):
    te = ToolExecution()
    assert te.tool_call_id is None
    assert te.tool_name is None
    assert te.tool_args is None
    assert te.result is None

  def test_creation_with_all_fields(self):
    te = ToolExecution(
      tool_call_id="call_123",
      tool_name="get_weather",
      tool_args={"city": "NYC"},
      result="Sunny, 75F",
    )
    assert te.tool_call_id == "call_123"
    assert te.tool_name == "get_weather"
    assert te.tool_args == {"city": "NYC"}
    assert te.result == "Sunny, 75F"


# ---------------------------------------------------------------------------
# ToolExecution: field storage
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToolExecutionFields:
  """ToolExecution stores all fields accurately."""

  def test_tool_call_id_stored(self):
    te = ToolExecution(tool_call_id="abc")
    assert te.tool_call_id == "abc"

  def test_tool_name_stored(self):
    te = ToolExecution(tool_name="search")
    assert te.tool_name == "search"

  def test_tool_args_stored(self):
    args = {"query": "hello", "limit": 5}
    te = ToolExecution(tool_args=args)
    assert te.tool_args == args

  def test_tool_call_error_stored(self):
    te = ToolExecution(tool_call_error=True)
    assert te.tool_call_error is True

  def test_result_stored(self):
    te = ToolExecution(result="success")
    assert te.result == "success"

  def test_child_run_id_default_none(self):
    te = ToolExecution()
    assert te.child_run_id is None

  def test_child_run_id_stored(self):
    te = ToolExecution(child_run_id="run_456")
    assert te.child_run_id == "run_456"

  def test_stop_after_tool_call_default_false(self):
    te = ToolExecution()
    assert te.stop_after_tool_call is False

  def test_stop_after_tool_call_set_true(self):
    te = ToolExecution(stop_after_tool_call=True)
    assert te.stop_after_tool_call is True

  def test_created_at_is_int(self):
    te = ToolExecution()
    assert isinstance(te.created_at, int)

  def test_metrics_default_none(self):
    te = ToolExecution()
    assert te.metrics is None

  def test_metrics_stored(self):
    m = Metrics(input_tokens=10, output_tokens=20)
    te = ToolExecution(metrics=m)
    assert te.metrics is not None
    assert te.metrics.input_tokens == 10


# ---------------------------------------------------------------------------
# ToolExecution: HITL fields
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToolExecutionHITLFields:
  """ToolExecution stores human-in-the-loop control fields."""

  def test_requires_confirmation_default_none(self):
    te = ToolExecution()
    assert te.requires_confirmation is None

  def test_requires_confirmation_stored(self):
    te = ToolExecution(requires_confirmation=True)
    assert te.requires_confirmation is True

  def test_confirmed_default_none(self):
    te = ToolExecution()
    assert te.confirmed is None

  def test_confirmed_stored(self):
    te = ToolExecution(confirmed=True)
    assert te.confirmed is True

  def test_confirmation_note_stored(self):
    te = ToolExecution(confirmation_note="Approved by admin")
    assert te.confirmation_note == "Approved by admin"

  def test_requires_user_input_default_none(self):
    te = ToolExecution()
    assert te.requires_user_input is None

  def test_answered_default_none(self):
    te = ToolExecution()
    assert te.answered is None

  def test_external_execution_required_default_none(self):
    te = ToolExecution()
    assert te.external_execution_required is None


# ---------------------------------------------------------------------------
# ToolExecution: is_paused property
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToolExecutionIsPaused:
  """ToolExecution.is_paused reflects confirmation/input/external flags."""

  def test_not_paused_by_default(self):
    te = ToolExecution()
    assert te.is_paused is False

  def test_paused_when_requires_confirmation(self):
    te = ToolExecution(requires_confirmation=True)
    assert te.is_paused is True

  def test_paused_when_requires_user_input(self):
    te = ToolExecution(requires_user_input=True)
    assert te.is_paused is True

  def test_paused_when_external_execution_required(self):
    te = ToolExecution(external_execution_required=True)
    assert te.is_paused is True

  def test_not_paused_when_all_false(self):
    te = ToolExecution(
      requires_confirmation=False,
      requires_user_input=False,
      external_execution_required=False,
    )
    assert te.is_paused is False


# ---------------------------------------------------------------------------
# ToolExecution: to_dict()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToolExecutionToDict:
  """ToolExecution.to_dict() serializes all fields."""

  def test_basic_serialization(self):
    te = ToolExecution(
      tool_call_id="call_1",
      tool_name="search",
      tool_args={"query": "test"},
      result="found",
    )
    d = te.to_dict()
    assert d["tool_call_id"] == "call_1"
    assert d["tool_name"] == "search"
    assert d["tool_args"] == {"query": "test"}
    assert d["result"] == "found"

  def test_includes_created_at(self):
    te = ToolExecution()
    d = te.to_dict()
    assert "created_at" in d

  def test_includes_stop_after_tool_call(self):
    te = ToolExecution(stop_after_tool_call=True)
    d = te.to_dict()
    assert d["stop_after_tool_call"] is True

  def test_metrics_serialized_when_present(self):
    m = Metrics(input_tokens=5, output_tokens=10, total_tokens=15)
    te = ToolExecution(metrics=m)
    d = te.to_dict()
    assert "metrics" in d
    assert d["metrics"]["input_tokens"] == 5


# ---------------------------------------------------------------------------
# ToolExecution: from_dict() round-trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToolExecutionFromDict:
  """ToolExecution.from_dict() reconstructs from a dictionary."""

  def test_basic_round_trip(self):
    original = ToolExecution(
      tool_call_id="call_1",
      tool_name="search",
      tool_args={"q": "test"},
      result="ok",
    )
    d = original.to_dict()
    restored = ToolExecution.from_dict(d)
    assert restored.tool_call_id == "call_1"
    assert restored.tool_name == "search"
    assert restored.tool_args == {"q": "test"}
    assert restored.result == "ok"

  def test_preserves_stop_after_tool_call(self):
    original = ToolExecution(stop_after_tool_call=True)
    d = original.to_dict()
    restored = ToolExecution.from_dict(d)
    assert restored.stop_after_tool_call is True

  def test_preserves_hitl_fields(self):
    original = ToolExecution(
      requires_confirmation=True,
      confirmed=False,
      confirmation_note="Pending",
    )
    d = original.to_dict()
    restored = ToolExecution.from_dict(d)
    assert restored.requires_confirmation is True
    assert restored.confirmed is False
    assert restored.confirmation_note == "Pending"

  def test_from_dict_with_empty_dict(self):
    restored = ToolExecution.from_dict({})
    assert restored.tool_call_id is None
    assert restored.tool_name is None
    assert restored.stop_after_tool_call is False


# ---------------------------------------------------------------------------
# ModelResponse: basic creation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelResponseCreation:
  """ModelResponse creates with sensible defaults."""

  def test_default_creation(self):
    resp = ModelResponse()
    assert resp.role is None
    assert resp.content is None
    assert resp.parsed is None
    assert resp.tool_calls == []
    assert resp.tool_executions == []

  def test_creation_with_content(self):
    resp = ModelResponse(role="assistant", content="Hello!")
    assert resp.role == "assistant"
    assert resp.content == "Hello!"

  def test_default_event_is_assistant_response(self):
    resp = ModelResponse()
    assert resp.event == ModelResponseEvent.assistant_response.value


# ---------------------------------------------------------------------------
# ModelResponse: fields
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelResponseFields:
  """ModelResponse stores all fields correctly."""

  def test_tool_calls_list(self):
    calls = [{"id": "call_1", "function": {"name": "foo", "arguments": "{}"}}]
    resp = ModelResponse(tool_calls=calls)
    assert len(resp.tool_calls) == 1

  def test_tool_executions_list(self):
    te = ToolExecution(tool_name="search", result="found")
    resp = ModelResponse(tool_executions=[te])
    assert resp.tool_executions is not None
    assert len(resp.tool_executions) == 1
    assert resp.tool_executions[0].tool_name == "search"

  def test_reasoning_content_stored(self):
    resp = ModelResponse(reasoning_content="I think therefore I am")
    assert resp.reasoning_content == "I think therefore I am"

  def test_provider_data_stored(self):
    resp = ModelResponse(provider_data={"model": "gpt-4o"})
    assert resp.provider_data is not None
    assert resp.provider_data["model"] == "gpt-4o"

  def test_response_usage_stored(self):
    m = Metrics(input_tokens=100, output_tokens=50)
    resp = ModelResponse(response_usage=m)
    assert resp.response_usage is not None
    assert resp.response_usage.input_tokens == 100

  def test_extra_stored(self):
    resp = ModelResponse(extra={"custom": "value"})
    assert resp.extra == {"custom": "value"}


# ---------------------------------------------------------------------------
# ModelResponse: to_dict() / from_dict()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelResponseSerialization:
  """ModelResponse serialization and deserialization."""

  def test_to_dict_includes_content(self):
    resp = ModelResponse(role="assistant", content="hi")
    d = resp.to_dict()
    assert d["role"] == "assistant"
    assert d["content"] == "hi"

  def test_to_dict_includes_tool_calls(self):
    calls = [{"id": "call_1", "function": {"name": "test", "arguments": "{}"}}]
    resp = ModelResponse(tool_calls=calls)
    d = resp.to_dict()
    assert len(d["tool_calls"]) == 1

  def test_from_dict_basic(self):
    d = {"role": "assistant", "content": "hello", "tool_calls": [], "tool_executions": []}
    resp = ModelResponse.from_dict(d)
    assert resp.role == "assistant"
    assert resp.content == "hello"

  def test_round_trip(self):
    original = ModelResponse(
      role="assistant",
      content="test content",
      reasoning_content="some reasoning",
    )
    d = original.to_dict()
    restored = ModelResponse.from_dict(d)
    assert restored.role == "assistant"
    assert restored.content == "test content"
    assert restored.reasoning_content == "some reasoning"


# ---------------------------------------------------------------------------
# ModelResponseEvent enum
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelResponseEvent:
  """ModelResponseEvent enum has expected values."""

  def test_tool_call_paused_value(self):
    assert ModelResponseEvent.tool_call_paused.value == "ToolCallPaused"

  def test_tool_call_started_value(self):
    assert ModelResponseEvent.tool_call_started.value == "ToolCallStarted"

  def test_tool_call_completed_value(self):
    assert ModelResponseEvent.tool_call_completed.value == "ToolCallCompleted"

  def test_assistant_response_value(self):
    assert ModelResponseEvent.assistant_response.value == "AssistantResponse"

  def test_is_string_enum(self):
    assert isinstance(ModelResponseEvent.assistant_response, str)


# ---------------------------------------------------------------------------
# FileType enum
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFileType:
  """FileType enum has expected media type values."""

  def test_mp4_value(self):
    assert FileType.MP4.value == "mp4"

  def test_gif_value(self):
    assert FileType.GIF.value == "gif"

  def test_mp3_value(self):
    assert FileType.MP3.value == "mp3"

  def test_wav_value(self):
    assert FileType.WAV.value == "wav"

  def test_is_string_enum(self):
    assert isinstance(FileType.MP4, str)
