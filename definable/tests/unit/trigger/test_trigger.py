"""
Unit tests for trigger types: TriggerEvent, Webhook, EventTrigger, TriggerExecutor.

Tests pure dataclass/class behavior: construction, defaults, name property,
executor return-value dispatch. No server, no real Agent.

Covers:
  - TriggerEvent construction and defaults
  - Webhook construction, path normalization, name property
  - EventTrigger construction and name property
  - TriggerExecutor.execute() with None/str/dict/awaitable return values
  - TriggerExecutor handles handler exceptions gracefully
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from definable.agent.trigger.base import BaseTrigger, TriggerEvent
from definable.agent.trigger.event import EventTrigger
from definable.agent.trigger.webhook import Webhook


# ---------------------------------------------------------------------------
# TriggerEvent
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTriggerEvent:
  """TriggerEvent dataclass construction and defaults."""

  def test_defaults_body_none(self):
    event = TriggerEvent()
    assert event.body is None

  def test_defaults_headers_none(self):
    event = TriggerEvent()
    assert event.headers is None

  def test_defaults_source_empty(self):
    event = TriggerEvent()
    assert event.source == ""

  def test_defaults_timestamp_is_float(self):
    event = TriggerEvent()
    assert isinstance(event.timestamp, float)
    assert event.timestamp > 0

  def test_defaults_raw_none(self):
    event = TriggerEvent()
    assert event.raw is None

  def test_custom_body(self):
    event = TriggerEvent(body={"action": "push"})
    assert event.body["action"] == "push"

  def test_custom_headers(self):
    event = TriggerEvent(headers={"Content-Type": "application/json"})
    assert event.headers["Content-Type"] == "application/json"

  def test_custom_source(self):
    event = TriggerEvent(source="POST /webhook")
    assert event.source == "POST /webhook"

  def test_custom_timestamp(self):
    event = TriggerEvent(timestamp=1234567890.0)
    assert event.timestamp == 1234567890.0

  def test_custom_raw(self):
    raw = object()
    event = TriggerEvent(raw=raw)
    assert event.raw is raw


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWebhook:
  """Webhook trigger construction and name property."""

  def test_path_with_leading_slash(self):
    wh = Webhook("/github")
    assert wh.path == "/github"

  def test_path_without_leading_slash_gets_normalized(self):
    wh = Webhook("github")
    assert wh.path == "/github"

  def test_default_method_is_post(self):
    wh = Webhook("/github")
    assert wh.method == "POST"

  def test_custom_method_uppercased(self):
    wh = Webhook("/github", method="get")
    assert wh.method == "GET"

  def test_name_property_combines_method_and_path(self):
    wh = Webhook("/github")
    assert wh.name == "POST /github"

  def test_name_with_custom_method(self):
    wh = Webhook("/events", method="PUT")
    assert wh.name == "PUT /events"

  def test_default_auth_none(self):
    wh = Webhook("/github")
    assert wh.auth is None

  def test_custom_auth(self):
    auth = MagicMock()
    wh = Webhook("/github", auth=auth)
    assert wh.auth is auth

  def test_is_base_trigger_subclass(self):
    wh = Webhook("/github")
    assert isinstance(wh, BaseTrigger)


# ---------------------------------------------------------------------------
# EventTrigger
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEventTrigger:
  """EventTrigger construction and name property."""

  def test_event_name_stored(self):
    trigger = EventTrigger("user_signup")
    assert trigger.event_name == "user_signup"

  def test_name_property(self):
    trigger = EventTrigger("user_signup")
    assert trigger.name == "event(user_signup)"

  def test_is_base_trigger_subclass(self):
    trigger = EventTrigger("test")
    assert isinstance(trigger, BaseTrigger)


# ---------------------------------------------------------------------------
# TriggerExecutor
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTriggerExecutor:
  """TriggerExecutor.execute() processes handler return values."""

  @pytest.fixture
  def mock_agent(self):
    agent = MagicMock()
    agent.arun = AsyncMock(return_value="agent_result")
    return agent

  @pytest.fixture
  def executor(self, mock_agent):
    from definable.agent.trigger.executor import TriggerExecutor

    return TriggerExecutor(agent=mock_agent)

  @pytest.mark.asyncio
  async def test_returns_none_when_no_handler(self, executor):
    trigger = EventTrigger("test")
    trigger.handler = None
    event = TriggerEvent()
    result = await executor.execute(trigger, event)
    assert result is None

  @pytest.mark.asyncio
  async def test_handler_returns_none(self, executor):
    trigger = EventTrigger("test")
    trigger.handler = lambda event: None
    event = TriggerEvent()
    result = await executor.execute(trigger, event)
    assert result is None

  @pytest.mark.asyncio
  async def test_handler_returns_string_calls_arun(self, executor, mock_agent):
    trigger = EventTrigger("test")
    trigger.handler = lambda event: "hello world"
    event = TriggerEvent()
    await executor.execute(trigger, event)
    mock_agent.arun.assert_awaited_once_with("hello world")

  @pytest.mark.asyncio
  async def test_handler_returns_dict_calls_arun_with_kwargs(self, executor, mock_agent):
    trigger = EventTrigger("test")
    trigger.handler = lambda event: {"message": "hello", "session_id": "abc"}
    event = TriggerEvent()
    await executor.execute(trigger, event)
    mock_agent.arun.assert_awaited_once_with(message="hello", session_id="abc")

  @pytest.mark.asyncio
  async def test_async_handler(self, executor):
    trigger = EventTrigger("test")

    async def async_handler(event):
      return None

    trigger.handler = async_handler
    event = TriggerEvent()
    result = await executor.execute(trigger, event)
    assert result is None

  @pytest.mark.asyncio
  async def test_handler_exception_returns_none(self, executor):
    trigger = EventTrigger("test")
    trigger.handler = lambda event: 1 / 0
    event = TriggerEvent()
    result = await executor.execute(trigger, event)
    assert result is None

  @pytest.mark.asyncio
  async def test_handler_receives_event(self, executor):
    trigger = EventTrigger("test")
    received = []
    trigger.handler = lambda event: received.append(event) or None
    event = TriggerEvent(body={"key": "value"})
    await executor.execute(trigger, event)
    assert len(received) == 1
    assert received[0].body == {"key": "value"}

  @pytest.mark.asyncio
  async def test_non_standard_return_passed_through(self, executor):
    trigger = EventTrigger("test")
    trigger.handler = lambda event: 42
    event = TriggerEvent()
    result = await executor.execute(trigger, event)
    assert result == 42
