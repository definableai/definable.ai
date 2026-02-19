"""
Unit tests for MockModel and agent test utilities.

Tests MockModel response sequencing, call tracking, assertion helpers,
create_test_agent convenience function, and AgentTestCase base class.
No API calls.
"""

import pytest

from definable.agent.testing import AgentTestCase, MockModel, create_test_agent


@pytest.mark.unit
class TestMockModelCreation:
  def test_default_responses(self):
    """MockModel() defaults to ['Mock response']."""
    model = MockModel()
    assert model.responses == ["Mock response"]

  def test_custom_responses(self):
    """MockModel(responses=[...]) stores the list."""
    model = MockModel(responses=["Hello", "World"])
    assert model.responses == ["Hello", "World"]

  def test_model_identity(self):
    """MockModel has id='mock-model' and provider='mock'."""
    model = MockModel()
    assert model.id == "mock-model"
    assert model.provider == "mock"

  def test_initial_call_count_is_zero(self):
    """Fresh MockModel has call_count=0."""
    model = MockModel()
    assert model.call_count == 0

  def test_initial_call_history_is_empty(self):
    """Fresh MockModel has empty call_history."""
    model = MockModel()
    assert model.call_history == []

  def test_supports_native_structured_outputs(self):
    """MockModel reports native structured output support."""
    model = MockModel()
    assert model.supports_native_structured_outputs is True


@pytest.mark.unit
class TestMockModelInvoke:
  @pytest.mark.asyncio
  async def test_ainvoke_returns_first_response(self):
    """ainvoke() returns the first response on first call."""
    model = MockModel(responses=["Hello!", "Goodbye!"])
    response = await model.ainvoke(messages=[])
    assert response.content == "Hello!"

  @pytest.mark.asyncio
  async def test_ainvoke_returns_responses_in_order(self):
    """Successive ainvoke() calls return responses sequentially."""
    model = MockModel(responses=["First", "Second", "Third"])

    r1 = await model.ainvoke(messages=[])
    assert r1.content == "First"

    r2 = await model.ainvoke(messages=[])
    assert r2.content == "Second"

    r3 = await model.ainvoke(messages=[])
    assert r3.content == "Third"

  @pytest.mark.asyncio
  async def test_ainvoke_clamps_to_last_response(self):
    """When calls exceed response count, last response is repeated."""
    model = MockModel(responses=["Only"])

    r1 = await model.ainvoke(messages=[])
    assert r1.content == "Only"

    r2 = await model.ainvoke(messages=[])
    assert r2.content == "Only"

  @pytest.mark.asyncio
  async def test_ainvoke_increments_call_count(self):
    """Each ainvoke() call increments call_count."""
    model = MockModel()
    await model.ainvoke(messages=[])
    assert model.call_count == 1
    await model.ainvoke(messages=[])
    assert model.call_count == 2

  @pytest.mark.asyncio
  async def test_ainvoke_records_call_history(self):
    """ainvoke() records arguments in call_history."""
    model = MockModel()
    messages = [{"role": "user", "content": "hi"}]
    await model.ainvoke(messages=messages, tools=None)

    assert len(model.call_history) == 1
    assert model.call_history[0]["messages"] is messages

  @pytest.mark.asyncio
  async def test_ainvoke_response_has_empty_tool_executions(self):
    """Default ainvoke() response has empty tool_executions."""
    model = MockModel()
    response = await model.ainvoke(messages=[])
    assert response.tool_executions == []

  def test_invoke_returns_response(self):
    """Sync invoke() wraps ainvoke() and returns a response."""
    model = MockModel(responses=["Sync response"])
    response = model.invoke(messages=[])
    assert response.content == "Sync response"
    assert model.call_count == 1


@pytest.mark.unit
class TestMockModelSideEffect:
  @pytest.mark.asyncio
  async def test_side_effect_called(self):
    """When side_effect is provided, it is called instead of canned responses."""
    from unittest.mock import MagicMock as StdMock

    from definable.model.metrics import Metrics

    def custom_fn(messages, tools, **kwargs):
      resp = StdMock()
      resp.content = f"Got {len(messages or [])} msgs"
      resp.tool_executions = []
      resp.tool_calls = []
      resp.response_usage = Metrics()
      resp.reasoning_content = None
      resp.citations = None
      resp.images = None
      resp.videos = None
      resp.audios = None
      return resp

    model = MockModel(side_effect=custom_fn)
    response = await model.ainvoke(messages=[{"role": "user"}])
    assert response.content == "Got 1 msgs"


@pytest.mark.unit
class TestMockModelStructuredResponses:
  @pytest.mark.asyncio
  async def test_structured_response_returned_for_output_schema(self):
    """When output_schema is set and structured_responses exist, use them."""
    model = MockModel(
      responses=["normal"],
      structured_responses=['{"name": "test"}'],
    )
    response = await model.ainvoke(messages=[], output_schema=object)
    assert response.content == '{"name": "test"}'

  @pytest.mark.asyncio
  async def test_normal_response_when_no_output_schema(self):
    """Without output_schema, canned responses are used."""
    model = MockModel(
      responses=["normal"],
      structured_responses=['{"name": "test"}'],
    )
    response = await model.ainvoke(messages=[])
    assert response.content == "normal"


@pytest.mark.unit
class TestMockModelReset:
  @pytest.mark.asyncio
  async def test_reset_clears_count_and_history(self):
    """reset() zeroes call_count and clears call_history."""
    model = MockModel()
    await model.ainvoke(messages=[])
    await model.ainvoke(messages=[])
    assert model.call_count == 2

    model.reset()
    assert model.call_count == 0
    assert model.call_history == []


@pytest.mark.unit
class TestMockModelAssertions:
  @pytest.mark.asyncio
  async def test_assert_called_passes_after_call(self):
    """assert_called() passes when model was called."""
    model = MockModel()
    await model.ainvoke(messages=[])
    model.assert_called()  # should not raise

  def test_assert_called_fails_when_not_called(self):
    """assert_called() raises when model was never called."""
    model = MockModel()
    with pytest.raises(AssertionError, match="not called"):
      model.assert_called()

  @pytest.mark.asyncio
  async def test_assert_called_times_passes(self):
    """assert_called_times(n) passes when call_count matches."""
    model = MockModel()
    await model.ainvoke(messages=[])
    await model.ainvoke(messages=[])
    model.assert_called_times(2)  # should not raise

  @pytest.mark.asyncio
  async def test_assert_called_times_fails_on_mismatch(self):
    """assert_called_times(n) raises on count mismatch."""
    model = MockModel()
    await model.ainvoke(messages=[])
    with pytest.raises(AssertionError, match="1 times, expected 3"):
      model.assert_called_times(3)


@pytest.mark.unit
class TestMockModelStreaming:
  @pytest.mark.asyncio
  async def test_ainvoke_stream_yields_characters(self):
    """ainvoke_stream() yields one chunk per character."""
    model = MockModel(responses=["Hi"])
    chunks = []
    async for chunk in model.ainvoke_stream(messages=[]):
      chunks.append(chunk.content)
    assert chunks == ["H", "i"]

  @pytest.mark.asyncio
  async def test_ainvoke_stream_increments_call_count(self):
    """ainvoke_stream() increments call_count."""
    model = MockModel(responses=["A"])
    async for _ in model.ainvoke_stream(messages=[]):
      pass
    assert model.call_count == 1


@pytest.mark.unit
class TestCreateTestAgent:
  def test_returns_agent(self):
    """create_test_agent() returns an Agent instance."""
    from definable.agent.agent import Agent

    agent = create_test_agent()
    assert isinstance(agent, Agent)

  def test_uses_mock_model(self):
    """create_test_agent() uses MockModel internally."""
    agent = create_test_agent(responses=["Hello"])
    assert isinstance(agent.model, MockModel)
    assert agent.model.responses == ["Hello"]

  def test_accepts_tools(self):
    """create_test_agent(tools=[...]) passes tools to Agent."""
    from definable.tool.decorator import tool

    @tool
    def my_tool(x: str) -> str:
      """A test tool."""
      return x

    agent = create_test_agent(tools=[my_tool])
    assert len(agent.tools) == 1

  def test_accepts_extra_kwargs(self):
    """create_test_agent(**kwargs) passes additional args to Agent."""
    agent = create_test_agent(instructions="Be concise.")
    assert agent.instructions == "Be concise."


@pytest.mark.unit
class TestAgentTestCase:
  def test_class_exists(self):
    """AgentTestCase base class can be instantiated."""
    tc = AgentTestCase()
    assert tc is not None

  def test_create_agent_method(self):
    """AgentTestCase.create_agent() returns an Agent."""
    from definable.agent.agent import Agent

    tc = AgentTestCase()
    agent = tc.create_agent()
    assert isinstance(agent, Agent)

  def test_create_agent_with_custom_model(self):
    """AgentTestCase.create_agent(model=...) accepts a MockModel."""
    tc = AgentTestCase()
    model = MockModel(responses=["Custom"])
    agent = tc.create_agent(model=model)
    assert agent.model is model

  def test_create_agent_with_instructions(self):
    """AgentTestCase.create_agent(instructions=...) passes instructions."""
    tc = AgentTestCase()
    agent = tc.create_agent(instructions="Test instructions")
    assert agent.instructions == "Test instructions"

  def test_assert_has_content_passes(self):
    """assert_has_content passes when output has content."""
    tc = AgentTestCase()
    output = type("FakeOutput", (), {"content": "hello"})()
    tc.assert_has_content(output)  # should not raise

  def test_assert_has_content_fails_on_empty(self):
    """assert_has_content raises when output has no content."""
    tc = AgentTestCase()
    output = type("FakeOutput", (), {"content": ""})()
    with pytest.raises(AssertionError):
      tc.assert_has_content(output)

  def test_assert_content_contains_passes(self):
    """assert_content_contains passes when substring is present."""
    tc = AgentTestCase()
    output = type("FakeOutput", (), {"content": "Hello, World!"})()
    tc.assert_content_contains(output, "World")  # should not raise

  def test_assert_content_contains_fails_on_missing(self):
    """assert_content_contains raises when substring is missing."""
    tc = AgentTestCase()
    output = type("FakeOutput", (), {"content": "Hello"})()
    with pytest.raises(AssertionError):
      tc.assert_content_contains(output, "Goodbye")

  def test_assert_no_errors_passes_on_completed(self):
    """assert_no_errors passes when status is completed."""
    from definable.agent.events import RunStatus

    tc = AgentTestCase()
    output = type("FakeOutput", (), {"status": RunStatus.completed})()
    tc.assert_no_errors(output)  # should not raise

  def test_assert_message_count(self):
    """assert_message_count passes when count matches."""
    tc = AgentTestCase()
    output = type("FakeOutput", (), {"messages": ["a", "b", "c"]})()
    tc.assert_message_count(output, 3)  # should not raise

  def test_assert_message_count_fails_on_mismatch(self):
    """assert_message_count raises on count mismatch."""
    tc = AgentTestCase()
    output = type("FakeOutput", (), {"messages": ["a"]})()
    with pytest.raises(AssertionError, match="Expected 5"):
      tc.assert_message_count(output, 5)
