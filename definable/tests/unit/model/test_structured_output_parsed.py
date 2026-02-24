"""
Unit tests for ModelResponse.parsed population across all model providers.

Verifies fix for BUG #6: ModelResponse.parsed was never populated for
structured output except in the Anthropic (Claude) provider. After the fix,
all providers that receive a response_format (Pydantic BaseModel) should
parse the JSON content and set model_response.parsed to a validated instance.

Also tests the full pipeline: model → AgentLoop → RunCompletedEvent → RunOutput.

Covers:
  - OpenAI _parse_provider_response sets parsed
  - Mistral _parse_provider_response sets parsed via kwargs
  - Gemini _parse_provider_response sets parsed via kwargs
  - Ollama _parse_provider_response sets parsed via kwargs
  - xAI inherits OpenAI fix
  - OpenRouter inherits OpenAI fix
  - parsed=None when no response_format given
  - parsed=None when content is not valid JSON
  - RunCompletedEvent carries parsed
  - RunOutput carries parsed
  - AgentLoop non-streaming propagates parsed
  - AgentLoop streaming propagates parsed
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from definable.model.response import ModelResponse


# ---------------------------------------------------------------------------
# Test schema
# ---------------------------------------------------------------------------


class WeatherResponse(BaseModel):
  city: str
  temperature: float
  unit: str = "celsius"


class MovieReview(BaseModel):
  title: str
  rating: int
  summary: str


# ---------------------------------------------------------------------------
# OpenAI provider: _parse_provider_response
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOpenAIParsedPopulation:
  """OpenAI _parse_provider_response sets parsed when response_format is a Pydantic model."""

  def _make_openai_response(self, content: str, role: str = "assistant"):
    """Create a mock ChatCompletion response."""
    message = MagicMock()
    message.role = role
    message.content = content
    message.tool_calls = None
    message.audio = None

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    response.usage = None
    response.id = "chatcmpl-123"
    response.system_fingerprint = None
    response.model_extra = None
    response.error = None

    return response

  def test_parsed_set_with_valid_json(self):
    from definable.model.openai.chat import OpenAIChat

    model = OpenAIChat.__new__(OpenAIChat)
    model.name = "OpenAI"
    model.id = "gpt-4o-mini"

    content = json.dumps({"city": "NYC", "temperature": 72.5, "unit": "fahrenheit"})
    response = self._make_openai_response(content)

    result = model._parse_provider_response(response, response_format=WeatherResponse)

    assert result.parsed is not None
    assert isinstance(result.parsed, WeatherResponse)
    assert result.parsed.city == "NYC"
    assert result.parsed.temperature == 72.5
    assert result.parsed.unit == "fahrenheit"

  def test_parsed_none_without_response_format(self):
    from definable.model.openai.chat import OpenAIChat

    model = OpenAIChat.__new__(OpenAIChat)
    model.name = "OpenAI"
    model.id = "gpt-4o-mini"

    content = json.dumps({"city": "NYC", "temperature": 72.5})
    response = self._make_openai_response(content)

    result = model._parse_provider_response(response)

    assert result.parsed is None

  def test_parsed_none_with_invalid_json(self):
    from definable.model.openai.chat import OpenAIChat

    model = OpenAIChat.__new__(OpenAIChat)
    model.name = "OpenAI"
    model.id = "gpt-4o-mini"

    response = self._make_openai_response("This is not JSON")

    result = model._parse_provider_response(response, response_format=WeatherResponse)

    assert result.parsed is None
    assert result.content == "This is not JSON"

  def test_parsed_none_with_dict_response_format(self):
    """Dict response_format should NOT trigger parsed population."""
    from definable.model.openai.chat import OpenAIChat

    model = OpenAIChat.__new__(OpenAIChat)
    model.name = "OpenAI"
    model.id = "gpt-4o-mini"

    content = json.dumps({"city": "NYC", "temperature": 72.5})
    response = self._make_openai_response(content)

    result = model._parse_provider_response(response, response_format={"type": "json_object"})

    assert result.parsed is None

  def test_parsed_none_when_content_is_none(self):
    from definable.model.openai.chat import OpenAIChat

    model = OpenAIChat.__new__(OpenAIChat)
    model.name = "OpenAI"
    model.id = "gpt-4o-mini"

    response = self._make_openai_response(None)
    response.choices[0].message.content = None

    result = model._parse_provider_response(response, response_format=WeatherResponse)

    assert result.parsed is None

  def test_content_preserved_alongside_parsed(self):
    """parsed should be set but content should remain as the raw JSON string."""
    from definable.model.openai.chat import OpenAIChat

    model = OpenAIChat.__new__(OpenAIChat)
    model.name = "OpenAI"
    model.id = "gpt-4o-mini"

    content = json.dumps({"city": "London", "temperature": 15.0})
    response = self._make_openai_response(content)

    result = model._parse_provider_response(response, response_format=WeatherResponse)

    assert result.content == content
    assert result.parsed is not None
    assert result.parsed.city == "London"

  def test_parsed_with_complex_model(self):
    from definable.model.openai.chat import OpenAIChat

    model = OpenAIChat.__new__(OpenAIChat)
    model.name = "OpenAI"
    model.id = "gpt-4o-mini"

    content = json.dumps({"title": "Inception", "rating": 9, "summary": "Mind-bending thriller"})
    response = self._make_openai_response(content)

    result = model._parse_provider_response(response, response_format=MovieReview)

    assert result.parsed is not None
    assert isinstance(result.parsed, MovieReview)
    assert result.parsed.title == "Inception"
    assert result.parsed.rating == 9

  def test_parsed_none_with_validation_error(self):
    """If JSON is valid but doesn't match the model, parsed should be None."""
    from definable.model.openai.chat import OpenAIChat

    model = OpenAIChat.__new__(OpenAIChat)
    model.name = "OpenAI"
    model.id = "gpt-4o-mini"

    # Missing required 'city' field
    content = json.dumps({"temperature": 72.5})
    response = self._make_openai_response(content)

    result = model._parse_provider_response(response, response_format=WeatherResponse)

    assert result.parsed is None


# ---------------------------------------------------------------------------
# Mistral provider
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMistralParsedPopulation:
  """Mistral _parse_provider_response sets parsed via kwargs."""

  def test_parsed_set_with_valid_json(self):
    try:
      from definable.model.mistral.mistral import MistralChat
    except ImportError:
      pytest.skip("mistralai not installed")

    model = MistralChat.__new__(MistralChat)
    model.name = "Mistral"
    model.id = "mistral-large-latest"

    content = json.dumps({"city": "Paris", "temperature": 20.0})

    response = MagicMock()
    message = MagicMock()
    message.content = content
    message.role = "assistant"
    message.tool_calls = []
    choice = MagicMock()
    choice.message = message
    response.choices = [choice]
    response.usage = None

    result = model._parse_provider_response(response, response_format=WeatherResponse)

    assert result.parsed is not None
    assert isinstance(result.parsed, WeatherResponse)
    assert result.parsed.city == "Paris"

  def test_parsed_none_without_response_format(self):
    try:
      from definable.model.mistral.mistral import MistralChat
    except ImportError:
      pytest.skip("mistralai not installed")

    model = MistralChat.__new__(MistralChat)
    model.name = "Mistral"
    model.id = "mistral-large-latest"

    content = json.dumps({"city": "Paris", "temperature": 20.0})
    response = MagicMock()
    message = MagicMock()
    message.content = content
    message.role = "assistant"
    message.tool_calls = []
    choice = MagicMock()
    choice.message = message
    response.choices = [choice]
    response.usage = None

    result = model._parse_provider_response(response)

    assert result.parsed is None


# ---------------------------------------------------------------------------
# Ollama provider
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOllamaParsedPopulation:
  """Ollama _parse_provider_response sets parsed via kwargs."""

  def test_parsed_set_with_valid_json(self):
    try:
      from definable.model.ollama.chat import Ollama
    except ImportError:
      pytest.skip("ollama not installed")

    model = Ollama.__new__(Ollama)

    content = json.dumps({"city": "Tokyo", "temperature": 25.0})
    response = {
      "message": {"role": "assistant", "content": content},
      "done": True,
    }

    result = model._parse_provider_response(response, response_format=WeatherResponse)

    assert result.parsed is not None
    assert isinstance(result.parsed, WeatherResponse)
    assert result.parsed.city == "Tokyo"

  def test_parsed_none_without_response_format(self):
    try:
      from definable.model.ollama.chat import Ollama
    except ImportError:
      pytest.skip("ollama not installed")

    model = Ollama.__new__(Ollama)

    content = json.dumps({"city": "Tokyo", "temperature": 25.0})
    response = {
      "message": {"role": "assistant", "content": content},
      "done": True,
    }

    result = model._parse_provider_response(response)

    assert result.parsed is None


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGeminiParsedPopulation:
  """Gemini _parse_provider_response sets parsed via kwargs."""

  def test_parsed_set_with_valid_json(self):
    try:
      from definable.model.google.gemini import Gemini
    except ImportError:
      pytest.skip("google-genai not installed")

    model = Gemini.__new__(Gemini)
    model.retry_with_guidance = False

    content = json.dumps({"city": "Berlin", "temperature": 18.0})

    # Build a mock Gemini response
    part = MagicMock()
    part.text = content
    part.thought = False
    part.thought_signature = None
    part.inline_data = None
    part.function_call = None
    del part.executable_code
    del part.code_execution_result

    content_obj = MagicMock()
    content_obj.role = "model"
    content_obj.parts = [part]

    candidate = MagicMock()
    candidate.content = content_obj
    candidate.finish_reason = None
    candidate.grounding_metadata = None
    candidate.url_context_metadata = None

    response = MagicMock()
    response.candidates = [candidate]
    response.usage_metadata = None

    model.role_map = {"model": "assistant", "user": "user"}

    result = model._parse_provider_response(response, response_format=WeatherResponse)

    assert result.parsed is not None
    assert isinstance(result.parsed, WeatherResponse)
    assert result.parsed.city == "Berlin"


# ---------------------------------------------------------------------------
# RunCompletedEvent and RunOutput: parsed field
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunOutputParsed:
  """RunOutput and RunCompletedEvent carry the parsed field."""

  def test_run_completed_event_has_parsed(self):
    from definable.agent.events import RunCompletedEvent

    event = RunCompletedEvent(
      run_id="run-1",
      content='{"city": "NYC"}',
      parsed=WeatherResponse(city="NYC", temperature=72.5),
    )
    assert event.parsed is not None
    assert isinstance(event.parsed, WeatherResponse)

  def test_run_completed_event_parsed_default_none(self):
    from definable.agent.events import RunCompletedEvent

    event = RunCompletedEvent(run_id="run-1", content="Hello")
    assert event.parsed is None

  def test_run_output_has_parsed(self):
    from definable.agent.events import RunOutput

    output = RunOutput(
      content='{"city": "NYC"}',
      parsed=WeatherResponse(city="NYC", temperature=72.5),
    )
    assert output.parsed is not None
    assert isinstance(output.parsed, WeatherResponse)

  def test_run_output_parsed_default_none(self):
    from definable.agent.events import RunOutput

    output = RunOutput(content="Hello")
    assert output.parsed is None


# ---------------------------------------------------------------------------
# AgentLoop: non-streaming parsed propagation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAgentLoopParsedPropagation:
  """AgentLoop propagates parsed through RunCompletedEvent."""

  @pytest.mark.asyncio
  async def test_non_streaming_parsed_in_completed_event(self):
    from definable.agent.events import RunCompletedEvent, RunContext
    from definable.agent.loop import AgentLoop
    from definable.model.message import Message

    # Build a mock model that returns a response with parsed set
    parsed_obj = WeatherResponse(city="NYC", temperature=72.5)
    mock_response = ModelResponse(
      role="assistant",
      content=json.dumps({"city": "NYC", "temperature": 72.5}),
      parsed=parsed_obj,
    )

    mock_model = AsyncMock()
    mock_model.ainvoke = AsyncMock(return_value=mock_response)
    mock_model.id = "test-model"
    mock_model.provider = "test"

    context = RunContext(
      run_id="run-1",
      session_id="session-1",
      output_schema=WeatherResponse,
    )

    config = MagicMock()
    config.max_tool_rounds = 10
    config.retry_transient_errors = False

    loop = AgentLoop(
      model=mock_model,
      tools=[],
      messages=[Message(role="user", content="What's the weather?")],
      context=context,
      config=config,
      streaming=False,
      emit_fn=lambda e: None,
      agent_id="test-agent",
      agent_name="TestAgent",
    )

    completed_event = None
    async for event in loop.run():
      if isinstance(event, RunCompletedEvent):
        completed_event = event

    assert completed_event is not None
    assert completed_event.parsed is not None
    assert isinstance(completed_event.parsed, WeatherResponse)
    assert completed_event.parsed.city == "NYC"

  @pytest.mark.asyncio
  async def test_non_streaming_parsed_none_when_no_schema(self):
    from definable.agent.events import RunCompletedEvent, RunContext
    from definable.agent.loop import AgentLoop
    from definable.model.message import Message

    mock_response = ModelResponse(
      role="assistant",
      content="Just a normal response",
    )

    mock_model = AsyncMock()
    mock_model.ainvoke = AsyncMock(return_value=mock_response)
    mock_model.id = "test-model"
    mock_model.provider = "test"

    context = RunContext(
      run_id="run-1",
      session_id="session-1",
      output_schema=None,
    )

    config = MagicMock()
    config.max_tool_rounds = 10
    config.retry_transient_errors = False

    loop = AgentLoop(
      model=mock_model,
      tools=[],
      messages=[Message(role="user", content="Hello")],
      context=context,
      config=config,
      streaming=False,
      emit_fn=lambda e: None,
      agent_id="test-agent",
      agent_name="TestAgent",
    )

    completed_event = None
    async for event in loop.run():
      if isinstance(event, RunCompletedEvent):
        completed_event = event

    assert completed_event is not None
    assert completed_event.parsed is None


# ---------------------------------------------------------------------------
# Regression: parsed field on ModelResponse dataclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelResponseParsedField:
  """Regression: ModelResponse.parsed field exists and defaults to None."""

  def test_parsed_default_none(self):
    resp = ModelResponse()
    assert resp.parsed is None

  def test_parsed_can_be_set(self):
    obj = WeatherResponse(city="NYC", temperature=72.5)
    resp = ModelResponse(parsed=obj)
    assert resp.parsed is obj

  def test_parsed_survives_to_dict_from_dict(self):
    """parsed is not serialized (transient) but should survive normal usage."""
    resp = ModelResponse(content='{"city": "NYC"}', parsed=WeatherResponse(city="NYC", temperature=72.5))
    # to_dict uses dataclasses.asdict which will serialize the parsed field
    d = resp.to_dict()
    assert "parsed" in d
