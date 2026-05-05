"""
Contract tests: Every Model implementation must satisfy these.

The Model ABC defines the contract:
  - invoke(messages, assistant_message) -> ModelResponse (sync)
  - ainvoke(messages, assistant_message) -> ModelResponse (async)
  - ainvoke_stream(messages, assistant_message) -> AsyncIterator of chunks
  - ModelResponse has: content (str|None), tool_executions (list)

IMPORTANT: Real models require `assistant_message` as a required second positional arg.

All provider tests require their respective API key -- will FAIL if not set.
To add a new Model: inherit ModelContractTests and provide a `model` fixture.
"""

import pytest

from definable.model.message import Message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_messages(text: str) -> list:
  return [Message(role="user", content=text)]


def make_assistant_message() -> Message:
  """Empty assistant message placeholder required by real model implementations."""
  return Message(role="assistant", content=None)


def check_model_response(response):
  """Assert ModelResponse has the expected structure."""
  assert response is not None
  has_content = hasattr(response, "content")
  has_tools = hasattr(response, "tool_executions")
  assert has_content or has_tools


# ---------------------------------------------------------------------------
# Contract definition
# ---------------------------------------------------------------------------


class ModelContractTests:
  """
  Abstract contract test suite for all Model implementations.

  Every concrete Model must pass ALL tests in this class.
  """

  @pytest.fixture
  def model(self):
    raise NotImplementedError("Subclass must provide a model fixture")

  @pytest.fixture
  def messages(self) -> list:
    return make_messages("What is 1+1? Reply with just the number.")

  @pytest.fixture
  def assistant_msg(self) -> Message:
    return make_assistant_message()

  # --- Contract: invoke ---

  @pytest.mark.contract
  def test_invoke_returns_response(self, model, messages, assistant_msg):
    response = model.invoke(messages=messages, assistant_message=assistant_msg)
    check_model_response(response)

  @pytest.mark.contract
  def test_invoke_response_has_content_attribute(self, model, messages, assistant_msg):
    response = model.invoke(messages=messages, assistant_message=assistant_msg)
    assert hasattr(response, "content")

  # --- Contract: ainvoke ---

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_ainvoke_returns_response(self, model, messages, assistant_msg):
    response = await model.ainvoke(messages=messages, assistant_message=assistant_msg)
    check_model_response(response)

  @pytest.mark.contract
  def test_ainvoke_response_matches_invoke_shape(self, model, messages, assistant_msg):
    """sync invoke and async ainvoke must return the same attribute shape."""
    import asyncio

    sync_response = model.invoke(messages=messages, assistant_message=make_assistant_message())
    async_response = asyncio.run(model.ainvoke(messages=messages, assistant_message=make_assistant_message()))
    assert hasattr(async_response, "content") == hasattr(sync_response, "content")

  # --- Contract: streaming ---

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_ainvoke_stream_yields_chunks(self, model, messages, assistant_msg):
    chunks = []
    async for chunk in model.ainvoke_stream(messages=messages, assistant_message=assistant_msg):
      chunks.append(chunk)
    assert len(chunks) >= 1

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_ainvoke_stream_chunks_have_some_attribute(self, model, messages, assistant_msg):
    """Chunks must have at least one recognizable attribute."""
    async for chunk in model.ainvoke_stream(messages=messages, assistant_message=assistant_msg):
      has_something = hasattr(chunk, "content") or hasattr(chunk, "tool_calls") or hasattr(chunk, "response_usage")
      assert has_something
      break

  # --- Contract: model identity ---

  @pytest.mark.contract
  def test_model_has_id_attribute(self, model):
    assert hasattr(model, "id")
    assert model.id is not None

  @pytest.mark.contract
  def test_model_has_provider_attribute(self, model):
    assert hasattr(model, "provider")

  @pytest.mark.contract
  def test_model_has_structured_output_capability_flag(self, model):
    assert hasattr(model, "supports_native_structured_outputs")
    assert isinstance(model.supports_native_structured_outputs, bool)


# ---------------------------------------------------------------------------
# Real providers -- require API keys (FAIL if not set)
# ---------------------------------------------------------------------------


@pytest.mark.contract
@pytest.mark.integration
@pytest.mark.openai
class TestOpenAIChatContract(ModelContractTests):
  """OpenAIChat satisfies the Model contract."""

  @pytest.fixture
  def model(self, openai_model):
    return openai_model

  def test_openai_provider_is_openai(self, model):
    assert model.provider is not None
    assert "openai" in str(model.provider).lower()


@pytest.mark.contract
@pytest.mark.integration
@pytest.mark.deepseek
class TestDeepSeekChatContract(ModelContractTests):
  """DeepSeekChat satisfies the Model contract."""

  @pytest.fixture
  def model(self, deepseek_model):
    return deepseek_model


@pytest.mark.contract
@pytest.mark.integration
@pytest.mark.moonshot
class TestMoonshotChatContract(ModelContractTests):
  """MoonshotChat satisfies the Model contract."""

  @pytest.fixture
  def model(self, moonshot_model):
    return moonshot_model


@pytest.mark.contract
@pytest.mark.integration
@pytest.mark.xai
class TestXAIContract(ModelContractTests):
  """xAI satisfies the Model contract."""

  @pytest.fixture
  def model(self, xai_model):
    return xai_model
