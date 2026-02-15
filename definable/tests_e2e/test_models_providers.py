"""E2E tests — Model Provider Testing.

Scenario: "I want to use different LLM providers."

Each provider class requires its own API key.
"""

import asyncio
import json

import pytest
from pydantic import BaseModel

from definable.models.message import Message


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.openai
class TestOpenAI:
  """OpenAI model end-to-end tests."""

  def test_invoke_basic(self, openai_model, simple_messages, assistant_message):
    """Sync invoke returns non-empty content."""
    response = openai_model.invoke(
      messages=simple_messages,
      assistant_message=assistant_message(),
    )
    assert response.content is not None
    assert len(response.content) > 0

  @pytest.mark.asyncio
  async def test_ainvoke_basic(self, openai_model, simple_messages, assistant_message):
    """Async invoke returns content."""
    response = await openai_model.ainvoke(
      messages=simple_messages,
      assistant_message=assistant_message(),
    )
    assert response.content is not None
    assert len(response.content) > 0

  def test_invoke_with_system_message(self, openai_model, assistant_message):
    """Model respects system message instructions."""
    messages = [
      Message(role="system", content="You only respond with the word 'PONG'."),
      Message(role="user", content="PING"),
    ]
    response = openai_model.invoke(messages=messages, assistant_message=assistant_message())
    assert response.content is not None
    assert "pong" in response.content.lower()

  def test_invoke_stream(self, openai_model, simple_messages, assistant_message):
    """Streaming invoke yields at least one chunk."""
    chunks = list(
      openai_model.invoke_stream(
        messages=simple_messages,
        assistant_message=assistant_message(),
      )
    )
    assert len(chunks) > 0

  @pytest.mark.asyncio
  async def test_ainvoke_stream(self, openai_model, simple_messages, assistant_message):
    """Async streaming invoke yields chunks."""
    chunks = []
    async for chunk in openai_model.ainvoke_stream(
      messages=simple_messages,
      assistant_message=assistant_message(),
    ):
      chunks.append(chunk)
    assert len(chunks) > 0

  def test_structured_output(self, openai_model, assistant_message):
    """Structured output with Pydantic model."""

    class MathResult(BaseModel):
      answer: int

    messages = [Message(role="user", content="What is 2+2? Return the answer.")]
    response = openai_model.invoke(
      messages=messages,
      assistant_message=assistant_message(),
      response_format=MathResult,
    )
    if response.parsed is not None:
      assert isinstance(response.parsed, MathResult)
      assert response.parsed.answer == 4
    else:
      parsed = json.loads(response.content)
      assert parsed["answer"] == 4

  def test_structured_output_nested(self, openai_model, assistant_message):
    """Structured output with nested Pydantic model."""

    class Address(BaseModel):
      city: str
      country: str

    class Person(BaseModel):
      name: str
      age: int
      address: Address

    messages = [
      Message(
        role="user",
        content="Create a person: John, age 30, New York, USA.",
      ),
    ]
    response = openai_model.invoke(
      messages=messages,
      assistant_message=assistant_message(),
      response_format=Person,
    )
    if response.parsed is not None:
      assert isinstance(response.parsed, Person)
      assert response.parsed.name == "John"
    else:
      data = json.loads(response.content)
      assert data["name"] == "John"

  def test_metrics_populated(self, openai_model, assistant_message):
    """Response includes usage metrics with token counts."""
    messages = [Message(role="user", content="Say hello.")]
    response = openai_model.invoke(messages=messages, assistant_message=assistant_message())

    assert response.response_usage is not None
    assert response.response_usage.input_tokens > 0
    assert response.response_usage.output_tokens > 0
    assert response.response_usage.total_tokens > 0

  def test_multi_turn(self, openai_model, assistant_message):
    """Model handles multi-turn conversation."""
    messages = [Message(role="user", content="My name is Alice.")]
    response1 = openai_model.invoke(messages=messages, assistant_message=assistant_message())
    assert response1.content is not None

    messages.append(Message(role="assistant", content=response1.content))
    messages.append(Message(role="user", content="What is my name?"))
    response2 = openai_model.invoke(messages=messages, assistant_message=assistant_message())
    assert response2.content is not None
    assert "alice" in response2.content.lower()

  @pytest.mark.asyncio
  async def test_concurrent_calls(self, openai_model, assistant_message):
    """Multiple async calls run concurrently."""
    messages = [Message(role="user", content="Say 'test' and nothing else.")]

    async def make_call():
      return await openai_model.ainvoke(messages=messages, assistant_message=assistant_message())

    results = await asyncio.gather(make_call(), make_call(), make_call())
    assert len(results) == 3
    for response in results:
      assert response.content is not None


# ---------------------------------------------------------------------------
# DeepSeek
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.deepseek
class TestDeepSeek:
  """DeepSeek model tests."""

  def test_invoke_basic(self, deepseek_model, simple_messages, assistant_message):
    """Sync invoke returns content."""
    response = deepseek_model.invoke(messages=simple_messages, assistant_message=assistant_message())
    assert response.content is not None
    assert len(response.content) > 0

  @pytest.mark.asyncio
  async def test_ainvoke_stream(self, deepseek_model, simple_messages, assistant_message):
    """Async streaming invoke yields chunks."""
    chunks = []
    async for chunk in deepseek_model.ainvoke_stream(
      messages=simple_messages,
      assistant_message=assistant_message(),
    ):
      chunks.append(chunk)
    assert len(chunks) > 0

  def test_multi_turn(self, deepseek_model, assistant_message):
    """Multi-turn conversation works."""
    messages = [Message(role="user", content="My name is Bob.")]
    r1 = deepseek_model.invoke(messages=messages, assistant_message=assistant_message())
    assert r1.content is not None

    messages.append(Message(role="assistant", content=r1.content))
    messages.append(Message(role="user", content="What is my name?"))
    r2 = deepseek_model.invoke(messages=messages, assistant_message=assistant_message())
    assert r2.content is not None
    assert "bob" in r2.content.lower()


# ---------------------------------------------------------------------------
# xAI
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.xai
class TestXAI:
  """xAI model tests."""

  def test_invoke_basic(self, xai_model, simple_messages, assistant_message):
    """Sync invoke returns content."""
    response = xai_model.invoke(messages=simple_messages, assistant_message=assistant_message())
    assert response.content is not None
    assert len(response.content) > 0

  @pytest.mark.asyncio
  async def test_ainvoke_stream(self, xai_model, simple_messages, assistant_message):
    """Async streaming invoke yields chunks."""
    chunks = []
    async for chunk in xai_model.ainvoke_stream(
      messages=simple_messages,
      assistant_message=assistant_message(),
    ):
      chunks.append(chunk)
    assert len(chunks) > 0


# ---------------------------------------------------------------------------
# Moonshot
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.moonshot
class TestMoonshot:
  """Moonshot model tests."""

  def test_invoke_basic(self, moonshot_model, simple_messages, assistant_message):
    """Sync invoke returns content."""
    response = moonshot_model.invoke(messages=simple_messages, assistant_message=assistant_message())
    assert response.content is not None
    assert len(response.content) > 0

  @pytest.mark.asyncio
  async def test_ainvoke_stream(self, moonshot_model, simple_messages, assistant_message):
    """Async streaming invoke yields chunks."""
    chunks = []
    async for chunk in moonshot_model.ainvoke_stream(
      messages=simple_messages,
      assistant_message=assistant_message(),
    ):
      chunks.append(chunk)
    assert len(chunks) > 0
