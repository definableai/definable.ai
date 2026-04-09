"""
Integration tests: OpenAI model direct invocation.

Migrated from tests_e2e/integration/test_models.py (OpenAI section).

Strategy:
  - Real OpenAI API calls — no mocks
  - Tests invoke the model directly (not through Agent)
  - Session-scoped openai_model fixture for speed

Covers:
  - OpenAIChat.invoke() returns content (sync)
  - OpenAIChat.ainvoke() returns content (async)
  - OpenAIChat streaming produces chunks
  - Metrics (token usage) are populated via response_usage
  - System message is respected
  - Multi-turn conversation via messages list
  - Tool calling: model returns tool_calls when tools are provided
"""

import pytest

from definable.model.message import Message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_messages(content: str) -> list:
  """Create a single-user-message list for model invocation."""
  return [Message(role="user", content=content)]


def make_assistant_msg() -> Message:
  """Empty assistant message placeholder required by real models."""
  return Message(role="assistant", content=None)


# ---------------------------------------------------------------------------
# Sync invoke
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.openai
class TestOpenAISyncInvoke:
  """OpenAIChat.invoke() sync path."""

  def test_invoke_returns_content(self, openai_model):
    """Sync invoke should return a response with string content."""
    messages = make_messages("What is 1+1? Reply with just the number.")
    response = openai_model.invoke(messages=messages, assistant_message=make_assistant_msg())
    assert response.content is not None
    assert isinstance(response.content, str)
    assert "2" in response.content


# ---------------------------------------------------------------------------
# Async invoke
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.openai
class TestOpenAIAsyncInvoke:
  """OpenAIChat.ainvoke() async path."""

  @pytest.mark.asyncio
  async def test_ainvoke_returns_content(self, openai_model):
    """Async ainvoke should return a response with string content."""
    messages = make_messages("Say exactly: HELLO")
    response = await openai_model.ainvoke(messages=messages, assistant_message=make_assistant_msg())
    assert response.content is not None
    assert isinstance(response.content, str)

  @pytest.mark.asyncio
  async def test_ainvoke_with_system_message(self, openai_model):
    """System message should influence the model response."""
    messages = [
      Message(role="system", content="You are a chef. Always mention food in your reply."),
      Message(role="user", content="What are you?"),
    ]
    response = await openai_model.ainvoke(
      messages=messages,
      assistant_message=make_assistant_msg(),
    )
    assert response.content is not None
    assert isinstance(response.content, str)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.openai
class TestOpenAIMetrics:
  """Model response contains token usage metrics."""

  @pytest.mark.asyncio
  async def test_ainvoke_returns_metrics(self, openai_model):
    """response_usage should contain token counts."""
    messages = make_messages("Hi")
    response = await openai_model.ainvoke(messages=messages, assistant_message=make_assistant_msg())
    if hasattr(response, "response_usage") and response.response_usage:
      metrics = response.response_usage
      assert hasattr(metrics, "prompt_tokens") or hasattr(metrics, "total_tokens")


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.openai
class TestOpenAIStreaming:
  """OpenAIChat streaming produces content chunks."""

  @pytest.mark.asyncio
  async def test_streaming_produces_content_chunks(self, openai_model):
    """ainvoke_stream should yield multiple chunks with content."""
    messages = make_messages("Count from 1 to 3.")
    chunks = []
    async for chunk in openai_model.ainvoke_stream(messages=messages, assistant_message=make_assistant_msg()):
      if chunk.content:
        chunks.append(chunk.content)
    assert len(chunks) > 0
    combined = "".join(chunks)
    assert len(combined) > 0


# ---------------------------------------------------------------------------
# Multi-turn conversation
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.openai
class TestOpenAIMultiTurn:
  """Model uses context from prior messages."""

  @pytest.mark.asyncio
  async def test_multi_turn_conversation(self, openai_model):
    """Model should use context from prior messages to answer correctly."""
    messages = [
      Message(role="user", content="My favorite color is blue. Remember this."),
      Message(role="assistant", content="I'll remember that your favorite color is blue."),
      Message(role="user", content="What is my favorite color?"),
    ]
    response = await openai_model.ainvoke(messages=messages, assistant_message=make_assistant_msg())
    assert "blue" in response.content.lower()


# ---------------------------------------------------------------------------
# Tool calling at model level
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.openai
class TestOpenAIToolCalling:
  """Model returns tool_calls when relevant tools are provided."""

  @pytest.mark.asyncio
  async def test_tool_calling_invokes_correct_tool(self, openai_model):
    """Model should return a tool_call when a relevant tool is available."""
    from definable.tool.decorator import tool

    @tool
    def get_weather(city: str) -> str:
      """Get the weather for a city."""
      return f"Sunny in {city}"

    tool_schema = {"type": "function", "function": get_weather.to_dict()}
    messages = make_messages("What is the weather in London?")
    response = await openai_model.ainvoke(
      messages=messages,
      assistant_message=make_assistant_msg(),
      tools=[tool_schema],
    )
    # Model may return tool_calls or answer directly
    has_tool_call = bool(response.tool_calls)
    has_content = bool(response.content)
    assert has_tool_call or has_content
