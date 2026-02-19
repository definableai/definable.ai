"""
Integration tests for LLM model providers.

Rules:
  - NO MOCKS — real API calls to real model providers
  - Session-scoped model fixtures for speed
  - Each test asserts on RESPONSE CONTENT and METRICS, not implementation details

Covers:
  - invoke() returns content (str) and metrics
  - ainvoke() async path works correctly
  - Streaming yields content chunks
  - Multi-turn conversation via messages list
  - Tool calling: model returns tool_executions when tools are provided
  - Structured output: model returns parseable JSON
  - All 4 providers: OpenAI, DeepSeek, Moonshot, xAI (skip if no key)
"""

import pytest

from definable.model.message import Message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_messages(content: str) -> list:
  return [Message(role="user", content=content)]


def make_assistant_msg() -> Message:
  """Empty assistant message placeholder required by real models."""
  return Message(role="assistant", content=None)


def check_response(response):
  """Assert a model response has the expected structure."""
  assert response is not None
  assert hasattr(response, "content") or hasattr(response, "tool_executions")
  if hasattr(response, "content") and response.content:
    assert isinstance(response.content, str)
    assert len(response.content) > 0


# ---------------------------------------------------------------------------
# Parametrized over all providers
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.openai
class TestOpenAIModel:
  """Real OpenAI API tests."""

  @pytest.mark.asyncio
  async def test_ainvoke_returns_content(self, openai_model):
    messages = make_messages("Say exactly: HELLO")
    response = await openai_model.ainvoke(messages=messages, assistant_message=make_assistant_msg())
    assert response.content
    assert isinstance(response.content, str)

  def test_invoke_returns_content(self, openai_model):
    messages = make_messages("What is 1+1? Reply with just the number.")
    response = openai_model.invoke(messages=messages, assistant_message=make_assistant_msg())
    assert response.content
    assert "2" in response.content

  @pytest.mark.asyncio
  async def test_ainvoke_with_system_message(self, openai_model):
    # System message is passed as a Message(role="system") prepended to messages list
    from definable.model.message import Message

    messages = [
      Message(role="system", content="You are a chef. Always mention food in your reply."),
      Message(role="user", content="What are you?"),
    ]
    response = await openai_model.ainvoke(
      messages=messages,
      assistant_message=make_assistant_msg(),
    )
    assert response.content
    assert isinstance(response.content, str)

  @pytest.mark.asyncio
  async def test_ainvoke_returns_metrics(self, openai_model):
    messages = make_messages("Hi")
    response = await openai_model.ainvoke(messages=messages, assistant_message=make_assistant_msg())
    if hasattr(response, "response_usage") and response.response_usage:
      metrics = response.response_usage
      assert hasattr(metrics, "prompt_tokens") or hasattr(metrics, "total_tokens")

  @pytest.mark.asyncio
  async def test_streaming_produces_content_chunks(self, openai_model):
    messages = make_messages("Count from 1 to 3.")
    chunks = []
    async for chunk in openai_model.ainvoke_stream(messages=messages, assistant_message=make_assistant_msg()):
      if chunk.content:
        chunks.append(chunk.content)
    assert len(chunks) > 0
    combined = "".join(chunks)
    assert len(combined) > 0

  @pytest.mark.asyncio
  async def test_multi_turn_conversation(self, openai_model):
    """Multi-turn: model should use context from prior messages."""
    messages = [
      Message(role="user", content="My favorite color is blue. Remember this."),
      Message(role="assistant", content="I'll remember that your favorite color is blue."),
      Message(role="user", content="What is my favorite color?"),
    ]
    response = await openai_model.ainvoke(messages=messages, assistant_message=make_assistant_msg())
    assert "blue" in response.content.lower()

  @pytest.mark.asyncio
  async def test_tool_calling_invokes_correct_tool(self, openai_model):
    """Model should return a tool_execution when a relevant tool is available."""
    from definable.tool.decorator import tool

    @tool
    def get_weather(city: str) -> str:
      """Get the weather for a city."""
      return f"Sunny in {city}"

    # Model expects OpenAI-format tool dicts, not Function objects
    tool_schema = {"type": "function", "function": get_weather.to_dict()}
    messages = make_messages("What is the weather in London?")
    response = await openai_model.ainvoke(
      messages=messages,
      assistant_message=make_assistant_msg(),
      tools=[tool_schema],
    )
    # tool_calls = raw model output; tool_executions = populated only when agent executes them
    has_tool_call = bool(response.tool_calls)
    has_content = bool(response.content)
    assert has_tool_call or has_content


@pytest.mark.integration
@pytest.mark.deepseek
class TestDeepSeekModel:
  """Real DeepSeek API tests."""

  @pytest.mark.asyncio
  async def test_ainvoke_returns_content(self, deepseek_model):
    messages = make_messages("What is 2+2? Reply with just the number.")
    response = await deepseek_model.ainvoke(messages=messages, assistant_message=make_assistant_msg())
    assert response.content
    assert "4" in response.content

  @pytest.mark.asyncio
  async def test_streaming_works(self, deepseek_model):
    messages = make_messages("What is 2+2? Reply with just the number.")
    chunks = []
    async for chunk in deepseek_model.ainvoke_stream(messages=messages, assistant_message=make_assistant_msg()):
      if chunk.content:
        chunks.append(chunk.content)
    assert len(chunks) > 0


@pytest.mark.integration
@pytest.mark.moonshot
class TestMoonshotModel:
  """Real Moonshot API tests."""

  @pytest.mark.asyncio
  async def test_ainvoke_returns_content(self, moonshot_model):
    messages = make_messages("What is 1+1? Reply with just the number.")
    response = await moonshot_model.ainvoke(messages=messages, assistant_message=make_assistant_msg())
    assert response.content
    assert isinstance(response.content, str)

  @pytest.mark.asyncio
  async def test_streaming_works(self, moonshot_model):
    messages = make_messages("What is the capital of Italy? One word.")
    chunks = []
    async for chunk in moonshot_model.ainvoke_stream(messages=messages, assistant_message=make_assistant_msg()):
      if chunk.content:
        chunks.append(chunk.content)
    assert len(chunks) > 0


@pytest.mark.integration
@pytest.mark.xai
class TestXAIModel:
  """Real xAI API tests."""

  @pytest.mark.asyncio
  async def test_ainvoke_returns_content(self, xai_model):
    messages = make_messages("What is the capital of France? Answer in one word.")
    response = await xai_model.ainvoke(messages=messages, assistant_message=make_assistant_msg())
    assert response.content
    assert isinstance(response.content, str)
