"""E2E tests for xAI provider."""

import pytest


@pytest.mark.e2e
@pytest.mark.xai
class TestXaiE2E:
  """End-to-end tests for xAI models."""

  def test_invoke_basic(self, xai_model, simple_messages, assistant_message):
    """Sync invoke returns non-empty content."""
    response = xai_model.invoke(
      messages=simple_messages,
      assistant_message=assistant_message(),
    )
    assert response.content is not None
    assert len(response.content) > 0

  @pytest.mark.asyncio
  async def test_ainvoke_basic(self, xai_model, simple_messages, assistant_message):
    """Async invoke returns ModelResponse with content."""
    response = await xai_model.ainvoke(
      messages=simple_messages,
      assistant_message=assistant_message(),
    )
    assert response.content is not None
    assert len(response.content) > 0

  def test_invoke_stream(self, xai_model, simple_messages, assistant_message):
    """Streaming invoke yields at least one chunk."""
    chunks = list(
      xai_model.invoke_stream(
        messages=simple_messages,
        assistant_message=assistant_message(),
      )
    )
    assert len(chunks) > 0

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
