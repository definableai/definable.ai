"""E2E tests for DeepSeek provider."""

import pytest


@pytest.mark.e2e
@pytest.mark.deepseek
class TestDeepSeekE2E:
    """End-to-end tests for DeepSeek models."""

    def test_invoke_basic(self, deepseek_model, simple_messages, assistant_message):
        """Sync invoke returns non-empty content."""
        response = deepseek_model.invoke(
            messages=simple_messages,
            assistant_message=assistant_message(),
        )
        assert response.content is not None
        assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_ainvoke_basic(self, deepseek_model, simple_messages, assistant_message):
        """Async invoke returns ModelResponse with content."""
        response = await deepseek_model.ainvoke(
            messages=simple_messages,
            assistant_message=assistant_message(),
        )
        assert response.content is not None
        assert len(response.content) > 0

    def test_invoke_stream(self, deepseek_model, simple_messages, assistant_message):
        """Streaming invoke yields at least one chunk."""
        chunks = list(
            deepseek_model.invoke_stream(
                messages=simple_messages,
                assistant_message=assistant_message(),
            )
        )
        assert len(chunks) > 0

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
