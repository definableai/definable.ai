"""E2E tests for Anthropic Claude provider."""

import pytest


@pytest.mark.e2e
@pytest.mark.anthropic
class TestAnthropicE2E:
    """End-to-end tests for Anthropic Claude models."""

    def test_invoke_basic(self, anthropic_model, simple_messages, assistant_message):
        """Sync invoke returns non-empty content."""
        response = anthropic_model.invoke(
            messages=simple_messages,
            assistant_message=assistant_message(),
        )
        assert response.content is not None
        assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_ainvoke_basic(self, anthropic_model, simple_messages, assistant_message):
        """Async invoke returns ModelResponse with content."""
        response = await anthropic_model.ainvoke(
            messages=simple_messages,
            assistant_message=assistant_message(),
        )
        assert response.content is not None
        assert len(response.content) > 0

    def test_invoke_stream(self, anthropic_model, simple_messages, assistant_message):
        """Streaming invoke yields at least one chunk."""
        chunks = list(
            anthropic_model.invoke_stream(
                messages=simple_messages,
                assistant_message=assistant_message(),
            )
        )
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_ainvoke_stream(self, anthropic_model, simple_messages, assistant_message):
        """Async streaming invoke yields chunks."""
        chunks = []
        async for chunk in anthropic_model.ainvoke_stream(
            messages=simple_messages,
            assistant_message=assistant_message(),
        ):
            chunks.append(chunk)
        assert len(chunks) > 0
