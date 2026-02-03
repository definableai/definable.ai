"""E2E tests for OpenAI provider."""

import pytest
from pydantic import BaseModel

from definable.models.message import Message


@pytest.mark.e2e
@pytest.mark.openai
class TestOpenAIE2E:
    """End-to-end tests for OpenAI models."""

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
        """Async invoke returns ModelResponse with content."""
        response = await openai_model.ainvoke(
            messages=simple_messages,
            assistant_message=assistant_message(),
        )
        assert response.content is not None
        assert len(response.content) > 0

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
        """Test structured output with Pydantic model (OpenAI supports this natively)."""
        import json

        class MathResult(BaseModel):
            answer: int

        messages = [Message(role="user", content="What is 2+2? Return the answer.")]
        response = openai_model.invoke(
            messages=messages,
            assistant_message=assistant_message(),
            response_format=MathResult,
        )
        # Check parsed field if populated, otherwise validate JSON content
        if response.parsed is not None:
            assert isinstance(response.parsed, MathResult)
            assert response.parsed.answer == 4
        else:
            # Fallback: validate the raw JSON response
            parsed = json.loads(response.content)
            assert parsed["answer"] == 4
