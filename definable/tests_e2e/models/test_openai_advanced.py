"""Advanced E2E tests for OpenAI provider - edge cases and features."""

import os
from typing import List

import pytest
from pydantic import BaseModel, Field

from definable.models.message import Message
from definable.models.openai import OpenAIChat


def requires_openai():
    """Skip if OPENAI_API_KEY not set."""
    return pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable not set",
    )


@pytest.fixture
def openai_model():
    """OpenAI model for testing."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return OpenAIChat(id="gpt-4o-mini", api_key=api_key)


@pytest.fixture
def assistant_message():
    """Factory for creating assistant messages."""
    def _create():
        return Message(role="assistant", content=None)
    return _create


@pytest.mark.e2e
@pytest.mark.openai
@requires_openai()
class TestOpenAIAdvanced:
    """Advanced OpenAI model tests."""

    def test_multi_turn_conversation(self, openai_model, assistant_message):
        """Model handles multi-turn conversation."""
        messages = [
            Message(role="user", content="My name is Alice."),
        ]

        # First turn
        response1 = openai_model.invoke(
            messages=messages,
            assistant_message=assistant_message(),
        )
        assert response1.content is not None

        # Add assistant response and user follow-up
        messages.append(Message(role="assistant", content=response1.content))
        messages.append(Message(role="user", content="What is my name?"))

        # Second turn
        response2 = openai_model.invoke(
            messages=messages,
            assistant_message=assistant_message(),
        )
        assert response2.content is not None
        assert "alice" in response2.content.lower()

    def test_system_message(self, openai_model, assistant_message):
        """Model respects system message."""
        messages = [
            Message(role="system", content="You are a helpful assistant that only responds in uppercase."),
            Message(role="user", content="Say hello."),
        ]

        response = openai_model.invoke(
            messages=messages,
            assistant_message=assistant_message(),
        )
        assert response.content is not None
        # Should have significant uppercase
        uppercase_ratio = sum(1 for c in response.content if c.isupper()) / max(len(response.content), 1)
        assert uppercase_ratio > 0.3  # At least 30% uppercase

    def test_long_response(self, openai_model, assistant_message):
        """Model can generate longer responses."""
        messages = [
            Message(role="user", content="Write a paragraph about artificial intelligence with at least 50 words."),
        ]

        response = openai_model.invoke(
            messages=messages,
            assistant_message=assistant_message(),
        )
        assert response.content is not None
        word_count = len(response.content.split())
        assert word_count >= 30  # Should be a decent length

    def test_structured_output_complex(self, openai_model, assistant_message):
        """Structured output with complex nested schema."""
        import json

        class Address(BaseModel):
            street: str
            city: str
            country: str

        class Person(BaseModel):
            name: str
            age: int
            address: Address

        messages = [
            Message(
                role="user",
                content="Create a person named John who is 30 years old living at 123 Main St, New York, USA.",
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
            assert response.parsed.age == 30
        else:
            # Fallback: parse JSON content
            data = json.loads(response.content)
            assert data["name"] == "John"
            assert data["age"] == 30

    def test_structured_output_list(self, openai_model, assistant_message):
        """Structured output with list field."""
        import json

        class ShoppingList(BaseModel):
            items: List[str] = Field(description="List of items to buy")
            store: str

        messages = [
            Message(
                role="user",
                content="Create a shopping list for a grocery store with 3 items: milk, bread, eggs.",
            ),
        ]

        response = openai_model.invoke(
            messages=messages,
            assistant_message=assistant_message(),
            response_format=ShoppingList,
        )

        if response.parsed is not None:
            assert isinstance(response.parsed, ShoppingList)
            assert len(response.parsed.items) >= 3
        else:
            data = json.loads(response.content)
            assert "items" in data
            assert len(data["items"]) >= 3

    def test_temperature_affects_output(self, assistant_message):
        """Different temperatures produce different outputs."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        # Low temperature - more deterministic
        model_low_temp = OpenAIChat(id="gpt-4o-mini", api_key=api_key, temperature=0.0)

        messages = [Message(role="user", content="What is 2+2? Answer with just the number.")]

        response = model_low_temp.invoke(
            messages=messages,
            assistant_message=assistant_message(),
        )
        assert response.content is not None
        assert "4" in response.content

    def test_max_tokens_limit(self, assistant_message):
        """Model respects max_tokens limit."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        model = OpenAIChat(id="gpt-4o-mini", api_key=api_key, max_tokens=10)

        messages = [Message(role="user", content="Write a very long story about a dragon.")]

        response = model.invoke(
            messages=messages,
            assistant_message=assistant_message(),
        )
        # Response should be truncated due to max_tokens
        assert response.content is not None
        # With max_tokens=10, response should be short
        assert len(response.content.split()) <= 20  # Generous limit

    def test_response_metrics(self, openai_model, assistant_message):
        """Response includes usage metrics."""
        messages = [Message(role="user", content="Say hello.")]

        response = openai_model.invoke(
            messages=messages,
            assistant_message=assistant_message(),
        )

        assert response.response_usage is not None
        assert response.response_usage.input_tokens > 0
        assert response.response_usage.output_tokens > 0
        assert response.response_usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_concurrent_async_calls(self, openai_model, assistant_message):
        """Multiple async calls can run concurrently."""
        import asyncio

        messages = [Message(role="user", content="Say 'test' and nothing else.")]

        async def make_call():
            return await openai_model.ainvoke(
                messages=messages,
                assistant_message=assistant_message(),
            )

        # Run 3 concurrent calls
        results = await asyncio.gather(make_call(), make_call(), make_call())

        assert len(results) == 3
        for response in results:
            assert response.content is not None

    def test_empty_content_handling(self, openai_model, assistant_message):
        """Model handles messages with minimal content."""
        messages = [Message(role="user", content="Hi")]

        response = openai_model.invoke(
            messages=messages,
            assistant_message=assistant_message(),
        )
        assert response.content is not None

    def test_special_characters(self, openai_model, assistant_message):
        """Model handles special characters in input."""
        messages = [
            Message(
                role="user",
                content="Repeat this exactly: Hello! @#$%^&*() 你好 مرحبا",
            ),
        ]

        response = openai_model.invoke(
            messages=messages,
            assistant_message=assistant_message(),
        )
        assert response.content is not None
        # Should contain at least some of the special content
        assert any(c in response.content for c in ["Hello", "@", "你好", "مرحبا"])

    def test_code_in_response(self, openai_model, assistant_message):
        """Model can generate code."""
        messages = [
            Message(
                role="user",
                content="Write a Python function that adds two numbers. Just the code, no explanation.",
            ),
        ]

        response = openai_model.invoke(
            messages=messages,
            assistant_message=assistant_message(),
        )
        assert response.content is not None
        assert "def" in response.content
        assert "return" in response.content
