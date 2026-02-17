"""Tests for structured output parsing in OpenAIChat._parse_provider_response.

Validates that ModelResponse.parsed is correctly populated when
response_format is a Pydantic BaseModel subclass.

Fixes: https://github.com/Anandesh-Sharma/definable.ai/issues/6
"""

import json
from typing import List, Optional
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from definable.models.openai import OpenAIChat
from definable.models.response import ModelResponse


# --- Test Pydantic models ---


class CityInfo(BaseModel):
    """Simple structured output model."""

    name: str
    country: str


class MathResult(BaseModel):
    """Model for math computation results."""

    answer: int


class Person(BaseModel):
    """Nested structured output model."""

    class Address(BaseModel):
        city: str
        country: str

    name: str
    age: int
    address: Address


class TaskList(BaseModel):
    """Complex structured output model with lists and optional fields."""

    class TaskItem(BaseModel):
        title: str
        priority: str = Field(description="Priority: high, medium, low")
        estimated_hours: Optional[float] = None

    tasks: List[TaskItem]
    total_estimated_hours: Optional[float] = None


# --- Helpers ---


def _make_chat_completion(content: str) -> MagicMock:
    """Create a mock ChatCompletion object matching the OpenAI SDK shape."""
    message = MagicMock()
    message.role = "assistant"
    message.content = content
    message.tool_calls = None
    message.audio = None
    # Ensure reasoning_content and reasoning are not present
    message.reasoning_content = None
    message.reasoning = None

    choice = MagicMock()
    choice.message = message

    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = None
    completion.error = None
    completion.id = "chatcmpl-test"
    completion.system_fingerprint = None
    completion.model_extra = None

    return completion


def _make_model() -> OpenAIChat:
    """Create an OpenAIChat instance without needing an API key (for unit testing)."""
    return OpenAIChat(id="gpt-4o-mini", api_key="test-key")


# --- Tests ---


class TestStructuredOutputParsing:
    """Tests for _parse_provider_response with structured output."""

    def test_parsed_populated_for_simple_model(self):
        """ModelResponse.parsed should contain a CityInfo instance when response_format=CityInfo."""
        model = _make_model()
        content = json.dumps({"name": "Paris", "country": "France"})
        completion = _make_chat_completion(content)

        response = model._parse_provider_response(completion, response_format=CityInfo)

        assert response.content == content
        assert response.parsed is not None
        assert isinstance(response.parsed, CityInfo)
        assert response.parsed.name == "Paris"
        assert response.parsed.country == "France"

    def test_parsed_populated_for_nested_model(self):
        """ModelResponse.parsed should work with nested Pydantic models."""
        model = _make_model()
        content = json.dumps({
            "name": "John",
            "age": 30,
            "address": {"city": "New York", "country": "USA"},
        })
        completion = _make_chat_completion(content)

        response = model._parse_provider_response(completion, response_format=Person)

        assert response.parsed is not None
        assert isinstance(response.parsed, Person)
        assert response.parsed.name == "John"
        assert response.parsed.age == 30
        assert response.parsed.address.city == "New York"
        assert response.parsed.address.country == "USA"

    def test_parsed_populated_for_complex_model(self):
        """ModelResponse.parsed should work with lists and optional fields."""
        model = _make_model()
        content = json.dumps({
            "tasks": [
                {"title": "Design UI", "priority": "high", "estimated_hours": 4.0},
                {"title": "Write tests", "priority": "medium"},
            ],
            "total_estimated_hours": 8.0,
        })
        completion = _make_chat_completion(content)

        response = model._parse_provider_response(completion, response_format=TaskList)

        assert response.parsed is not None
        assert isinstance(response.parsed, TaskList)
        assert len(response.parsed.tasks) == 2
        assert response.parsed.tasks[0].title == "Design UI"
        assert response.parsed.tasks[0].estimated_hours == 4.0
        assert response.parsed.tasks[1].estimated_hours is None
        assert response.parsed.total_estimated_hours == 8.0

    def test_parsed_none_when_no_response_format(self):
        """ModelResponse.parsed should be None when response_format is not provided."""
        model = _make_model()
        content = "Just a plain text response"
        completion = _make_chat_completion(content)

        response = model._parse_provider_response(completion, response_format=None)

        assert response.content == content
        assert response.parsed is None

    def test_parsed_none_when_response_format_is_dict(self):
        """ModelResponse.parsed should be None when response_format is a dict (not a Pydantic model)."""
        model = _make_model()
        content = json.dumps({"key": "value"})
        completion = _make_chat_completion(content)

        response = model._parse_provider_response(
            completion,
            response_format={"type": "json_object"},
        )

        assert response.content == content
        assert response.parsed is None

    def test_parsed_none_when_content_is_invalid_json(self):
        """ModelResponse.parsed should be None (with warning) when content is not valid JSON."""
        model = _make_model()
        content = "This is not valid JSON"
        completion = _make_chat_completion(content)

        response = model._parse_provider_response(completion, response_format=CityInfo)

        assert response.content == content
        assert response.parsed is None

    def test_parsed_none_when_json_doesnt_match_schema(self):
        """ModelResponse.parsed should be None (with warning) when JSON doesn't match the schema."""
        model = _make_model()
        content = json.dumps({"wrong_field": "value"})
        completion = _make_chat_completion(content)

        response = model._parse_provider_response(completion, response_format=CityInfo)

        assert response.content == content
        assert response.parsed is None

    def test_parsed_with_math_result(self):
        """ModelResponse.parsed should work with simple numeric models."""
        model = _make_model()
        content = json.dumps({"answer": 42})
        completion = _make_chat_completion(content)

        response = model._parse_provider_response(completion, response_format=MathResult)

        assert response.parsed is not None
        assert isinstance(response.parsed, MathResult)
        assert response.parsed.answer == 42

    def test_content_preserved_alongside_parsed(self):
        """Both content (raw JSON string) and parsed (Pydantic instance) should be available."""
        model = _make_model()
        data = {"name": "Tokyo", "country": "Japan"}
        content = json.dumps(data)
        completion = _make_chat_completion(content)

        response = model._parse_provider_response(completion, response_format=CityInfo)

        # Raw content is the JSON string
        assert response.content == content
        # Parsed is the validated Pydantic model
        assert response.parsed is not None
        assert response.parsed.name == "Tokyo"

    def test_parsed_none_when_content_is_empty(self):
        """ModelResponse.parsed should be None when content is empty."""
        model = _make_model()
        completion = _make_chat_completion("")

        # Empty string content - response_message.content is "" which is not None
        # but model_response.content will be "" which is falsy
        response = model._parse_provider_response(completion, response_format=CityInfo)

        # Empty string is falsy, so parsed should not be attempted
        assert response.parsed is None

    def test_parsed_none_when_content_is_none(self):
        """ModelResponse.parsed should be None when content is None from provider."""
        model = _make_model()
        message = MagicMock()
        message.role = "assistant"
        message.content = None
        message.tool_calls = None
        message.audio = None
        message.reasoning_content = None
        message.reasoning = None

        choice = MagicMock()
        choice.message = message

        completion = MagicMock()
        completion.choices = [choice]
        completion.usage = None
        completion.error = None
        completion.id = "chatcmpl-test"
        completion.system_fingerprint = None
        completion.model_extra = None

        response = model._parse_provider_response(completion, response_format=CityInfo)

        assert response.content is None
        assert response.parsed is None

    def test_other_response_fields_unaffected(self):
        """Structured output parsing should not affect other ModelResponse fields."""
        model = _make_model()
        content = json.dumps({"name": "Berlin", "country": "Germany"})
        completion = _make_chat_completion(content)

        response = model._parse_provider_response(completion, response_format=CityInfo)

        assert response.role == "assistant"
        assert response.parsed is not None
        # Tool calls should be empty list (default)
        assert response.tool_calls == []
