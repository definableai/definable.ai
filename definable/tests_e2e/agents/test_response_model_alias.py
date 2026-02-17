"""E2E tests for response_model alias parameter (issue #13).

Verifies that Agent.run(), Agent.arun(), Agent.run_stream(), and Agent.arun_stream()
accept ``response_model`` as an alias for ``output_schema``, following the
Pydantic / FastAPI naming convention.
"""

import warnings
from typing import List

import pytest
from pydantic import BaseModel

from definable.agents import Agent, AgentConfig, TracingConfig
from definable.agents.testing import MockModel


# --- Fixtures ---


class Person(BaseModel):
    name: str
    age: int


class Summary(BaseModel):
    title: str
    points: List[str]


# --- Tests ---


@pytest.mark.e2e
class TestResponseModelAlias:
    """Tests for the response_model parameter alias on Agent run methods."""

    # --- run() ---

    def test_run_accepts_response_model(self):
        """Agent.run() should accept response_model without TypeError."""
        model = MockModel(responses=["John is 30."])
        agent = Agent(
            model=model,
            config=AgentConfig(tracing=TracingConfig(enabled=False)),
        )
        output = agent.run("Extract: John is 30", response_model=Person)
        assert output.content == "John is 30."

    def test_run_output_schema_still_works(self):
        """Agent.run() should still accept output_schema (backwards compat)."""
        model = MockModel(responses=["John is 30."])
        agent = Agent(
            model=model,
            config=AgentConfig(tracing=TracingConfig(enabled=False)),
        )
        output = agent.run("Extract: John is 30", output_schema=Person)
        assert output.content == "John is 30."

    def test_run_both_params_warns(self):
        """Agent.run() warns when both output_schema and response_model are given."""
        model = MockModel(responses=["John is 30."])
        agent = Agent(
            model=model,
            config=AgentConfig(tracing=TracingConfig(enabled=False)),
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output = agent.run(
                "Extract: John is 30",
                output_schema=Summary,
                response_model=Person,
            )
            assert len(w) == 1
            assert "response_model" in str(w[0].message)
            assert "output_schema" in str(w[0].message)
        assert output.content == "John is 30."

    # --- arun() ---

    @pytest.mark.asyncio
    async def test_arun_accepts_response_model(self):
        """Agent.arun() should accept response_model without TypeError."""
        model = MockModel(responses=["John is 30."])
        agent = Agent(
            model=model,
            config=AgentConfig(tracing=TracingConfig(enabled=False)),
        )
        output = await agent.arun("Extract: John is 30", response_model=Person)
        assert output.content == "John is 30."

    @pytest.mark.asyncio
    async def test_arun_output_schema_still_works(self):
        """Agent.arun() should still accept output_schema (backwards compat)."""
        model = MockModel(responses=["John is 30."])
        agent = Agent(
            model=model,
            config=AgentConfig(tracing=TracingConfig(enabled=False)),
        )
        output = await agent.arun("Extract: John is 30", output_schema=Person)
        assert output.content == "John is 30."

    @pytest.mark.asyncio
    async def test_arun_both_params_warns(self):
        """Agent.arun() warns when both output_schema and response_model are given."""
        model = MockModel(responses=["John is 30."])
        agent = Agent(
            model=model,
            config=AgentConfig(tracing=TracingConfig(enabled=False)),
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output = await agent.arun(
                "Extract: John is 30",
                output_schema=Summary,
                response_model=Person,
            )
            assert len(w) == 1
            assert "response_model" in str(w[0].message)
        assert output.content == "John is 30."

    # --- run_stream() ---

    def test_run_stream_accepts_response_model(self):
        """Agent.run_stream() should accept response_model without TypeError."""
        model = MockModel(responses=["Streamed response."])
        agent = Agent(
            model=model,
            config=AgentConfig(tracing=TracingConfig(enabled=False)),
        )
        events = list(agent.run_stream("Hello", response_model=Person))
        assert len(events) > 0

    def test_run_stream_output_schema_still_works(self):
        """Agent.run_stream() should still accept output_schema."""
        model = MockModel(responses=["Streamed response."])
        agent = Agent(
            model=model,
            config=AgentConfig(tracing=TracingConfig(enabled=False)),
        )
        events = list(agent.run_stream("Hello", output_schema=Person))
        assert len(events) > 0

    # --- arun_stream() ---

    @pytest.mark.asyncio
    async def test_arun_stream_accepts_response_model(self):
        """Agent.arun_stream() should accept response_model without TypeError."""
        model = MockModel(responses=["Streamed response."])
        agent = Agent(
            model=model,
            config=AgentConfig(tracing=TracingConfig(enabled=False)),
        )
        events = []
        async for event in agent.arun_stream("Hello", response_model=Person):
            events.append(event)
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_arun_stream_output_schema_still_works(self):
        """Agent.arun_stream() should still accept output_schema."""
        model = MockModel(responses=["Streamed response."])
        agent = Agent(
            model=model,
            config=AgentConfig(tracing=TracingConfig(enabled=False)),
        )
        events = []
        async for event in agent.arun_stream("Hello", output_schema=Person):
            events.append(event)
        assert len(events) > 0

    # --- Edge cases ---

    @pytest.mark.asyncio
    async def test_neither_param_provided(self):
        """Omitting both output_schema and response_model should work normally."""
        model = MockModel(responses=["Plain response."])
        agent = Agent(
            model=model,
            config=AgentConfig(tracing=TracingConfig(enabled=False)),
        )
        output = await agent.arun("Hello")
        assert output.content == "Plain response."

    @pytest.mark.asyncio
    async def test_response_model_takes_precedence_over_output_schema(self):
        """When both are provided, response_model should be the one used."""
        from unittest.mock import MagicMock
        from definable.models.metrics import Metrics

        call_kwargs = {}

        def side_effect(messages, tools, **kwargs):
            call_kwargs.update(kwargs)
            response = MagicMock()
            response.content = "Result"
            response.tool_calls = []
            response.tool_executions = []
            response.metrics = Metrics()
            response.reasoning_content = None
            response.citations = None
            response.images = None
            response.videos = None
            response.audios = None
            return response

        model = MockModel(side_effect=side_effect)
        agent = Agent(
            model=model,
            config=AgentConfig(tracing=TracingConfig(enabled=False)),
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            await agent.arun(
                "Extract info",
                output_schema=Summary,
                response_model=Person,
            )

        # The model should have been called with Person as response_format
        assert call_kwargs.get("response_format") is Person

    def test_resolve_output_schema_static_method(self):
        """_resolve_output_schema correctly resolves parameter combinations."""
        # Only output_schema
        assert Agent._resolve_output_schema(Person, None) is Person

        # Only response_model
        assert Agent._resolve_output_schema(None, Person) is Person

        # Neither
        assert Agent._resolve_output_schema(None, None) is None

        # Both (response_model wins)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = Agent._resolve_output_schema(Summary, Person)
            assert result is Person
            assert len(w) == 1
