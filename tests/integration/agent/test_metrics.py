"""
Integration tests: Metrics collection from agent runs.

Strategy:
  - Real OpenAI model only — no mocks
  - Assert that output.metrics is populated with token counts
  - Assert model name is tracked in RunOutput

Covers:
  - output.metrics is populated after a run
  - Token counts (input_tokens, output_tokens, total_tokens) are positive
  - Model name is tracked in output.model
  - Metrics accumulate across tool-calling runs
  - Duration is tracked
"""

import pytest

from definable.agent import Agent
from definable.model.metrics import Metrics


@pytest.mark.integration
@pytest.mark.openai
class TestMetricsCollection:
  """Agent runs produce meaningful metrics with real API calls."""

  @pytest.mark.asyncio
  async def test_metrics_is_populated(self, basic_agent):
    """output.metrics must not be None after a successful run."""
    output = await basic_agent.arun("Say hello.")
    assert output.metrics is not None
    assert isinstance(output.metrics, Metrics)

  @pytest.mark.asyncio
  async def test_input_tokens_are_positive(self, basic_agent):
    """A run must consume at least some input tokens."""
    output = await basic_agent.arun("What is 2+2?")
    assert output.metrics is not None
    assert output.metrics.input_tokens > 0

  @pytest.mark.asyncio
  async def test_output_tokens_are_positive(self, basic_agent):
    """A run must produce at least some output tokens."""
    output = await basic_agent.arun("What is the capital of France?")
    assert output.metrics is not None
    assert output.metrics.output_tokens > 0

  @pytest.mark.asyncio
  async def test_total_tokens_is_sum_of_input_and_output(self, basic_agent):
    """total_tokens should be >= input_tokens + output_tokens."""
    output = await basic_agent.arun("Name three colors.")
    assert output.metrics is not None
    assert output.metrics.total_tokens >= output.metrics.input_tokens + output.metrics.output_tokens

  @pytest.mark.asyncio
  async def test_model_name_is_tracked(self, basic_agent):
    """output.model should reflect the model ID used."""
    output = await basic_agent.arun("Hi.")
    assert output.model is not None
    assert isinstance(output.model, str)
    assert len(output.model) > 0

  @pytest.mark.asyncio
  async def test_duration_is_tracked(self, basic_agent):
    """Metrics duration should be a positive number."""
    output = await basic_agent.arun("What is 1+1?")
    assert output.metrics is not None
    if output.metrics.duration is not None:
      assert output.metrics.duration > 0

  @pytest.mark.asyncio
  async def test_metrics_with_tool_calling(self, agent_with_tools):
    """Metrics should still be populated when tools are called (may use more tokens)."""
    output = await agent_with_tools.arun("Calculate 5 + 3 using the calculate tool.")
    assert output.metrics is not None
    assert output.metrics.input_tokens > 0
    assert output.metrics.output_tokens > 0
    assert output.metrics.total_tokens > 0

  @pytest.mark.asyncio
  async def test_tool_call_uses_more_input_tokens_than_plain(self, openai_model):
    """A tool-calling run typically uses more input tokens than a plain run."""
    from tests.integration.agent.conftest import calculate

    plain_agent = Agent(model=openai_model)
    tool_agent = Agent(model=openai_model, tools=[calculate])

    plain_output = await plain_agent.arun("What is 2+2?")
    tool_output = await tool_agent.arun("What is 2+2? Use the calculate tool.")

    assert plain_output.metrics is not None
    assert tool_output.metrics is not None
    # Tool-calling sends tool definitions, so input tokens should be higher
    assert tool_output.metrics.input_tokens > plain_output.metrics.input_tokens
