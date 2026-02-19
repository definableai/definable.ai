"""
Integration tests: Agent basic response behavior with real OpenAI API.

Migrated from tests_e2e/behavioral/test_basic_responses.py.

Strategy:
  - Real OpenAI model only — no mocks
  - Assertions on OUTCOMES (content, status, structure)
  - Uses fixtures from conftest: basic_agent, openai_model

Covers:
  - Agent.arun() returns RunOutput with correct fields
  - Agent.run() sync path works
  - Status is completed
  - Agent answers factual questions
  - Agent follows system instructions
  - Messages contain user and assistant roles
  - Run ID is unique per run
  - Streaming produces accumulated content
"""

import pytest

from definable.agent import Agent
from definable.agent.events import RunOutput, RunStatus


# ---------------------------------------------------------------------------
# Basic response structure
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.openai
class TestAgentBasicResponses:
  """Agent produces correct response structure with real LLM."""

  @pytest.mark.asyncio
  async def test_arun_returns_run_output_with_required_fields(self, basic_agent):
    """arun() must return a RunOutput with content, messages, status, and run_id."""
    output = await basic_agent.arun("What is 1+1? Reply with just the number.")
    assert isinstance(output, RunOutput)
    assert output.content is not None
    assert output.messages is not None
    assert output.status is not None
    assert output.run_id is not None

  @pytest.mark.asyncio
  async def test_run_output_status_is_completed(self, basic_agent):
    """A successful run must have status == COMPLETED."""
    output = await basic_agent.arun("Say hello.")
    assert output.status == RunStatus.completed

  @pytest.mark.asyncio
  async def test_agent_answers_factual_question(self, basic_agent):
    """Agent should answer a factual question correctly."""
    output = await basic_agent.arun("What is the capital of France? Answer in one word.")
    assert "paris" in output.content.lower()

  @pytest.mark.asyncio
  async def test_agent_follows_instructions(self, openai_model):
    """Agent with custom instructions should reflect them in responses."""
    agent = Agent(
      model=openai_model,
      instructions="You are a pirate. Always mention ships in your response.",
    )
    output = await agent.arun("Tell me about yourself.")
    content = output.content.lower()
    assert any(kw in content for kw in ["ship", "sea", "sail", "pirate", "arr", "vessel"])

  @pytest.mark.asyncio
  async def test_arun_messages_contain_user_and_assistant(self, basic_agent):
    """Messages list must contain both user and assistant roles."""
    output = await basic_agent.arun("Hello")
    roles = [m.role for m in output.messages]
    assert "user" in roles
    assert "assistant" in roles

  @pytest.mark.asyncio
  async def test_run_id_is_unique_per_run(self, basic_agent):
    """Each arun() invocation must produce a distinct run_id."""
    out1 = await basic_agent.arun("First question: what is 1+1?")
    out2 = await basic_agent.arun("Second question: what is 2+2?")
    assert out1.run_id is not None
    assert out2.run_id is not None
    assert out1.run_id != out2.run_id

  @pytest.mark.asyncio
  async def test_content_is_non_empty_string(self, basic_agent):
    """Content must be a non-empty string for a simple prompt."""
    output = await basic_agent.arun("What is 2+2? Reply with just the number.")
    assert isinstance(output.content, str)
    assert len(output.content) > 0


# ---------------------------------------------------------------------------
# Sync path
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.openai
class TestAgentSyncRun:
  """Agent.run() sync path works correctly."""

  def test_sync_run_returns_content(self, openai_model):
    """Sync run() should return RunOutput with content and completed status."""
    agent = Agent(model=openai_model)
    output = agent.run("What is 2+2? Reply with just the number.")
    assert "4" in output.content
    assert output.status == RunStatus.completed


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.openai
class TestAgentStreaming:
  """Agent streaming produces accumulated content."""

  @pytest.mark.asyncio
  async def test_streaming_produces_content(self, basic_agent):
    """arun_stream() should yield events with content chunks."""
    chunks = []
    async for event in basic_agent.arun_stream("Count from 1 to 3."):
      if hasattr(event, "content") and event.content:
        chunks.append(event.content)
    assert len(chunks) > 0
    combined = "".join(str(c) for c in chunks)
    assert len(combined) > 0
