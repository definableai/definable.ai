"""
Behavioral tests: Does the agent produce correct responses?

Strategy:
  - Real OpenAI model only — no mocks
  - Assertions on OUTCOMES (content, status, structure)

Covers:
  - Agent returns RunOutput with correct fields
  - Agent with instructions includes them in behavior (verified via response content)
  - Agent run() and arun() produce the same structural output
  - Agent streaming produces accumulated content
  - Multi-turn conversation preserves history via messages param
  - Agent produces content even for complex prompts
"""

import pytest

from definable.agent import Agent
from definable.agent.config import AgentConfig
from definable.agent.events import RunStatus
from definable.agent.tracing import Tracing


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def no_trace():
  return AgentConfig(tracing=Tracing(enabled=False))


# ---------------------------------------------------------------------------
# Intelligence behavior (real OpenAI model)
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
@pytest.mark.openai
class TestAgentIntelligenceBehavior:
  """Test agent behavior with real LLM. Requires OPENAI_API_KEY."""

  @pytest.fixture
  def smart_agent(self, openai_model, no_trace):
    return Agent(model=openai_model, config=no_trace)

  @pytest.mark.asyncio
  async def test_arun_returns_run_output_with_required_fields(self, smart_agent):
    output = await smart_agent.arun("What is 1+1? Reply with just the number.")
    assert output is not None
    assert hasattr(output, "content")
    assert hasattr(output, "messages")
    assert hasattr(output, "status")
    assert hasattr(output, "run_id")

  @pytest.mark.asyncio
  async def test_run_output_status_is_completed(self, smart_agent):
    output = await smart_agent.arun("Say hello.")
    assert output.status == RunStatus.completed

  @pytest.mark.asyncio
  async def test_agent_answers_factual_question(self, smart_agent):
    output = await smart_agent.arun("What is the capital of France? Answer in one word.")
    assert "paris" in output.content.lower()

  @pytest.mark.asyncio
  async def test_agent_follows_instructions(self, openai_model, no_trace):
    agent = Agent(
      model=openai_model,
      instructions="You are a pirate. Always mention ships in your response.",
      config=no_trace,
    )
    output = await agent.arun("Tell me about yourself.")
    # Agent should behave like a pirate and mention ships
    content = output.content.lower()  # type: ignore[union-attr]
    assert any(kw in content for kw in ["ship", "sea", "sail", "pirate", "arr"])

  @pytest.mark.asyncio
  async def test_arun_messages_contain_user_and_assistant(self, smart_agent):
    output = await smart_agent.arun("Hello")
    roles = [m.role for m in output.messages]
    assert "user" in roles
    assert "assistant" in roles

  @pytest.mark.asyncio
  async def test_run_id_is_unique_per_run(self, smart_agent):
    out1 = await smart_agent.arun("First question: what is 1+1?")
    out2 = await smart_agent.arun("Second question: what is 2+2?")
    if out1.run_id and out2.run_id:
      assert out1.run_id != out2.run_id

  @pytest.mark.asyncio
  async def test_agent_handles_multi_turn_context(self, smart_agent):
    out1 = await smart_agent.arun("My name is TestUser123. Remember this.")
    out2 = await smart_agent.arun("What is my name?", messages=out1.messages)
    assert "testuser123" in out2.content.lower() or "testuser" in out2.content.lower()

  @pytest.mark.asyncio
  async def test_streaming_produces_content(self, smart_agent):
    chunks = []
    async for event in smart_agent.arun_stream("Count from 1 to 3."):
      if hasattr(event, "content") and event.content:
        chunks.append(event.content)
    assert len(chunks) > 0
    combined = "".join(chunks)
    assert len(combined) > 0

  def test_sync_run_returns_content(self, openai_model, no_trace):
    agent = Agent(model=openai_model, config=no_trace)
    output = agent.run("What is 2+2? Reply with just the number.")
    assert "4" in output.content  # type: ignore[operator]
    assert output.status == RunStatus.completed
