"""
Integration tests: Multi-turn conversation via chat history.

Strategy:
  - Real OpenAI model only — no mocks
  - Multi-turn is achieved by passing messages=r1.messages to subsequent calls
  - Assert that the agent recalls context from prior turns

Covers:
  - Messages accumulate across turns (via messages=r1.messages)
  - Agent recalls context from previous messages
  - Run IDs differ between turns
  - Three-turn conversation retains facts from all turns
"""

import pytest

from definable.agent.events import RunStatus


@pytest.mark.integration
@pytest.mark.openai
class TestMultiTurnConversation:
  """Agent maintains context across turns when messages are passed explicitly."""

  @pytest.mark.asyncio
  async def test_agent_recalls_context_from_previous_turn(self, basic_agent):
    """Agent should recall a fact stated in the first turn when asked in the second."""
    out1 = await basic_agent.arun("My name is TestUser123. Remember this.")
    out2 = await basic_agent.arun("What is my name?", messages=out1.messages)
    assert "testuser123" in out2.content.lower() or "testuser" in out2.content.lower()

  @pytest.mark.asyncio
  async def test_messages_accumulate_across_turns(self, basic_agent):
    """Each turn should add messages; the second turn has more messages than the first."""
    out1 = await basic_agent.arun("Hello, I like Python.")
    msg_count_1 = len(out1.messages)

    out2 = await basic_agent.arun("What language do I like?", messages=out1.messages)
    msg_count_2 = len(out2.messages)

    assert msg_count_2 > msg_count_1
    assert "python" in out2.content.lower()

  @pytest.mark.asyncio
  async def test_run_ids_differ_between_turns(self, basic_agent):
    """Each call to arun() must produce a unique run_id, even in multi-turn."""
    out1 = await basic_agent.arun("First turn.")
    out2 = await basic_agent.arun("Second turn.", messages=out1.messages)
    assert out1.run_id is not None
    assert out2.run_id is not None
    assert out1.run_id != out2.run_id

  @pytest.mark.asyncio
  async def test_three_turn_conversation_retains_all_context(self, basic_agent):
    """Agent should recall facts from turn 1 and turn 2 when asked in turn 3."""
    out1 = await basic_agent.arun("I work at a company called Acme Corp.")
    out2 = await basic_agent.arun("I prefer Python over JavaScript.", messages=out1.messages)
    out3 = await basic_agent.arun(
      "Where do I work and what is my preferred language?",
      messages=out2.messages,
    )

    content = out3.content.lower()
    assert "acme" in content
    assert "python" in content

  @pytest.mark.asyncio
  async def test_status_is_completed_on_every_turn(self, basic_agent):
    """Every turn in a multi-turn conversation should complete successfully."""
    out1 = await basic_agent.arun("Hello.")
    assert out1.status == RunStatus.completed

    out2 = await basic_agent.arun("How are you?", messages=out1.messages)
    assert out2.status == RunStatus.completed
