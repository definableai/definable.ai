"""Tests for SummarizeStrategy — hybrid pin + summarize-middle + keep-recent."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from definable.memory.strategies.summarize import SummarizeStrategy
from definable.memory.types import MemoryEntry


def _make_entries(count: int, session_id: str = "s1") -> list[MemoryEntry]:
  """Create a list of alternating user/assistant entries."""
  entries = []
  for i in range(count):
    role = "user" if i % 2 == 0 else "assistant"
    entries.append(
      MemoryEntry(
        memory_id=f"m-{i}",
        session_id=session_id,
        role=role,
        content=f"Message {i}",
        created_at=float(i),
        updated_at=float(i),
      )
    )
  return entries


def _make_mock_model(summary_text: str = "This is a summary.") -> MagicMock:
  """Create a mock model that returns a summary."""
  model = MagicMock()
  response = MagicMock()
  response.content = summary_text
  model.ainvoke = AsyncMock(return_value=response)
  return model


class TestSummarizeStrategy:
  @pytest.mark.asyncio
  async def test_no_op_below_threshold(self):
    """If entries <= pin_count + recent_count, return as-is."""
    strategy = SummarizeStrategy(pin_count=2, recent_count=3)
    entries = _make_entries(5)
    model = _make_mock_model()

    result = await strategy.optimize(entries, model)
    assert len(result) == 5
    model.ainvoke.assert_not_called()

  @pytest.mark.asyncio
  async def test_basic_summarization(self):
    """Middle entries get summarized; pin + recent preserved."""
    strategy = SummarizeStrategy(pin_count=2, recent_count=3)
    entries = _make_entries(10)
    model = _make_mock_model("Summarized middle conversation.")

    result = await strategy.optimize(entries, model)

    # pin(2) + summary(1) + recent(3) = 6
    assert len(result) == 6
    model.ainvoke.assert_called_once()

    # Verify pinned entries preserved
    assert result[0].content == "Message 0"
    assert result[1].content == "Message 1"

    # Verify summary entry
    assert result[2].role == "summary"
    assert result[2].content == "Summarized middle conversation."

    # Verify recent entries preserved
    assert result[3].content == "Message 7"
    assert result[4].content == "Message 8"
    assert result[5].content == "Message 9"

  @pytest.mark.asyncio
  async def test_summary_entry_has_correct_metadata(self):
    """Summary entry inherits session_id and user_id."""
    strategy = SummarizeStrategy(pin_count=1, recent_count=1)
    entries = _make_entries(5, session_id="test-session")
    for e in entries:
      e.user_id = "alice"
    model = _make_mock_model("Summary text.")

    result = await strategy.optimize(entries, model)
    summary = [e for e in result if e.role == "summary"]
    assert len(summary) == 1
    assert summary[0].session_id == "test-session"
    assert summary[0].user_id == "alice"

  @pytest.mark.asyncio
  async def test_tool_call_boundary_pin(self):
    """Tool results at the start of middle get pulled into pinned."""
    strategy = SummarizeStrategy(pin_count=2, recent_count=2)
    entries = [
      MemoryEntry(memory_id="m0", session_id="s1", role="user", content="Hi", created_at=0.0, updated_at=0.0),
      MemoryEntry(memory_id="m1", session_id="s1", role="assistant", content="Calling tool", created_at=1.0, updated_at=1.0),
      MemoryEntry(memory_id="m2", session_id="s1", role="tool", content="Tool result", created_at=2.0, updated_at=2.0),
      MemoryEntry(memory_id="m3", session_id="s1", role="user", content="Thanks", created_at=3.0, updated_at=3.0),
      MemoryEntry(memory_id="m4", session_id="s1", role="assistant", content="Welcome", created_at=4.0, updated_at=4.0),
      MemoryEntry(memory_id="m5", session_id="s1", role="user", content="Bye", created_at=5.0, updated_at=5.0),
      MemoryEntry(memory_id="m6", session_id="s1", role="assistant", content="Goodbye", created_at=6.0, updated_at=6.0),
    ]
    model = _make_mock_model("Mid summary")

    result = await strategy.optimize(entries, model)

    # Tool result at index 2 should be pulled into pinned
    # pinned = [m0, m1, m2], middle = [m3, m4], recent = [m5, m6]
    # Result: 3 pinned + 1 summary + 2 recent = 6
    pinned = [e for e in result if e.role != "summary"]
    assert any(e.content == "Tool result" for e in pinned[:3])

  @pytest.mark.asyncio
  async def test_tool_call_boundary_recent(self):
    """Tool results at start of recent pull preceding entries from middle."""
    strategy = SummarizeStrategy(pin_count=1, recent_count=2)
    entries = [
      MemoryEntry(memory_id="m0", session_id="s1", role="user", content="Start", created_at=0.0, updated_at=0.0),
      MemoryEntry(memory_id="m1", session_id="s1", role="assistant", content="Middle1", created_at=1.0, updated_at=1.0),
      MemoryEntry(memory_id="m2", session_id="s1", role="user", content="Middle2", created_at=2.0, updated_at=2.0),
      MemoryEntry(memory_id="m3", session_id="s1", role="assistant", content="Calling", created_at=3.0, updated_at=3.0),
      MemoryEntry(memory_id="m4", session_id="s1", role="tool", content="Result", created_at=4.0, updated_at=4.0),
      MemoryEntry(memory_id="m5", session_id="s1", role="user", content="Final", created_at=5.0, updated_at=5.0),
    ]
    model = _make_mock_model("Mid summary")

    result = await strategy.optimize(entries, model)

    # Recent starts at [m4(tool), m5]. m4 is tool → pull m3 from middle.
    # So recent becomes [m3, m4, m5]
    recent_content = [e.content for e in result if e.role != "summary"]
    assert "Calling" in recent_content
    assert "Result" in recent_content
    assert "Final" in recent_content

  @pytest.mark.asyncio
  async def test_recursive_summary(self):
    """When pinned contains a summary, it's included in the prompt."""
    strategy = SummarizeStrategy(pin_count=2, recent_count=2)
    entries = [
      MemoryEntry(memory_id="m0", session_id="s1", role="summary", content="Previous: user asked about weather", created_at=0.0, updated_at=0.0),
      MemoryEntry(memory_id="m1", session_id="s1", role="user", content="What about tomorrow?", created_at=1.0, updated_at=1.0),
      MemoryEntry(memory_id="m2", session_id="s1", role="assistant", content="Tomorrow will be sunny", created_at=2.0, updated_at=2.0),
      MemoryEntry(memory_id="m3", session_id="s1", role="user", content="And Thursday?", created_at=3.0, updated_at=3.0),
      MemoryEntry(memory_id="m4", session_id="s1", role="assistant", content="Thursday rainy", created_at=4.0, updated_at=4.0),
      MemoryEntry(memory_id="m5", session_id="s1", role="user", content="Thanks", created_at=5.0, updated_at=5.0),
    ]
    model = _make_mock_model("Extended weather discussion summary")

    await strategy.optimize(entries, model)

    # The LLM call should have been made with prior summary context
    call_args = model.ainvoke.call_args
    prompt_content = call_args[1]["messages"][0].content
    assert "Previous: user asked about weather" in prompt_content

  @pytest.mark.asyncio
  async def test_fallback_on_model_failure(self):
    """If LLM fails, a fallback summary is generated."""
    strategy = SummarizeStrategy(pin_count=1, recent_count=1)
    entries = _make_entries(5)
    model = MagicMock()
    model.ainvoke = AsyncMock(side_effect=RuntimeError("API error"))

    result = await strategy.optimize(entries, model)

    # Should still produce output with fallback summary
    summaries = [e for e in result if e.role == "summary"]
    assert len(summaries) == 1
    assert "Summary of" in summaries[0].content

  @pytest.mark.asyncio
  async def test_count_tokens(self):
    strategy = SummarizeStrategy()
    entries = [
      MemoryEntry(session_id="s1", content="Hello world", created_at=1.0, updated_at=1.0),
      MemoryEntry(session_id="s1", content="How are you?", created_at=2.0, updated_at=2.0),
    ]
    tokens = strategy.count_tokens(entries)
    assert tokens > 0
    # ~23 chars / 4 = ~5 tokens
    assert tokens == (len("Hello world") + len("How are you?")) // 4

  @pytest.mark.asyncio
  async def test_empty_middle_after_adjustments(self):
    """If tool-call adjustments consume all middle entries, return pin+recent."""
    strategy = SummarizeStrategy(pin_count=2, recent_count=2)
    entries = [
      MemoryEntry(memory_id="m0", session_id="s1", role="user", content="Hi", created_at=0.0, updated_at=0.0),
      MemoryEntry(memory_id="m1", session_id="s1", role="assistant", content="Call", created_at=1.0, updated_at=1.0),
      MemoryEntry(memory_id="m2", session_id="s1", role="tool", content="Result", created_at=2.0, updated_at=2.0),
      MemoryEntry(memory_id="m3", session_id="s1", role="user", content="Ok", created_at=3.0, updated_at=3.0),
      MemoryEntry(memory_id="m4", session_id="s1", role="assistant", content="Done", created_at=4.0, updated_at=4.0),
    ]
    model = _make_mock_model()

    result = await strategy.optimize(entries, model)

    # Middle after pin(2) and recent(2) = entries[2:3] = [tool]
    # Tool gets pulled into pinned → empty middle → no summarization
    model.ainvoke.assert_not_called()
    assert len(result) == 5  # All entries preserved
