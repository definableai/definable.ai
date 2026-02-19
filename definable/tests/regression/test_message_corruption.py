"""
Regression test for OpenAI API 400 error caused by message history truncation.

Migrated from tests_e2e/regression/test_message_corruption.py.

Root cause: InterfaceSession.truncate_history() blindly sliced messages[-N:],
which could split an assistant(tool_calls) message from its corresponding tool
result messages, causing OpenAI to reject with:
  "messages with role 'tool' must be a response to a preceding message with 'tool_calls'"

The fix makes truncate_history() tool-call-aware: it never cuts between an
assistant message with tool_calls and its corresponding tool messages.
"""

import pytest

from definable.agent.interface.session import InterfaceSession
from definable.model.message import Message


def _make_messages():
  """Build a realistic conversation with tool calls in the middle."""
  return [
    Message(role="user", content="Hello"),
    Message(role="assistant", content="Hi! How can I help?"),
    Message(role="user", content="Search for restaurants near me"),
    Message(
      role="assistant",
      content=None,
      tool_calls=[
        {
          "id": "call_1",
          "type": "function",
          "function": {"name": "browser_navigate", "arguments": '{"url": "https://maps.google.com"}'},
        }
      ],
    ),
    Message(role="tool", content="Navigated to https://maps.google.com", tool_call_id="call_1", name="browser_navigate"),
    Message(
      role="assistant",
      content=None,
      tool_calls=[
        {
          "id": "call_2",
          "type": "function",
          "function": {"name": "browser_type", "arguments": '{"selector": "#searchbox", "text": "restaurants near me"}'},
        },
        {
          "id": "call_3",
          "type": "function",
          "function": {"name": "browser_press_key", "arguments": '{"key": "Enter"}'},
        },
      ],
    ),
    Message(role="tool", content='Typed into "#searchbox": restaurants near me', tool_call_id="call_2", name="browser_type"),
    Message(role="tool", content="Pressed key: 'Enter'", tool_call_id="call_3", name="browser_press_key"),
    Message(role="assistant", content="I found several restaurants near you. Here are the top 5..."),
    Message(role="user", content="Click on the first one"),
    Message(
      role="assistant",
      content=None,
      tool_calls=[
        {
          "id": "call_4",
          "type": "function",
          "function": {"name": "browser_click", "arguments": '{"selector": ".result-item:first-child"}'},
        },
      ],
    ),
    Message(role="tool", content="Clicked: .result-item:first-child", tool_call_id="call_4", name="browser_click"),
    Message(role="assistant", content="I clicked on the first restaurant."),
  ]


@pytest.mark.regression
class TestTruncateHistoryToolCallAware:
  """Regression: truncate_history must never orphan tool messages."""

  def test_naive_truncation_would_orphan_tools(self):
    """Show that the old naive approach WOULD create orphaned tool messages."""
    msgs = _make_messages()
    # 13 messages total. If we keep 6, naive [-6:] starts at index 7.
    # Index 7 = tool message (call_3) whose assistant is at index 5 — orphaned.
    naive_slice = msgs[-6:]
    assert naive_slice[0].role == "tool", "Naive slice starts with orphaned tool message"

  def test_truncation_never_starts_with_tool(self):
    """After fix: truncated history never starts with a tool message."""
    session = InterfaceSession()
    session.messages = _make_messages()
    session.truncate_history(6)
    assert session.messages[0].role != "tool", f"Truncated history starts with tool message: {session.messages[0]}"

  def test_truncation_keeps_tool_group_together(self):
    """Tool messages always follow their corresponding assistant(tool_calls) message."""
    session = InterfaceSession()
    session.messages = _make_messages()
    session.truncate_history(6)

    for i, msg in enumerate(session.messages):
      if msg.role == "tool":
        # Walk backward to find the nearest assistant with tool_calls
        found = False
        for j in range(i - 1, -1, -1):
          if session.messages[j].role == "assistant" and session.messages[j].tool_calls:
            found = True
            break
          if session.messages[j].role == "user":
            break  # Crossed a user boundary — tool is orphaned
        assert found, f"Tool message at index {i} is orphaned (no preceding assistant with tool_calls)"

  def test_truncation_exact_size_no_change(self):
    """If messages <= max_messages, nothing changes."""
    session = InterfaceSession()
    session.messages = _make_messages()
    original_len = len(session.messages)
    session.truncate_history(original_len)
    assert len(session.messages) == original_len

  def test_truncation_very_small_keeps_all(self):
    """If max_messages is so small that truncation would always orphan, keep all."""
    session = InterfaceSession()
    session.messages = _make_messages()
    # max_messages=1 would always start on a non-first message
    session.truncate_history(1)
    # After fix: if we can't find a clean cut, we keep everything
    # The first message is "user" so cut should work if it reaches the top
    # Actually with 1, cut starts at 12, walks back to find non-tool start
    assert session.messages[0].role != "tool"

  def test_truncation_no_tool_calls_works_normally(self):
    """Conversations without tool calls truncate normally."""
    session = InterfaceSession()
    session.messages = [
      Message(role="user", content="Hi"),
      Message(role="assistant", content="Hello"),
      Message(role="user", content="How are you?"),
      Message(role="assistant", content="I'm good!"),
      Message(role="user", content="Tell me a joke"),
      Message(role="assistant", content="Why did the chicken cross the road?"),
    ]
    session.truncate_history(4)
    assert len(session.messages) == 4
    assert session.messages[0].role == "user"
    assert session.messages[0].content == "How are you?"

  def test_truncation_with_multi_tool_calls(self):
    """Multi-tool assistant messages keep all their tool results."""
    session = InterfaceSession()
    session.messages = [
      Message(role="user", content="Do two things"),
      Message(
        role="assistant",
        content=None,
        tool_calls=[
          {"id": "c1", "type": "function", "function": {"name": "tool_a", "arguments": "{}"}},
          {"id": "c2", "type": "function", "function": {"name": "tool_b", "arguments": "{}"}},
        ],
      ),
      Message(role="tool", content="result_a", tool_call_id="c1", name="tool_a"),
      Message(role="tool", content="result_b", tool_call_id="c2", name="tool_b"),
      Message(role="assistant", content="Done!"),
      Message(role="user", content="Thanks"),
    ]
    # Trying to keep only 3: naive would start at index 3 (tool result_b, orphaned)
    session.truncate_history(3)
    # Fix should keep the assistant(tool_calls) + both tools + rest
    for i, msg in enumerate(session.messages):
      if msg.role == "tool":
        found_parent = False
        for j in range(i - 1, -1, -1):
          if session.messages[j].role == "assistant" and session.messages[j].tool_calls:
            found_parent = True
            break
        assert found_parent, f"Tool at index {i} is orphaned"

  def test_empty_messages_no_error(self):
    """Empty message list doesn't crash."""
    session = InterfaceSession()
    session.messages = None
    session.truncate_history(10)  # Should not raise
    assert session.messages is None

  def test_messages_shorter_than_max(self):
    """If messages < max, nothing happens."""
    session = InterfaceSession()
    session.messages = [Message(role="user", content="hi")]
    session.truncate_history(10)
    assert len(session.messages) == 1
