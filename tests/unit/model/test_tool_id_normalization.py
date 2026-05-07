"""Cross-provider tool-id normalization + Gemini combined-tool-msg expansion."""

from definable.model.message import Message
from definable.model.normalize import normalize_tool_messages, reformat_tool_call_ids


# ---------------------------------------------------------------------------
# normalize_tool_messages
# ---------------------------------------------------------------------------


def test_normalize_passthrough_canonical():
  msgs = [
    Message(role="user", content="hi"),
    Message(role="tool", tool_call_id="call_1", tool_name="t", content="result"),
  ]
  out = normalize_tool_messages(msgs)
  assert len(out) == 2
  assert out[1].tool_call_id == "call_1"


def test_normalize_splits_combined_gemini_tool_message():
  combined = Message(
    role="tool",
    tool_call_id=None,
    content=["result_a", "result_b"],
    tool_calls=[
      {"tool_call_id": "id_a", "tool_name": "search", "content": "ignored"},
      {"tool_call_id": "id_b", "tool_name": "calc", "content": "ignored"},
    ],
  )
  out = normalize_tool_messages([combined])
  assert len(out) == 2
  assert out[0].tool_call_id == "id_a"
  assert out[0].tool_name == "search"
  assert out[0].content == "result_a"
  assert out[1].tool_call_id == "id_b"
  assert out[1].content == "result_b"


# ---------------------------------------------------------------------------
# reformat_tool_call_ids
# ---------------------------------------------------------------------------


def test_reformat_unknown_provider_passthrough():
  msgs = [Message(role="user", content="hi")]
  assert reformat_tool_call_ids(msgs, "nonexistent") is msgs


def test_reformat_gemini_no_change():
  # Gemini accepts any format → prefix is None → passthrough
  msgs = [
    Message(role="assistant", tool_calls=[{"id": "weirdformat-123", "function": {"name": "t"}}]),
  ]
  out = reformat_tool_call_ids(msgs, "gemini")
  assert out[0].tool_calls is not None
  assert out[0].tool_calls[0]["id"] == "weirdformat-123"


def test_reformat_claude_to_openai_chat():
  msgs = [
    Message(role="user", content="hi"),
    Message(role="assistant", tool_calls=[{"id": "toolu_abc", "function": {"name": "search"}}]),
    Message(role="tool", tool_call_id="toolu_abc", content="result"),
  ]
  out = reformat_tool_call_ids(msgs, "openai_chat")
  assert out[1].tool_calls is not None
  new_id = out[1].tool_calls[0]["id"]
  assert new_id.startswith("call_")
  assert out[2].tool_call_id == new_id


def test_reformat_keeps_already_compliant_ids():
  msgs = [
    Message(role="assistant", tool_calls=[{"id": "call_already_ok", "function": {"name": "t"}}]),
  ]
  out = reformat_tool_call_ids(msgs, "openai_chat")
  assert out is msgs  # no remap needed


def test_reformat_mistral_alphanumeric_length_9():
  msgs = [
    Message(role="user", content="hi"),
    Message(role="assistant", tool_calls=[{"id": "toolu_abc-xyz", "function": {"name": "t"}}]),
    Message(role="tool", tool_call_id="toolu_abc-xyz", content="r"),
  ]
  out = reformat_tool_call_ids(msgs, "mistral")
  assert out[1].tool_calls is not None
  new_id = out[1].tool_calls[0]["id"]
  assert len(new_id) == 9
  assert new_id.isalnum()
  assert out[2].tool_call_id == new_id
