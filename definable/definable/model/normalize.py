"""Cross-provider message normalization.

When a conversation moves between providers, tool-call IDs from one provider
(e.g. Claude's ``toolu_*``, Gemini's free-form IDs) are not always accepted
by another (e.g. OpenAI requires ``call_*``, Mistral requires alphanumeric
length 9). These helpers fix two specific issues:

- ``normalize_tool_messages``: split legacy combined Gemini tool-result messages
  back into canonical per-call messages.
- ``reformat_tool_call_ids``: rewrite IDs to match the target provider's
  prefix and length constraints, keeping assistant tool_calls and tool result
  ``tool_call_id`` references in sync.
"""

from typing import Dict, List, Optional, Union

from definable.model.message import Message

PROVIDER_TOOL_ID_CONFIG: Dict[str, Dict[str, Union[str, int, None]]] = {
  "openai_chat": {
    "prefix": "call_",
    "max_length": 40,
    "call_id_prefix": None,
  },
  "openai_responses": {
    "prefix": "fc_",
    "max_length": None,
    "call_id_prefix": "call_",
  },
  "claude": {
    "prefix": "toolu_",
    "max_length": None,
    "call_id_prefix": None,
  },
  "gemini": {
    "prefix": None,  # Accepts any format
    "max_length": None,
    "call_id_prefix": None,
  },
  "mistral": {
    "prefix": "",  # Alphanumeric only, length 9 — reformat all foreign IDs
    "max_length": 9,
    "call_id_prefix": None,
  },
}


def normalize_tool_messages(messages: List[Message]) -> List[Message]:
  """Split combined Gemini tool-result messages into per-call canonical messages.

  Older Gemini sessions stored all tool results in a single Message with
  ``role="tool"``, ``tool_call_id=None``, ``content=[list]`` and
  ``tool_calls=[{"tool_call_id": ..., "tool_name": ..., "content": ...}, ...]``.
  This function expands that shape into N canonical messages. Messages already
  in canonical form pass through unchanged.
  """
  result: List[Message] = []
  for msg in messages:
    if msg.role == "tool" and msg.tool_call_id is None and msg.tool_calls and isinstance(msg.tool_calls, list):
      content_list = msg.content if isinstance(msg.content, list) else []
      for idx, tc in enumerate(msg.tool_calls):
        if idx < len(content_list):
          tc_content = content_list[idx]
        else:
          tc_content = tc.get("content", "")
        split_msg = Message(
          role="tool",
          tool_call_id=tc.get("tool_call_id"),
          tool_name=tc.get("tool_name"),
          content=tc_content,
        )
        if idx == 0 and msg.metrics is not None:
          split_msg.metrics = msg.metrics
        result.append(split_msg)
    else:
      result.append(msg)
  return result


def reformat_tool_call_ids(messages: List[Message], provider: str) -> List[Message]:
  """Rewrite tool_call IDs to match the target provider's prefix/length rules.

  Builds a foreign-id → new-id map from assistant ``tool_calls``, then applies
  it to both ``tool_calls[].id`` and tool-result ``tool_call_id`` so the
  conversation stays internally consistent. For providers that need a separate
  ``call_id`` (OpenAI Responses), generates one with the matching prefix.

  Returns a new list. Messages that don't need rewriting are returned by
  reference; rewritten ones are pydantic-copied.
  """
  config = PROVIDER_TOOL_ID_CONFIG.get(provider)
  if config is None:
    return messages

  prefix = config.get("prefix")
  if prefix is None:
    return messages

  max_length: Optional[int] = config.get("max_length")  # type: ignore[assignment]
  call_id_prefix: Optional[str] = config.get("call_id_prefix")  # type: ignore[assignment]

  id_map: Dict[str, str] = {}
  call_id_map: Dict[str, str] = {}
  counter = 0
  for msg in messages:
    if msg.role == "assistant" and msg.tool_calls:
      for tc in msg.tool_calls:
        old_id = tc.get("id")
        if not (old_id and isinstance(old_id, str)):
          continue

        needs_reformat = not old_id.startswith(prefix)
        if not needs_reformat and max_length and len(old_id) > max_length:
          needs_reformat = True
        if not needs_reformat and prefix == "" and not old_id.isalnum():
          needs_reformat = True

        if needs_reformat and old_id not in id_map:
          prefix_len = len(prefix) if isinstance(prefix, str) else 0
          id_digits = (max_length - prefix_len) if max_length else 8
          new_id = f"{prefix}{counter:0{id_digits}x}"
          id_map[old_id] = new_id

          existing_call_id = tc.get("call_id")
          if existing_call_id and isinstance(existing_call_id, str) and existing_call_id != old_id:
            id_map[existing_call_id] = new_id

          if call_id_prefix:
            if existing_call_id and isinstance(existing_call_id, str) and existing_call_id.startswith(call_id_prefix):
              call_id_map[old_id] = existing_call_id
            else:
              call_id_map[old_id] = f"{call_id_prefix}{counter:08x}"

          counter += 1

  if not id_map:
    return messages

  result: List[Message] = []
  for msg in messages:
    if msg.role == "assistant" and msg.tool_calls:
      msg_copy = msg.model_copy(deep=True)
      if msg_copy.tool_calls:
        for tc in msg_copy.tool_calls:
          old_id = tc.get("id")
          if old_id and old_id in id_map:
            tc["id"] = id_map[old_id]
            if call_id_prefix:
              tc["call_id"] = call_id_map.get(old_id, id_map[old_id])
      result.append(msg_copy)
    elif msg.role == "tool" and msg.tool_call_id and msg.tool_call_id in id_map:
      msg_copy = msg.model_copy(deep=True)
      msg_copy.tool_call_id = id_map[msg.tool_call_id]
      result.append(msg_copy)
    else:
      result.append(msg)
  return result
