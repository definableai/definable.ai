"""Message history trimming with tool-call pair protection.

Provides three trimming strategies (tail, head_and_tail, summarize)
that operate on message *groups* — an assistant message with tool_calls
is always kept together with its corresponding tool result messages.
"""

from dataclasses import dataclass
from typing import Any, List, Literal, Optional

from definable.model.message import Message


@dataclass(frozen=True)
class MessageGroup:
  """An atomic group of messages that must never be split.

  Either a single standalone message, or an assistant message with
  tool_calls followed by its tool result messages.
  """

  messages: tuple  # tuple[Message, ...]
  start_idx: int  # original index of first message in the flat list

  @property
  def size(self) -> int:
    return len(self.messages)


def group_messages(messages: List[Message]) -> List[MessageGroup]:
  """Group messages into atomic units.

  An assistant message with tool_calls is grouped with all immediately
  following tool-role messages. Everything else is a group of 1.

  Args:
    messages: Flat list of messages.

  Returns:
    List of MessageGroup, preserving original order.
  """
  groups: List[MessageGroup] = []
  i = 0
  n = len(messages)

  while i < n:
    msg = messages[i]

    # Check if this is an assistant message with tool calls
    if msg.role == "assistant" and msg.tool_calls:
      group_msgs = [msg]
      j = i + 1
      # Collect all immediately following tool-role messages
      while j < n and messages[j].role == "tool":
        group_msgs.append(messages[j])
        j += 1
      groups.append(MessageGroup(messages=tuple(group_msgs), start_idx=i))
      i = j
    else:
      groups.append(MessageGroup(messages=(msg,), start_idx=i))
      i += 1

  return groups


def flatten_groups(groups: List[MessageGroup]) -> List[Message]:
  """Flatten message groups back into a flat message list."""
  result: List[Message] = []
  for group in groups:
    result.extend(group.messages)
  return result


def trim_tail(
  messages: List[Message],
  max_messages: int,
) -> List[Message]:
  """Keep the most recent messages, respecting tool-call pairs.

  Groups messages into atomic units, then keeps groups from the
  end until the total message count reaches max_messages. Never
  splits a tool-call group.

  Args:
    messages: Conversation messages (excluding system).
    max_messages: Target maximum message count.

  Returns:
    Trimmed message list.
  """
  if len(messages) <= max_messages:
    return messages

  groups = group_messages(messages)

  # Walk backwards, accumulating groups until we hit the budget
  kept: List[MessageGroup] = []
  total = 0
  for group in reversed(groups):
    if total + group.size > max_messages and kept:
      break
    kept.append(group)
    total += group.size

  kept.reverse()
  return flatten_groups(kept)


def trim_head_and_tail(
  messages: List[Message],
  keep_first: int,
  keep_last: int,
) -> List[Message]:
  """Keep the first N and last M messages, dropping the middle.

  Respects tool-call pairs — groups at the boundary of the
  head/tail sections are kept intact.

  Args:
    messages: Conversation messages (excluding system).
    keep_first: Number of messages to keep from the start.
    keep_last: Number of messages to keep from the end.

  Returns:
    Trimmed message list (head + tail, no overlap).
  """
  if len(messages) <= keep_first + keep_last:
    return messages

  groups = group_messages(messages)

  # Collect head groups
  head: List[MessageGroup] = []
  head_count = 0
  for group in groups:
    if head_count + group.size > keep_first and head:
      break
    head.append(group)
    head_count += group.size

  # Collect tail groups (from the end)
  tail: List[MessageGroup] = []
  tail_count = 0
  for group in reversed(groups):
    # Don't overlap with head
    if group in head:
      break
    if tail_count + group.size > keep_last and tail:
      break
    tail.append(group)
    tail_count += group.size
  tail.reverse()

  return flatten_groups(head + tail)


class HistoryTrimmer:
  """Trims message history using a configurable strategy.

  All strategies respect tool-call pair integrity — an assistant
  message with tool_calls is never separated from its tool result messages.

  The ``summarize`` strategy uses an LLM to distill dropped messages
  into a compact summary that is injected as the first message.

  Example:
    trimmer = HistoryTrimmer(strategy="tail", max_messages=50)
    trimmed = trimmer.trim(messages)

    # With summarization (requires async):
    trimmer = HistoryTrimmer(strategy="summarize", max_messages=10, model=model)
    trimmed = await trimmer.atrim(messages)
  """

  def __init__(
    self,
    strategy: Literal["none", "tail", "head_and_tail", "summarize"] = "tail",
    max_messages: Optional[int] = 50,
    keep_first: int = 4,
    model: Optional[Any] = None,
  ) -> None:
    self._strategy = strategy
    self._max_messages = max_messages
    self._keep_first = keep_first
    self._model = model  # For summarize strategy

  @property
  def strategy(self) -> str:
    return self._strategy

  def trim(self, messages: List[Message]) -> List[Message]:
    """Trim messages according to the configured strategy (sync).

    For the ``summarize`` strategy, use ``atrim()`` instead.
    Sync ``trim()`` with ``summarize`` falls back to tail trimming.

    Args:
      messages: Conversation messages (should NOT include system messages).

    Returns:
      Trimmed message list.
    """
    if self._strategy == "none" or self._max_messages is None:
      return messages

    if self._strategy == "tail":
      return trim_tail(messages, self._max_messages)

    if self._strategy == "head_and_tail":
      return trim_head_and_tail(messages, self._keep_first, self._max_messages)

    # "summarize" sync fallback: trim without summarization.
    # Use atrim() for actual LLM summarization.
    return trim_tail(messages, self._max_messages)

  async def atrim(self, messages: List[Message]) -> List[Message]:
    """Trim messages according to the configured strategy (async).

    For the ``summarize`` strategy, dropped messages are summarized
    by an LLM and the summary is injected as the first message.

    Args:
      messages: Conversation messages (should NOT include system messages).

    Returns:
      Trimmed message list (with summary injected for ``summarize`` strategy).
    """
    if self._strategy == "none" or self._max_messages is None:
      return messages

    if self._strategy == "tail":
      return trim_tail(messages, self._max_messages)

    if self._strategy == "head_and_tail":
      return trim_head_and_tail(messages, self._keep_first, self._max_messages)

    # "summarize" or any other — summarize with LLM
    return await self._trim_with_summary(messages)

  async def _trim_with_summary(self, messages: List[Message]) -> List[Message]:
    """Trim using tail strategy, but summarize the dropped messages first."""
    if self._max_messages is None or len(messages) <= self._max_messages:
      return messages

    # Split into what to drop and what to keep
    groups = group_messages(messages)
    kept: List[MessageGroup] = []
    total = 0
    for group in reversed(groups):
      if total + group.size > self._max_messages and kept:
        break
      kept.append(group)
      total += group.size
    kept.reverse()

    # Determine which messages will be dropped
    kept_flat = flatten_groups(kept)
    kept_set = set(id(m) for m in kept_flat)
    dropped = [m for m in messages if id(m) not in kept_set]

    if not dropped:
      return messages

    # Summarize dropped messages
    summary_text = ""
    if self._model is not None and dropped:
      from definable.agent.context.summarizer import summarize_messages

      summary_text = await summarize_messages(dropped, self._model)

    if not summary_text:
      # Fallback: simple extraction
      from definable.agent.context.summarizer import _fallback_summary

      summary_text = _fallback_summary(dropped)

    # Inject summary as the first message
    if summary_text:
      from definable.agent.context.summarizer import make_summary_message

      summary_msg = make_summary_message(summary_text)
      return [summary_msg] + kept_flat

    return kept_flat
