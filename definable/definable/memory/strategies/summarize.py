"""SummarizeStrategy — pin + summarize-middle + keep-recent.

Hybrid strategy that preserves the first N and last M messages,
summarizing everything in between via an LLM call. Tool-call-aware:
tool result entries at boundaries are pulled into adjacent sections
to avoid orphaned tool calls.
"""

from typing import Any, List

from definable.memory.strategies.base import MemoryStrategy
from definable.memory.types import MemoryEntry
from definable.utils.log import log_debug, log_warning


_SUMMARIZE_PROMPT = """\
You are a conversation summarizer. Given a section of conversation history,
produce a concise summary that captures the key information, decisions, and context.

{prior_context}Conversation to summarize:
{conversation}

Write a brief, information-dense summary paragraph. Focus on facts, decisions,
and important context. Do NOT include greetings or filler."""


class SummarizeStrategy(MemoryStrategy):
  """Hybrid strategy: pin first N + summarize middle + keep recent M.

  Tool-call-aware: tool result entries at boundaries are pulled into
  the adjacent section to avoid orphaned tool calls.
  """

  def __init__(self, pin_count: int = 2, recent_count: int = 3) -> None:
    self.pin_count = pin_count
    self.recent_count = recent_count

  async def optimize(self, entries: List[MemoryEntry], model: Any) -> List[MemoryEntry]:
    """Optimize entries by summarizing the middle section.

    Args:
      entries: All session entries, ordered by created_at.
      model: LLM model for summarization.

    Returns:
      Optimized list: pinned + summary + recent.
    """
    if len(entries) <= self.pin_count + self.recent_count:
      return entries

    pinned = list(entries[: self.pin_count])
    recent = list(entries[-self.recent_count :])
    middle = list(entries[self.pin_count : -self.recent_count])

    # Tool call boundary: pull tool entries from start of middle into pinned
    while middle and middle[0].role == "tool":
      pinned.append(middle.pop(0))

    # Tool call boundary: pull preceding entries when recent starts with tool
    while recent and recent[0].role == "tool" and middle:
      recent.insert(0, middle.pop(-1))

    if not middle:
      return entries  # Nothing to summarize after adjustments

    # Build prior context from pinned summaries
    prior_parts = []
    for e in pinned:
      if e.role == "summary":
        prior_parts.append(f"Previous summary: {e.content}")
    prior_context = "\n".join(prior_parts) if prior_parts else ""
    if prior_context:
      prior_context = f"Prior context:\n{prior_context}\n\n"

    # Build conversation text from middle entries
    conv_lines = []
    for e in middle:
      conv_lines.append(f"{e.role}: {e.content}")
    conversation = "\n".join(conv_lines)

    prompt = _SUMMARIZE_PROMPT.format(prior_context=prior_context, conversation=conversation)

    # Call LLM for summarization
    try:
      from definable.model.message import Message

      messages = [Message(role="user", content=prompt)]
      assistant_message = Message(role="assistant", content="")
      response = await model.ainvoke(messages=messages, assistant_message=assistant_message)
      summary_text = response.content or f"Summary of {len(middle)} messages."
    except Exception as exc:
      log_warning(f"SummarizeStrategy: LLM call failed: {exc}")
      summary_text = f"Summary of {len(middle)} messages."

    # Create summary entry inheriting metadata from first middle entry
    first_middle = middle[0]
    summary = MemoryEntry(
      session_id=first_middle.session_id,
      user_id=first_middle.user_id,
      role="summary",
      content=summary_text,
      created_at=first_middle.created_at,
      updated_at=first_middle.updated_at,
    )

    log_debug(f"SummarizeStrategy: {len(entries)} entries -> {len(pinned) + 1 + len(recent)} (pin={len(pinned)}, recent={len(recent)})")
    return pinned + [summary] + recent
