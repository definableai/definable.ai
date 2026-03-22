"""LLM-based conversation history summarization.

When the history trimmer uses the ``summarize`` strategy, messages
that would be dropped are first summarized into a compact paragraph.
The summary is injected as the first user message in the trimmed
history, preserving key facts, decisions, and context.
"""

from textwrap import dedent
from typing import TYPE_CHECKING, List

from definable.model.message import Message
from definable.utils.log import log_error

if TYPE_CHECKING:
  from definable.model.base import Model

SUMMARIZE_SYSTEM_PROMPT = dedent("""\
  You are summarizing a conversation between a user and an AI assistant.
  Your summary will replace the original messages in the conversation history,
  so it MUST preserve all important information.

  ALWAYS PRESERVE:
  - User's name, identity, role, preferences, and personal details
  - Specific facts: numbers, dates, locations, names of people/pets/projects
  - Decisions made, agreements reached, tasks assigned
  - Key questions asked and answers given
  - Any constraints, requirements, or preferences stated
  - Technical details: tools used, code discussed, configurations
  - Emotional context if relevant (frustration, excitement, urgency)

  FORMAT:
  Write a concise but complete summary in 2-4 paragraphs. Use bullet points
  for lists of facts. Start with the most important information.
  Do NOT use phrases like "In summary" or "The conversation covered" —
  write as if noting facts for someone who needs to continue this conversation.
""")


async def summarize_messages(
  messages: List[Message],
  model: "Model",
  *,
  max_summary_tokens: int = 500,
) -> str:
  """Summarize a list of messages into a compact text block.

  Uses an LLM to distill the conversation into key facts and context.

  Args:
    messages: Messages to summarize.
    model: Model to use for summarization.
    max_summary_tokens: Approximate max tokens for the summary.

  Returns:
    Summary text, or a simple fallback if summarization fails.
  """
  if not messages:
    return ""

  # Format messages for the summarizer
  conversation_text = _format_messages_for_summary(messages)
  if not conversation_text.strip():
    return ""

  user_prompt = (
    f"Summarize this conversation in {max_summary_tokens} tokens or fewer. "
    f"Preserve ALL specific facts (names, numbers, locations, preferences).\n\n"
    f"Conversation:\n{conversation_text}"
  )

  try:
    response = await model.aresponse(
      messages=[
        Message(role="system", content=SUMMARIZE_SYSTEM_PROMPT),
        Message(role="user", content=user_prompt),
      ]
    )
    summary = response.content or ""
    if summary.strip():
      return summary.strip()
  except Exception as e:
    log_error(f"Summarization failed: {e}")

  # Fallback: simple extraction of user messages
  return _fallback_summary(messages)


def make_summary_message(summary: str) -> Message:
  """Create a synthetic user message containing the conversation summary.

  This message is inserted at the start of the trimmed history so the
  model has context about what happened earlier.

  Args:
    summary: The summary text.

  Returns:
    A Message with role="user" containing the summary.
  """
  return Message(
    role="user",
    content=f"[Previous conversation summary]\n{summary}\n[End of summary — conversation continues below]",
  )


def _format_messages_for_summary(messages: List[Message]) -> str:
  """Format messages into a readable conversation transcript."""
  lines: list[str] = []
  for msg in messages:
    role = msg.role
    content = msg.content or ""
    if isinstance(content, list):
      # Multimodal content — extract text parts
      text_parts = [block.get("text", "") for block in content if isinstance(block, dict) and "text" in block]
      content = " ".join(text_parts)
    if not content or not str(content).strip():
      continue
    # Truncate very long messages to keep the summarizer input reasonable
    content_str = str(content).strip()
    if len(content_str) > 500:
      content_str = content_str[:500] + "..."
    lines.append(f"{role}: {content_str}")
  return "\n".join(lines)


def _fallback_summary(messages: List[Message]) -> str:
  """Simple fallback: extract key user statements."""
  facts: list[str] = []
  for msg in messages:
    if msg.role == "user" and msg.content:
      content = str(msg.content).strip()
      if content and len(content) < 200:
        facts.append(f"- {content}")
  if facts:
    return "Key points from earlier conversation:\n" + "\n".join(facts[:10])
  return ""
