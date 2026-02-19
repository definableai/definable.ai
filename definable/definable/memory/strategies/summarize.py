"""Summarize strategy — combines N memories into fewer, comprehensive summaries."""

from typing import Any, List

from definable.memory.strategies.base import MemoryOptimizationStrategy
from definable.memory.types import UserMemory
from definable.utils.log import log_debug


_SUMMARIZE_PROMPT = """\
You are given a list of memories about a user. Consolidate them into a smaller set
of comprehensive, non-redundant memories. Combine related facts, remove duplicates,
and preserve all important information.

Current memories:
{memories}

Return the consolidated memories as a JSON array of objects, each with:
- "memory": the consolidated fact (string)
- "topics": topic tags (array of strings)

Return ONLY the JSON array, no other text."""


class SummarizeStrategy(MemoryOptimizationStrategy):
  """Combines multiple memories into fewer, comprehensive summaries via LLM."""

  async def aoptimize(self, memories: List[UserMemory], model: Any) -> List[UserMemory]:
    """Summarize N memories into fewer comprehensive ones.

    Args:
      memories: The current list of user memories.
      model: The LLM model to use for summarization.

    Returns:
      A list of consolidated UserMemory objects.
    """
    if len(memories) <= 2:
      return memories  # Not worth summarizing

    from definable.model.message import Message

    # Format existing memories
    mem_lines = []
    for i, m in enumerate(memories, 1):
      topics = f" [{', '.join(m.topics)}]" if m.topics else ""
      mem_lines.append(f"  {i}.{topics} {m.memory}")
    memories_text = "\n".join(mem_lines)

    prompt = _SUMMARIZE_PROMPT.format(memories=memories_text)

    messages = [Message(role="user", content=prompt)]
    assistant_message = Message(role="assistant", content="")

    response = await model.ainvoke(messages=messages, assistant_message=assistant_message)
    content = response.content or "[]"

    # Parse the JSON response
    import json

    try:
      # Strip markdown code fences if present
      cleaned = content.strip()
      if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
          cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

      items = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
      log_debug(f"SummarizeStrategy: Failed to parse LLM response: {content[:200]}")
      return memories  # Return originals if parsing fails

    # Build new UserMemory objects
    # Preserve user_id/agent_id from the first original memory
    first = memories[0]
    result: List[UserMemory] = []
    for item in items:
      if isinstance(item, dict) and "memory" in item:
        result.append(
          UserMemory(
            memory=item["memory"],
            topics=item.get("topics", []),
            user_id=first.user_id,
            agent_id=first.agent_id,
          )
        )

    log_debug(f"SummarizeStrategy: {len(memories)} memories -> {len(result)} consolidated")
    return result or memories
