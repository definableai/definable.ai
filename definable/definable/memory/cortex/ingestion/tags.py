"""Tag generator — LLM-based hierarchical tag extraction."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, List, Optional

from definable.utils.log import log_warning

if TYPE_CHECKING:
  from definable.model.base import Model


TAGS_PROMPT = """Generate hierarchical tags for this conversation turn.
Tags should use "/" as separator to form a hierarchy (e.g., "work/project/deadlines").

Categories to consider:
- Domain (e.g., "technical/python", "personal/health")
- Activity (e.g., "activity/debugging", "activity/planning")
- Emotion (e.g., "emotion/frustration", "emotion/satisfaction")
- Topic specifics (e.g., "topic/memory-systems", "topic/testing")

Return a JSON array of tag strings. Include 3-8 tags. Use lowercase.

Conversation turn:
{text}

Respond with ONLY the JSON array, no markdown fences."""


class TagGenerator:
  """Generates hierarchical tags from conversation turns using an LLM."""

  def __init__(self, model: Optional["Model"] = None):
    self.model = model

  async def generate(self, text: str) -> List[str]:
    """Generate hierarchical tags for a conversation turn.

    Returns empty list if no model or generation fails.
    """
    if self.model is None:
      return []

    from definable.model.message import Message

    prompt = TAGS_PROMPT.format(text=text)
    try:
      response = await self.model.ainvoke(
        messages=[Message(role="user", content=prompt)],
        assistant_message=Message(role="assistant", content=""),
      )
      content = response.content if isinstance(response.content, str) else str(response.content or "")
      return self._parse_response(content)
    except Exception as exc:
      log_warning(f"TagGenerator: generation failed: {exc}")
      return []

  def _parse_response(self, text: str) -> List[str]:
    """Parse LLM JSON response into list of tag strings."""
    try:
      cleaned = text.strip()
      if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
          cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
      data = json.loads(cleaned)
      if not isinstance(data, list):
        return []
      return [str(tag).lower().strip() for tag in data if isinstance(tag, str)]
    except (json.JSONDecodeError, KeyError) as exc:
      log_warning(f"TagGenerator: parse failed: {exc}")
      return []
