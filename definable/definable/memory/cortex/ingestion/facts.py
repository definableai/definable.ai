"""Fact extractor — LLM-based atomic fact extraction."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, List, Optional

from definable.memory.cortex.record.types import Fact
from definable.utils.log import log_warning

if TYPE_CHECKING:
  from definable.model.base import Model


FACTS_PROMPT = """Extract atomic, self-contained facts from this conversation turn.
Each fact should:
- Be a complete statement with all pronouns and references resolved
- Be independently understandable without context
- Capture a single piece of information

Return a JSON array of objects, each with:
- "content": The self-contained factual statement
- "confidence": How certain this fact is (0.0-1.0)
- "entities": List of named entities in the fact

Conversation turn:
{text}

Respond with ONLY the JSON array, no markdown fences. If no facts can be extracted, return []."""


class FactExtractor:
  """Extracts atomic facts from conversation turns using an LLM."""

  def __init__(self, model: Optional["Model"] = None):
    self.model = model

  async def extract(self, text: str, turn_index: int = 0) -> List[Fact]:
    """Extract atomic facts from a conversation turn.

    Returns empty list if no model or extraction fails.
    """
    if self.model is None:
      return []

    from definable.model.message import Message

    prompt = FACTS_PROMPT.format(text=text)
    try:
      response = await self.model.ainvoke(
        messages=[Message(role="user", content=prompt)],
        assistant_message=Message(role="assistant", content=""),
      )
      content = response.content if isinstance(response.content, str) else str(response.content or "")
      return self._parse_response(content, turn_index)
    except Exception as exc:
      log_warning(f"FactExtractor: extraction failed: {exc}")
      return []

  def _parse_response(self, text: str, turn_index: int) -> List[Fact]:
    """Parse LLM JSON response into list of Facts."""
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
      facts = []
      for item in data:
        facts.append(
          Fact(
            content=item.get("content", ""),
            confidence=item.get("confidence", 1.0),
            entities=item.get("entities", []),
            source_turns=[turn_index],
          )
        )
      return facts
    except (json.JSONDecodeError, KeyError) as exc:
      log_warning(f"FactExtractor: parse failed: {exc}")
      return []
