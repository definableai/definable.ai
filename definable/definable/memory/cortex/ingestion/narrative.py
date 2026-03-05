"""Narrative builder — LLM-based narrative episode extraction."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

from definable.memory.cortex.record.types import NarrativeEpisode
from definable.utils.log import log_warning

if TYPE_CHECKING:
  from definable.model.base import Model


NARRATIVE_PROMPT = """Analyze this conversation turn and create a narrative episode.
Return a JSON object with these fields:
- "content": A 1-3 sentence narrative summary capturing what happened, the context, and emotional undercurrents.
- "participants": List of participant names/roles mentioned (e.g., ["user", "assistant"]).
- "emotional_tone": One word describing the emotional tone (e.g., "frustrated", "curious", "satisfied", "neutral").
- "causal_chain": List of cause-effect links in order (e.g., ["asked about X", "discovered Y", "decided Z"]).

Conversation turn:
{text}

Respond with ONLY the JSON object, no markdown fences."""


class NarrativeBuilder:
  """Builds narrative episodes from conversation turns using an LLM."""

  def __init__(self, model: Optional["Model"] = None):
    self.model = model

  async def build(self, text: str, role: str = "user") -> Optional[NarrativeEpisode]:
    """Extract a narrative episode from a conversation turn.

    Returns None if no model is available or extraction fails.
    """
    if self.model is None:
      return None

    from definable.model.message import Message

    prompt = NARRATIVE_PROMPT.format(text=text)
    try:
      response = await self.model.ainvoke(
        messages=[Message(role="user", content=prompt)],
        assistant_message=Message(role="assistant", content=""),
      )
      content = response.content if isinstance(response.content, str) else str(response.content or "")
      return self._parse_response(content)
    except Exception as exc:
      log_warning(f"NarrativeBuilder: extraction failed: {exc}")
      return None

  def _parse_response(self, text: str) -> Optional[NarrativeEpisode]:
    """Parse LLM JSON response into a NarrativeEpisode."""
    try:
      # Strip markdown fences if present
      cleaned = text.strip()
      if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
          cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
      data = json.loads(cleaned)
      return NarrativeEpisode(
        content=data.get("content", ""),
        participants=data.get("participants", []),
        emotional_tone=data.get("emotional_tone"),
        causal_chain=data.get("causal_chain", []),
      )
    except (json.JSONDecodeError, KeyError) as exc:
      log_warning(f"NarrativeBuilder: parse failed: {exc}")
      return None
