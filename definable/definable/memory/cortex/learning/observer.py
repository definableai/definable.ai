"""Behavioral observer — extracts signals from interactions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, List, Optional

from definable.memory.cortex.learning.traits import Observation, TraitCategory
from definable.utils.log import log_warning

if TYPE_CHECKING:
  from definable.model.base import Model


OBSERVER_PROMPT = """Analyze this interaction and identify behavioral signals about the user.
Look for:
- Communication style (direct/verbose, formal/casual, emoji usage)
- Technical preferences (tools, languages, patterns, standards)
- Decision-making patterns (evidence-based, gut feeling, cautious, bold)
- Values (quality, speed, thoroughness, simplicity)
- Emotional patterns (frustration triggers, satisfaction signals)
- Workflow preferences (planning style, testing approach, review process)

Return a JSON array of observations, each with:
- "content": What was observed (specific behavioral signal)
- "category": One of: communication, technical, decision, values, emotional, workflow
- "signal_strength": How strong this signal is (0.0-1.0, where 1.0 = very clear signal)

Interaction:
{text}

Role: {role}

Respond with ONLY the JSON array. If no clear signals, return []."""


class BehavioralObserver:
  """Watches interactions and extracts behavioral signals.

  Can operate in two modes:
  1. LLM-based: Uses a model to extract nuanced signals
  2. Rule-based: Uses regex/keyword patterns for basic signals
  """

  def __init__(self, model: Optional["Model"] = None):
    self.model = model

  async def observe(self, text: str, role: str = "user", record_id: str = "") -> List[Observation]:
    """Extract behavioral observations from an interaction.

    Args:
      text: The interaction content.
      role: The role of the speaker.
      record_id: Source record ID for traceability.

    Returns:
      List of Observation objects.
    """
    # Only observe user messages (not assistant)
    if role != "user":
      return []

    if self.model:
      return await self._observe_llm(text, role, record_id)
    return self._observe_rules(text, record_id)

  async def _observe_llm(self, text: str, role: str, record_id: str) -> List[Observation]:
    """LLM-based observation extraction."""
    from definable.model.message import Message

    prompt = OBSERVER_PROMPT.format(text=text, role=role)
    try:
      response = await self.model.ainvoke(  # type: ignore[union-attr]
        messages=[Message(role="user", content=prompt)],
        assistant_message=Message(role="assistant", content=""),
      )
      content = response.content if isinstance(response.content, str) else str(response.content or "")
      return self._parse_response(content, record_id)
    except Exception as exc:
      log_warning(f"BehavioralObserver: LLM extraction failed: {exc}")
      return self._observe_rules(text, record_id)

  def _observe_rules(self, text: str, record_id: str) -> List[Observation]:
    """Rule-based observation extraction (fallback)."""
    observations: List[Observation] = []
    text_lower = text.lower()

    # Directness signals
    if any(w in text_lower for w in ["don't", "never", "always", "must", "stop"]):
      observations.append(
        Observation(
          content="Uses direct/imperative language",
          category=TraitCategory.COMMUNICATION,
          signal_strength=0.4,
          source_record_id=record_id,
        )
      )

    # Technical preference signals
    if any(w in text_lower for w in ["test", "testing", "pytest", "unit test"]):
      observations.append(
        Observation(
          content="Values testing and test-driven development",
          category=TraitCategory.TECHNICAL,
          signal_strength=0.5,
          source_record_id=record_id,
        )
      )

    # Quality signals
    if any(w in text_lower for w in ["quality", "thorough", "careful", "verify", "validate"]):
      observations.append(
        Observation(
          content="Values thoroughness and quality over speed",
          category=TraitCategory.VALUES,
          signal_strength=0.5,
          source_record_id=record_id,
        )
      )

    # Frustration signals
    if any(w in text_lower for w in ["frustrated", "annoyed", "waste", "wrong", "broken"]):
      observations.append(
        Observation(
          content="Expresses frustration with quality issues",
          category=TraitCategory.EMOTIONAL,
          signal_strength=0.6,
          source_record_id=record_id,
        )
      )

    # Workflow signals
    if any(w in text_lower for w in ["plan", "phase", "step", "incremental", "small"]):
      observations.append(
        Observation(
          content="Prefers incremental, phased approach",
          category=TraitCategory.WORKFLOW,
          signal_strength=0.4,
          source_record_id=record_id,
        )
      )

    return observations

  def _parse_response(self, text: str, record_id: str) -> List[Observation]:
    """Parse LLM JSON response into Observations."""
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
      observations = []
      for item in data:
        try:
          cat = TraitCategory(item.get("category", "communication"))
        except ValueError:
          cat = TraitCategory.COMMUNICATION
        observations.append(
          Observation(
            content=item.get("content", ""),
            category=cat,
            signal_strength=item.get("signal_strength", 0.5),
            source_record_id=record_id,
          )
        )
      return observations
    except (json.JSONDecodeError, KeyError) as exc:
      log_warning(f"BehavioralObserver: parse failed: {exc}")
      return []
