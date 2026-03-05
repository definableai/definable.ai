"""User model — structured profile built from behavioral observations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from definable.memory.cortex.learning.traits import Trait, TraitCategory


@dataclass
class UserModel:
  """A structured model of a user's traits, preferences, and patterns.

  Built incrementally from behavioral observations. Can generate
  a natural-language style guide for system prompt injection.
  """

  user_id: str = "default"
  traits: List[Trait] = field(default_factory=list)
  predictions: List[Dict[str, Any]] = field(default_factory=list)
  created_at: float = 0.0
  updated_at: float = 0.0

  def __post_init__(self) -> None:
    now = time.time()
    if self.created_at == 0.0:
      self.created_at = now
    if self.updated_at == 0.0:
      self.updated_at = now

  def add_trait(self, trait: Trait) -> None:
    """Add a new trait to the model."""
    self.traits.append(trait)
    self.updated_at = time.time()

  def get_trait(self, name: str) -> Optional[Trait]:
    """Find a trait by name."""
    for t in self.traits:
      if t.name == name:
        return t
    return None

  def get_traits_by_category(self, category: TraitCategory) -> List[Trait]:
    """Get all traits in a category."""
    return [t for t in self.traits if t.category == category]

  def get_strong_traits(self, threshold: float = 0.7) -> List[Trait]:
    """Get traits above a confidence threshold."""
    return [t for t in self.traits if t.confidence >= threshold]

  @property
  def trait_count(self) -> int:
    return len(self.traits)

  @property
  def strong_trait_count(self) -> int:
    return len(self.get_strong_traits())

  def generate_style_guide(self) -> str:
    """Generate a natural-language description of the user.

    Produces a prose summary organized by category, suitable for
    injection into a system prompt.
    """
    strong = self.get_strong_traits(threshold=0.5)
    if not strong:
      return "No strong user traits have been identified yet."

    parts: list[str] = [f"User Profile (based on {len(self.traits)} observed traits):\n"]

    by_category: Dict[TraitCategory, List[Trait]] = {}
    for trait in sorted(strong, key=lambda t: t.confidence, reverse=True):
      by_category.setdefault(trait.category, []).append(trait)

    category_labels = {
      TraitCategory.COMMUNICATION: "Communication Style",
      TraitCategory.TECHNICAL: "Technical Preferences",
      TraitCategory.DECISION: "Decision Making",
      TraitCategory.VALUES: "Core Values",
      TraitCategory.EMOTIONAL: "Emotional Patterns",
      TraitCategory.WORKFLOW: "Working Style",
    }

    for category, traits in by_category.items():
      label = category_labels.get(category, category.value.title())
      parts.append(f"\n{label}:")
      for trait in traits:
        confidence_label = "very likely" if trait.confidence > 0.8 else "likely" if trait.confidence > 0.6 else "possible"
        parts.append(f"  - {trait.description} ({confidence_label}, {trait.observation_count} observations)")

    return "\n".join(parts)

  def predict(self, scenario: str) -> Optional[str]:
    """Predict how the user would respond to a scenario.

    Uses strong traits to generate a prediction. Returns None if
    insufficient trait data.
    """
    strong = self.get_strong_traits()
    if not strong:
      return None

    # Build prediction from strongest traits
    relevant = []
    scenario_lower = scenario.lower()
    for trait in strong:
      # Simple keyword matching for relevance
      if any(word in scenario_lower for word in trait.name.lower().split()):
        relevant.append(trait)

    if not relevant:
      relevant = strong[:3]  # Fall back to top traits

    prediction_parts = ["Based on observed traits, the user would likely:"]
    for trait in relevant[:5]:
      prediction_parts.append(f"- {trait.description}")

    prediction = "\n".join(prediction_parts)
    self.predictions.append({
      "scenario": scenario,
      "prediction": prediction,
      "traits_used": [t.trait_id for t in relevant[:5]],
      "timestamp": time.time(),
    })
    return prediction

  def to_dict(self) -> Dict[str, Any]:
    return {
      "user_id": self.user_id,
      "traits": [t.to_dict() for t in self.traits],
      "predictions": self.predictions,
      "created_at": self.created_at,
      "updated_at": self.updated_at,
    }

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "UserModel":
    return cls(
      user_id=data.get("user_id", "default"),
      traits=[Trait.from_dict(t) for t in data.get("traits", [])],
      predictions=data.get("predictions", []),
      created_at=data.get("created_at", 0.0),
      updated_at=data.get("updated_at", 0.0),
    )
