"""Trait types for behavioral learning."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class TraitCategory(str, Enum):
  """Categories of user traits."""

  COMMUNICATION = "communication"  # How the user communicates
  TECHNICAL = "technical"  # Technical preferences and skills
  DECISION = "decision"  # How the user makes decisions
  VALUES = "values"  # Core values and principles
  EMOTIONAL = "emotional"  # Emotional patterns
  WORKFLOW = "workflow"  # Working style and process


@dataclass
class Observation:
  """A single behavioral observation from an interaction.

  Observations are the raw signals extracted by the BehavioralObserver.
  Multiple observations of the same pattern increase trait confidence.
  """

  observation_id: str = ""
  content: str = ""
  category: TraitCategory = TraitCategory.COMMUNICATION
  signal_strength: float = 0.5  # How strong this signal is (0-1)
  source_record_id: str = ""
  created_at: float = 0.0

  def __post_init__(self) -> None:
    if not self.observation_id:
      from uuid import uuid4

      self.observation_id = str(uuid4())
    if self.created_at == 0.0:
      self.created_at = time.time()

  def to_dict(self) -> Dict[str, Any]:
    return {
      "observation_id": self.observation_id,
      "content": self.content,
      "category": self.category.value,
      "signal_strength": self.signal_strength,
      "source_record_id": self.source_record_id,
      "created_at": self.created_at,
    }

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "Observation":
    return cls(
      observation_id=data.get("observation_id", ""),
      content=data.get("content", ""),
      category=TraitCategory(data.get("category", "communication")),
      signal_strength=data.get("signal_strength", 0.5),
      source_record_id=data.get("source_record_id", ""),
      created_at=data.get("created_at", 0.0),
    )


@dataclass
class Trait:
  """A learned user trait with confidence tracking.

  Traits start at low confidence and grow with reinforcing observations.
  Contradictions decay confidence multiplicatively.
  """

  trait_id: str = ""
  name: str = ""
  description: str = ""
  category: TraitCategory = TraitCategory.COMMUNICATION
  confidence: float = 0.3  # Start low, grow with evidence
  observations: List[Observation] = field(default_factory=list)
  created_at: float = 0.0
  updated_at: float = 0.0

  def __post_init__(self) -> None:
    if not self.trait_id:
      from uuid import uuid4

      self.trait_id = str(uuid4())
    now = time.time()
    if self.created_at == 0.0:
      self.created_at = now
    if self.updated_at == 0.0:
      self.updated_at = now

  def reinforce(self, observation: Observation, boost: float = 0.15) -> None:
    """Add a reinforcing observation, increasing confidence."""
    self.observations.append(observation)
    self.confidence = min(1.0, self.confidence + boost * observation.signal_strength)
    self.updated_at = time.time()

  def contradict(self, decay: float = 0.85) -> None:
    """Record a contradiction, decaying confidence multiplicatively."""
    self.confidence *= decay
    self.updated_at = time.time()

  @property
  def observation_count(self) -> int:
    return len(self.observations)

  @property
  def is_strong(self) -> bool:
    """Whether this trait has enough confidence to be considered reliable."""
    return self.confidence >= 0.7

  def to_dict(self) -> Dict[str, Any]:
    return {
      "trait_id": self.trait_id,
      "name": self.name,
      "description": self.description,
      "category": self.category.value,
      "confidence": self.confidence,
      "observations": [o.to_dict() for o in self.observations],
      "created_at": self.created_at,
      "updated_at": self.updated_at,
    }

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "Trait":
    return cls(
      trait_id=data.get("trait_id", ""),
      name=data.get("name", ""),
      description=data.get("description", ""),
      category=TraitCategory(data.get("category", "communication")),
      confidence=data.get("confidence", 0.3),
      observations=[Observation.from_dict(o) for o in data.get("observations", [])],
      created_at=data.get("created_at", 0.0),
      updated_at=data.get("updated_at", 0.0),
    )
