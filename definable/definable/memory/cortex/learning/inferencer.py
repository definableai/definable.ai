"""Trait inferencer — converts observations into trait updates."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from definable.memory.cortex.learning.traits import Observation, Trait
from definable.memory.cortex.learning.user_model import UserModel
from definable.utils.log import log_debug

if TYPE_CHECKING:
  from definable.memory.cortex.config import CortexConfig


class TraitInferencer:
  """Converts behavioral observations into trait model updates.

  Rules:
  - New observation matching no existing trait → create at confidence 0.3
  - Observation reinforcing existing trait → boost confidence
  - Observation contradicting existing trait → decay confidence
  - Matching uses keyword overlap between observation content and trait name/description
  """

  def __init__(self, config: Optional["CortexConfig"] = None):
    if config:
      self._boost = config.learning_reinforcement_boost
      self._decay = config.learning_confidence_decay
    else:
      self._boost = 0.15
      self._decay = 0.85

  def process(self, observations: List[Observation], model: UserModel) -> int:
    """Process observations and update the user model.

    Args:
      observations: New observations to process.
      model: The user model to update.

    Returns:
      Number of trait updates made.
    """
    updates = 0
    for obs in observations:
      matched = self._find_matching_trait(obs, model)
      if matched:
        matched.reinforce(obs, boost=self._boost)
        updates += 1
        log_debug(
          f"Trait reinforced: '{matched.name}' → {matched.confidence:.2f}",
          log_level=2,
        )
      else:
        # Create new trait
        trait = Trait(
          name=self._generate_trait_name(obs),
          description=obs.content,
          category=obs.category,
          confidence=0.3 * obs.signal_strength,
          observations=[obs],
        )
        model.add_trait(trait)
        updates += 1
        log_debug(f"New trait: '{trait.name}' at {trait.confidence:.2f}", log_level=2)

    return updates

  def check_contradictions(self, observations: List[Observation], model: UserModel) -> int:
    """Check for contradictions between new observations and existing traits.

    A contradiction is when an observation in the same category suggests
    the opposite of an existing strong trait.

    Returns number of traits whose confidence was decayed.
    """
    decayed = 0
    for obs in observations:
      for trait in model.get_traits_by_category(obs.category):
        if trait.confidence < 0.3:
          continue  # Don't bother with weak traits
        if self._is_contradictory(obs, trait):
          trait.contradict(decay=self._decay)
          decayed += 1
          log_debug(
            f"Trait contradicted: '{trait.name}' → {trait.confidence:.2f}",
            log_level=2,
          )
    return decayed

  def _find_matching_trait(self, obs: Observation, model: UserModel) -> Optional[Trait]:
    """Find an existing trait that matches this observation."""
    obs_words = set(obs.content.lower().split())
    best_match: Optional[Trait] = None
    best_overlap = 0

    for trait in model.get_traits_by_category(obs.category):
      trait_words = set(trait.description.lower().split())
      overlap = len(obs_words & trait_words)
      if overlap > best_overlap and overlap >= 2:
        best_overlap = overlap
        best_match = trait

    return best_match

  def _is_contradictory(self, obs: Observation, trait: Trait) -> bool:
    """Check if an observation contradicts a trait.

    Simple heuristic: opposite sentiment words in similar context.
    """
    negation_pairs = [
      ("direct", "verbose"),
      ("simple", "complex"),
      ("fast", "thorough"),
      ("casual", "formal"),
      ("bold", "cautious"),
    ]
    obs_lower = obs.content.lower()
    trait_lower = trait.description.lower()
    for a, b in negation_pairs:
      if (a in obs_lower and b in trait_lower) or (b in obs_lower and a in trait_lower):
        return True
    return False

  def _generate_trait_name(self, obs: Observation) -> str:
    """Generate a short trait name from an observation."""
    words = obs.content.split()[:5]
    return " ".join(words).rstrip(".,;:")
