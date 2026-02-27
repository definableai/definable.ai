"""Atom consolidation — decay, merge, and prune for long-term memory health.

Consolidation runs after optimization to keep the atom pool lean and accurate.
Three operations, applied in order:

  1. **Decay** — older atoms lose importance exponentially over time.
  2. **Merge** — near-duplicate atoms (cosine sim > threshold) are merged:
     the higher-importance one survives, the other is soft-deleted.
  3. **Prune** — atoms whose importance falls below a floor are soft-deleted.

All deletions are *soft* — ``superseded_by`` is set, entries are never removed.
The search layer already filters superseded atoms.
"""

import time
from dataclasses import dataclass
from typing import List, Optional

from definable.memory.types import MemoryEntry
from definable.utils.log import log_debug


@dataclass
class ConsolidationPolicy:
  """Tunable knobs for atom consolidation.

  Attributes:
    decay_half_life_days: How many days until an atom's importance halves.
        Set to 0 to disable decay entirely.
    merge_similarity_threshold: Cosine similarity above which two atoms
        are considered near-duplicates and merged.
    min_importance: Atoms that drop below this floor are soft-deleted.
    enabled: Master switch. When False, consolidation is a no-op.
  """

  decay_half_life_days: float = 30.0
  merge_similarity_threshold: float = 0.92
  min_importance: float = 0.05
  enabled: bool = True


async def consolidate(
  atoms: List[MemoryEntry],
  policy: ConsolidationPolicy,
  now: Optional[float] = None,
) -> List[MemoryEntry]:
  """Run decay → merge → prune on a list of atom entries.

  Mutates entries in-place (importance, superseded_by) and returns the
  full list (including soft-deleted ones) so the caller can persist changes.

  Only operates on atoms that are not already superseded.

  Args:
    atoms: All atom entries from a session (may include superseded).
    policy: Consolidation tuning knobs.
    now: Current epoch seconds (injectable for testing).

  Returns:
    The same list with mutations applied.
  """
  if not policy.enabled or not atoms:
    return atoms

  current_time = now or time.time()
  active = [a for a in atoms if a.superseded_by is None]

  if not active:
    return atoms

  # Phase 1: Decay
  if policy.decay_half_life_days > 0:
    _apply_decay(active, policy.decay_half_life_days, current_time)

  # Phase 2: Merge near-duplicates
  _merge_duplicates(active, policy.merge_similarity_threshold)

  # Phase 3: Prune low-importance (only atoms still active after merge)
  still_active = [a for a in active if a.superseded_by is None]
  _prune_low_importance(still_active, policy.min_importance)

  pruned_count = sum(1 for a in active if a.superseded_by is not None)
  if pruned_count:
    log_debug(f"Consolidation: {pruned_count} atoms soft-deleted out of {len(active)} active")

  return atoms


def _apply_decay(atoms: List[MemoryEntry], half_life_days: float, now: float) -> None:
  """Exponential importance decay based on age."""
  half_life_seconds = half_life_days * 86400.0
  for atom in atoms:
    age = now - (atom.created_at or now)
    if age <= 0:
      continue
    decay_factor = 0.5 ** (age / half_life_seconds)
    atom.importance = atom.importance * decay_factor


def _merge_duplicates(atoms: List[MemoryEntry], threshold: float) -> None:
  """Merge near-duplicate atoms by cosine similarity.

  For each pair above the threshold, the lower-importance atom is
  soft-deleted with superseded_by pointing to the winner.
  Only compares atoms that both have vectors.
  """
  from definable.memory.manager import _cosine_similarity

  n = len(atoms)
  for i in range(n):
    ai = atoms[i]
    if ai.superseded_by is not None or not ai.vector:
      continue
    vec_i = ai.vector
    for j in range(i + 1, n):
      aj = atoms[j]
      if aj.superseded_by is not None or not aj.vector:
        continue
      sim = _cosine_similarity(vec_i, aj.vector)
      if sim >= threshold:
        # Keep the higher-importance atom.
        if ai.importance >= aj.importance:
          winner, loser = ai, aj
        else:
          winner, loser = aj, ai
        loser.superseded_by = winner.memory_id


def _prune_low_importance(atoms: List[MemoryEntry], min_importance: float) -> None:
  """Soft-delete atoms below the importance floor."""
  for atom in atoms:
    if atom.superseded_by is not None:
      continue
    if atom.importance < min_importance:
      atom.superseded_by = "pruned"
