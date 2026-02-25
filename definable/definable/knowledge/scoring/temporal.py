"""Temporal decay — exponential score decay based on document age.

Older documents receive progressively lower scores, making recent
knowledge more prominent in retrieval. Evergreen documents are exempt.

Usage::

    from definable.knowledge.scoring import TemporalDecay

    decay = TemporalDecay(half_life_days=30.0)
    scored = decay.apply(documents)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional

from definable.knowledge.document import Document


@dataclass
class TemporalDecay:
  """Exponential temporal decay for document relevance scores.

  The decay factor follows: ``score *= exp(-ln(2) / half_life_days * age_days)``.
  After one half-life, a document's score is halved.

  Attributes:
    half_life_days: Number of days until score is halved. Default 30.
    enabled: Whether decay is active.
  """

  half_life_days: float = 30.0
  enabled: bool = True

  def decay_factor(self, age_days: float) -> float:
    """Compute the decay multiplier for a given age in days.

    Args:
      age_days: Age of the document in days.

    Returns:
      Multiplier in (0, 1].
    """
    if age_days <= 0 or self.half_life_days <= 0:
      return 1.0
    return math.exp(-math.log(2) / self.half_life_days * age_days)

  def apply(
    self,
    documents: List[Document],
    now: Optional[float] = None,
  ) -> List[Document]:
    """Apply temporal decay to document scores.

    Documents with ``meta_data["evergreen"] = True`` are exempt.
    Documents without an ``inserted_at`` or ``created_at`` timestamp
    in their metadata are left unchanged.

    Args:
      documents: Documents with reranking_score or relevance scores.
      now: Current timestamp (seconds since epoch). Defaults to time.time().

    Returns:
      Documents sorted by decayed score (highest first).
    """
    if not self.enabled or not documents:
      return documents

    current_time = now or time.time()
    result: list[Document] = []

    for doc in documents:
      meta = doc.meta_data or {}

      # Skip evergreen documents
      if meta.get("evergreen") is True:
        result.append(doc)
        continue

      # Get document timestamp
      inserted_at = meta.get("inserted_at") or meta.get("created_at")
      if inserted_at is None:
        result.append(doc)
        continue

      # Compute age and apply decay
      try:
        age_seconds = current_time - float(inserted_at)
      except (TypeError, ValueError):
        result.append(doc)
        continue

      age_days = age_seconds / 86400.0
      factor = self.decay_factor(age_days)

      # Apply to reranking_score (preserving original)
      score = doc.reranking_score if doc.reranking_score is not None else 1.0
      doc.reranking_score = score * factor
      result.append(doc)

    # Sort by decayed score (highest first)
    result.sort(key=lambda d: d.reranking_score if d.reranking_score is not None else 0.0, reverse=True)
    return result
