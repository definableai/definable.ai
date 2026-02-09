"""Shared utilities for memory store implementations."""

import math
from typing import List


def cosine_similarity(a: List[float], b: List[float]) -> float:
  """Compute cosine similarity between two vectors."""
  if len(a) != len(b) or not a:
    return 0.0
  dot = sum(x * y for x, y in zip(a, b))
  mag_a = math.sqrt(sum(x * x for x in a))
  mag_b = math.sqrt(sum(x * x for x in b))
  if mag_a == 0.0 or mag_b == 0.0:
    return 0.0
  return dot / (mag_a * mag_b)
