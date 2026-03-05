"""CortexMemory configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CortexConfig:
  """Configuration for the Cortex memory layer.

  Controls which subsystems are enabled and their parameters.

  Attributes:
    db_path: Path to SQLite database. None = auto via workspace_path.
    enable_signatures: Enable binary signature index for fast filtering.
    enable_graph: Enable graph index for relational traversal.
    enable_tags: Enable hierarchical tag index.
    enable_learning: Enable behavioral learning (observer + inferencer).
    enable_consolidation: Enable background consolidation (merge/prune).
    signature_dims: Dimensionality of random indexing signatures.
    signature_nnz: Non-zero elements per random index vector.
    graph_max_hops: Max BFS depth for graph traversal.
    tag_separator: Path separator for hierarchical tags.
    consolidation_interval_s: Seconds between consolidation runs.
    duplicate_threshold: Cosine similarity threshold for duplicate detection.
    staleness_decay: Per-hop decay factor for cascade propagation.
    slow_path_enabled: Enable LLM-based slow ingestion path.
    retrieval_top_k: Default number of results from retrieval.
    scratchpad_max_beliefs: Max beliefs in scratchpad before pruning.
    learning_min_observations: Observations needed before trait inference.
    learning_confidence_decay: Multiplicative decay on contradiction.
    learning_reinforcement_boost: Confidence boost per reinforcing observation.
  """

  db_path: Optional[str] = None

  # Subsystem toggles
  enable_signatures: bool = True
  enable_graph: bool = True
  enable_tags: bool = True
  enable_learning: bool = True
  enable_consolidation: bool = False  # Off for tests, on in production

  # Signature index
  signature_dims: int = 1024
  signature_nnz: int = 8

  # Graph index
  graph_max_hops: int = 3

  # Tag index
  tag_separator: str = "/"

  # Consolidation
  consolidation_interval_s: float = 300.0
  duplicate_threshold: float = 0.92
  staleness_decay: float = 0.5

  # Ingestion
  slow_path_enabled: bool = True

  # Retrieval
  retrieval_top_k: int = 10

  # Scratchpad
  scratchpad_max_beliefs: int = 50

  # Learning
  learning_min_observations: int = 1
  learning_confidence_decay: float = 0.85  # multiply on contradiction
  learning_reinforcement_boost: float = 0.15  # add per reinforcement
