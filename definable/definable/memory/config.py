"""Configuration for the cognitive memory system."""

from dataclasses import dataclass, field


@dataclass
class ScoringWeights:
  """Weights for the 5-factor composite relevance scorer.

  All weights must sum to 1.0. Adjust to prioritize different retrieval
  signals — e.g. increase semantic_similarity when using an embedder,
  or increase recency for short-lived conversational agents.
  """

  semantic_similarity: float = 0.35  # Cosine similarity between query and memory embedding (requires embedder)
  recency: float = 0.25  # Exponential decay based on time since last access
  access_frequency: float = 0.15  # How often the memory has been retrieved (log-normalized)
  predicted_need: float = 0.15  # Whether memory topics match predicted next topics
  emotional_salience: float = 0.10  # abs(sentiment) — extreme emotions are more memorable

  def __post_init__(self):
    total = self.semantic_similarity + self.recency + self.access_frequency + self.predicted_need + self.emotional_salience
    if abs(total - 1.0) > 1e-6:
      raise ValueError(f"ScoringWeights must sum to 1.0, got {total:.6f}")


@dataclass
class MemoryConfig:
  """Configuration for the CognitiveMemory system.

  All values have sensible defaults. Override only what you need::

      config = MemoryConfig(
          decay_half_life_days=7,  # Faster forgetting
          scoring_weights=ScoringWeights(
              semantic_similarity=0.5,  # Prioritize embedding search
              recency=0.2,
              access_frequency=0.1,
              predicted_need=0.1,
              emotional_salience=0.1,
          ),
      )
  """

  # How many days until a memory's recency score drops to 50%.
  # Lower = more aggressive forgetting; higher = longer memory.
  decay_half_life_days: float = 14.0

  # Weights for the 5-factor relevance scorer.
  scoring_weights: ScoringWeights = field(default_factory=ScoringWeights)

  # Distillation age thresholds (seconds). Episodes older than these
  # thresholds are progressively compressed:
  #   stage 0 -> 1 (raw -> summary) after 1 hour
  #   stage 1 -> 2 (summary -> facts) after 24 hours
  #   stage 2 -> 3 (facts -> atoms) after 7 days
  #   stage 3+ (archive/merge low-confidence atoms) after 30 days
  distillation_stage_0_age: float = 3600.0
  distillation_stage_1_age: float = 86400.0
  distillation_stage_2_age: float = 604800.0
  distillation_stage_3_age: float = 2592000.0

  # Max episodes to process per distillation run.
  distillation_batch_size: int = 10

  # Confidence boost when a fact is re-observed (0.0–1.0).
  reinforcement_boost: float = 0.15

  # Topic transition model: require at least this many observations
  # and this probability before predicting a topic transition.
  topic_transition_min_count: int = 3
  topic_transition_min_probability: float = 0.3

  # How many candidates to retrieve per vector-search path.
  retrieval_top_k: int = 20

  # Max recent session episodes to include in recall.
  recent_episodes_limit: int = 5
