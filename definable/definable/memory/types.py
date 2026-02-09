"""Core types for the cognitive memory system."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Episode:
  """A single conversation turn stored in memory."""

  id: str
  user_id: Optional[str]
  session_id: str
  role: str  # "user" or "assistant"
  content: str
  embedding: Optional[List[float]] = None
  topics: List[str] = field(default_factory=list)
  sentiment: float = 0.0  # -1.0 to 1.0
  token_count: int = 0
  compression_stage: int = 0  # 0=raw, 1=summary, 2=facts, 3=atoms
  created_at: float = 0.0
  last_accessed_at: float = 0.0
  access_count: int = 0


@dataclass
class KnowledgeAtom:
  """An extracted fact stored as a subject-predicate-object triple."""

  id: str
  user_id: Optional[str]
  subject: str
  predicate: str
  object: str
  content: str  # Human-readable form: "user lives in San Francisco"
  embedding: Optional[List[float]] = None
  confidence: float = 1.0  # 0.0 to 1.0
  reinforcement_count: int = 0
  topics: List[str] = field(default_factory=list)
  token_count: int = 0
  source_episode_ids: List[str] = field(default_factory=list)
  created_at: float = 0.0
  last_accessed_at: float = 0.0
  last_reinforced_at: float = 0.0
  access_count: int = 0


@dataclass
class Procedure:
  """A learned behavioral pattern (trigger -> action)."""

  id: str
  user_id: Optional[str]
  trigger: str  # "user asks for code"
  action: str  # "use Python, add type hints"
  confidence: float = 0.5
  observation_count: int = 1
  created_at: float = 0.0
  last_accessed_at: float = 0.0


@dataclass
class TopicTransition:
  """A transition between topics with frequency and probability."""

  from_topic: str
  to_topic: str
  count: int = 0
  probability: float = 0.0  # count / total_transitions_from_topic


@dataclass
class MemoryPayload:
  """Result of a memory recall operation, ready for injection into system prompt."""

  context: str  # Formatted XML string for injection
  tokens_used: int = 0
  chunks_included: int = 0
  chunks_available: int = 0
