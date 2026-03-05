"""Core data types for the Cortex memory system."""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class EdgeType(str, Enum):
  """Types of edges in the knowledge graph."""

  SEMANTIC = "semantic"
  TEMPORAL = "temporal"
  CAUSAL = "causal"
  ENTITY = "entity"


class MemorySource(str, Enum):
  """How a memory was created."""

  CONVERSATION = "conversation"
  OBSERVATION = "observation"
  INFERENCE = "inference"
  EXTERNAL = "external"
  CONSOLIDATION = "consolidation"


@dataclass
class NarrativeEpisode:
  """A narrative summary of an interaction or event.

  Captures the story-form of what happened, preserving emotional tone,
  causal chains, and contextual detail that atomic facts lose.
  """

  episode_id: str = ""
  content: str = ""
  participants: List[str] = field(default_factory=list)
  emotional_tone: Optional[str] = None
  causal_chain: List[str] = field(default_factory=list)
  created_at: float = 0.0

  def __post_init__(self) -> None:
    if not self.episode_id:
      self.episode_id = str(uuid4())
    if self.created_at == 0.0:
      self.created_at = time.time()

  def to_dict(self) -> Dict[str, Any]:
    return {
      "episode_id": self.episode_id,
      "content": self.content,
      "participants": self.participants,
      "emotional_tone": self.emotional_tone,
      "causal_chain": self.causal_chain,
      "created_at": self.created_at,
    }

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "NarrativeEpisode":
    return cls(
      episode_id=data.get("episode_id", ""),
      content=data.get("content", ""),
      participants=data.get("participants", []),
      emotional_tone=data.get("emotional_tone"),
      causal_chain=data.get("causal_chain", []),
      created_at=data.get("created_at", 0.0),
    )


@dataclass
class Fact:
  """An atomic, self-contained factual statement.

  Extracted from conversation with all pronouns and relative references resolved.
  """

  fact_id: str = ""
  content: str = ""
  confidence: float = 1.0
  source_turns: List[int] = field(default_factory=list)
  entities: List[str] = field(default_factory=list)
  created_at: float = 0.0

  def __post_init__(self) -> None:
    if not self.fact_id:
      self.fact_id = str(uuid4())
    if self.created_at == 0.0:
      self.created_at = time.time()

  def to_dict(self) -> Dict[str, Any]:
    return {
      "fact_id": self.fact_id,
      "content": self.content,
      "confidence": self.confidence,
      "source_turns": self.source_turns,
      "entities": self.entities,
      "created_at": self.created_at,
    }

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "Fact":
    return cls(
      fact_id=data.get("fact_id", ""),
      content=data.get("content", ""),
      confidence=data.get("confidence", 1.0),
      source_turns=data.get("source_turns", []),
      entities=data.get("entities", []),
      created_at=data.get("created_at", 0.0),
    )


@dataclass
class Edge:
  """A directed edge in the knowledge graph."""

  source_id: str = ""
  target_id: str = ""
  edge_type: EdgeType = EdgeType.SEMANTIC
  weight: float = 1.0
  label: Optional[str] = None
  created_at: float = 0.0

  def __post_init__(self) -> None:
    if self.created_at == 0.0:
      self.created_at = time.time()

  def to_dict(self) -> Dict[str, Any]:
    return {
      "source_id": self.source_id,
      "target_id": self.target_id,
      "edge_type": self.edge_type.value,
      "weight": self.weight,
      "label": self.label,
      "created_at": self.created_at,
    }

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "Edge":
    return cls(
      source_id=data.get("source_id", ""),
      target_id=data.get("target_id", ""),
      edge_type=EdgeType(data.get("edge_type", "semantic")),
      weight=data.get("weight", 1.0),
      label=data.get("label"),
      created_at=data.get("created_at", 0.0),
    )


@dataclass
class MemoryRecord:
  """A unified memory record in Cortex.

  Multi-representation: a single interaction can produce a narrative episode,
  multiple facts, a binary signature, tags, and graph edges.
  """

  record_id: str = ""
  session_id: str = "default"
  user_id: str = "default"
  source: MemorySource = MemorySource.CONVERSATION
  raw_content: str = ""
  role: str = "user"
  turn_index: int = 0
  created_at: float = 0.0
  updated_at: float = 0.0

  # Multi-representation fields
  narrative: Optional[NarrativeEpisode] = None
  facts: List[Fact] = field(default_factory=list)
  tags: List[str] = field(default_factory=list)
  signature: Optional[bytes] = None
  embedding: Optional[List[float]] = None

  # Lifecycle
  staleness: float = 0.0
  superseded_by: Optional[str] = None

  def __post_init__(self) -> None:
    if not self.record_id:
      self.record_id = str(uuid4())
    now = time.time()
    if self.created_at == 0.0:
      self.created_at = now
    if self.updated_at == 0.0:
      self.updated_at = now

  @property
  def is_active(self) -> bool:
    return self.superseded_by is None

  def to_dict(self) -> Dict[str, Any]:
    d: Dict[str, Any] = {
      "record_id": self.record_id,
      "session_id": self.session_id,
      "user_id": self.user_id,
      "source": self.source.value,
      "raw_content": self.raw_content,
      "role": self.role,
      "turn_index": self.turn_index,
      "created_at": self.created_at,
      "updated_at": self.updated_at,
      "staleness": self.staleness,
      "superseded_by": self.superseded_by,
    }
    if self.narrative:
      d["narrative"] = self.narrative.to_dict()
    if self.facts:
      d["facts"] = [f.to_dict() for f in self.facts]
    if self.tags:
      d["tags"] = self.tags
    if self.signature is not None:
      import base64

      d["signature"] = base64.b64encode(self.signature).decode("ascii")
    if self.embedding is not None:
      d["embedding"] = self.embedding
    return d

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "MemoryRecord":
    narrative = None
    if "narrative" in data and data["narrative"]:
      narrative = NarrativeEpisode.from_dict(data["narrative"])
    facts = [Fact.from_dict(f) for f in data.get("facts", [])]
    signature = None
    if "signature" in data and data["signature"]:
      import base64

      signature = base64.b64decode(data["signature"])
    return cls(
      record_id=data.get("record_id", ""),
      session_id=data.get("session_id", "default"),
      user_id=data.get("user_id", "default"),
      source=MemorySource(data.get("source", "conversation")),
      raw_content=data.get("raw_content", ""),
      role=data.get("role", "user"),
      turn_index=data.get("turn_index", 0),
      created_at=data.get("created_at", 0.0),
      updated_at=data.get("updated_at", 0.0),
      narrative=narrative,
      facts=facts,
      tags=data.get("tags", []),
      signature=signature,
      embedding=data.get("embedding"),
      staleness=data.get("staleness", 0.0),
      superseded_by=data.get("superseded_by"),
    )
