"""Core data types for the memory system."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4


@dataclass
class MemoryEntry:
  """A single entry in session memory.

  Represents a message, summary, or extracted memory atom stored for a session.
  Entries are scoped by session_id and user_id.

  Attributes:
    session_id: Session this entry belongs to.
    memory_id: Unique identifier (auto-generated UUID if not provided).
    user_id: User this entry is associated with.
    role: Message role ("user", "assistant", "tool", "system", "summary", "atom").
    content: Text content of the entry.
    message_data: Full serialized message data (preserves tool_calls, etc.).
    created_at: Epoch seconds when entry was created.
    updated_at: Epoch seconds when entry was last updated.
    entry_type: Type discriminator — "message" (raw), "atom" (extracted fact), "summary".
    lossless_content: Self-contained fact with all pronouns/relative time resolved.
    keywords: Search keywords for BM25/lexical retrieval.
    entities: Named entities (companies, products, technologies).
    persons: Person names referenced in this entry.
    topic: Short topic phrase (2-5 words).
    importance: Relevance weight (0.0–1.0). Used for consolidation ranking.
    vector: Embedding vector for similarity search. Set by Memory.embedder.
    superseded_by: Memory ID of the entry that replaced this one (soft-delete).
  """

  session_id: str
  memory_id: Optional[str] = None
  user_id: str = "default"
  role: str = "user"
  content: str = ""
  message_data: Optional[Dict[str, Any]] = None
  created_at: Optional[float] = None
  updated_at: Optional[float] = None

  # Semantic fields — populated by SemanticStrategy, empty for raw messages.
  entry_type: str = "message"
  lossless_content: Optional[str] = None
  keywords: List[str] = field(default_factory=list)
  entities: List[str] = field(default_factory=list)
  persons: List[str] = field(default_factory=list)
  topic: Optional[str] = None
  importance: float = 0.5
  vector: Optional[List[float]] = None
  superseded_by: Optional[str] = None

  def __post_init__(self) -> None:
    if self.memory_id is None:
      self.memory_id = str(uuid4())
    now = time.time()
    if self.created_at is None:
      self.created_at = now
    if self.updated_at is None:
      self.updated_at = now

  def to_dict(self) -> Dict[str, Any]:
    """Serialize to a plain dict."""
    d: Dict[str, Any] = {
      "memory_id": self.memory_id,
      "session_id": self.session_id,
      "user_id": self.user_id,
      "role": self.role,
      "content": self.content,
      "message_data": self.message_data,
      "created_at": self.created_at,
      "updated_at": self.updated_at,
    }
    # Include semantic fields only when non-default (keeps regular messages compact).
    if self.entry_type != "message":
      d["entry_type"] = self.entry_type
    if self.lossless_content is not None:
      d["lossless_content"] = self.lossless_content
    if self.keywords:
      d["keywords"] = self.keywords
    if self.entities:
      d["entities"] = self.entities
    if self.persons:
      d["persons"] = self.persons
    if self.topic is not None:
      d["topic"] = self.topic
    if self.importance != 0.5:
      d["importance"] = self.importance
    if self.vector is not None:
      d["vector"] = self.vector
    if self.superseded_by is not None:
      d["superseded_by"] = self.superseded_by
    return d

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
    """Deserialize from a plain dict."""
    return cls(
      memory_id=data.get("memory_id"),
      session_id=data.get("session_id", ""),
      user_id=data.get("user_id", "default"),
      role=data.get("role", "user"),
      content=data.get("content", ""),
      message_data=data.get("message_data"),
      created_at=data.get("created_at"),
      updated_at=data.get("updated_at"),
      entry_type=data.get("entry_type", "message"),
      lossless_content=data.get("lossless_content"),
      keywords=data.get("keywords", []),
      entities=data.get("entities", []),
      persons=data.get("persons", []),
      topic=data.get("topic"),
      importance=data.get("importance", 0.5),
      vector=data.get("vector"),
      superseded_by=data.get("superseded_by"),
    )
