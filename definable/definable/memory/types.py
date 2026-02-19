"""Core data types for the agentic memory system."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4


@dataclass
class UserMemory:
  """A single memory about a user, managed by the MemoryManager.

  Memories are natural-language statements extracted by the LLM from conversations.
  The LLM decides what to remember, update, or forget — no heuristic scoring.

  Attributes:
    memory: The memory content (natural language, e.g. "User prefers dark mode").
    memory_id: UUID, auto-generated if not provided.
    topics: LLM-assigned topic tags for filtering.
    user_id: User this memory belongs to.
    agent_id: Agent that created this memory.
    input: Original user message that triggered this memory.
    created_at: Epoch seconds when memory was created.
    updated_at: Epoch seconds when memory was last updated.
  """

  memory: str
  memory_id: Optional[str] = None
  topics: Optional[List[str]] = field(default_factory=list)
  user_id: Optional[str] = None
  agent_id: Optional[str] = None
  input: Optional[str] = None
  created_at: Optional[float] = None
  updated_at: Optional[float] = None

  def __post_init__(self) -> None:
    if self.memory_id is None:
      self.memory_id = str(uuid4())
    now = time.time()
    if self.created_at is None:
      self.created_at = now
    if self.updated_at is None:
      self.updated_at = now
    if self.topics is None:
      self.topics = []

  def to_dict(self) -> Dict[str, Any]:
    """Serialize to a plain dict."""
    return {
      "memory_id": self.memory_id,
      "memory": self.memory,
      "topics": self.topics or [],
      "user_id": self.user_id,
      "agent_id": self.agent_id,
      "input": self.input,
      "created_at": self.created_at,
      "updated_at": self.updated_at,
    }

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "UserMemory":
    """Deserialize from a plain dict."""
    return cls(
      memory=data["memory"],
      memory_id=data.get("memory_id"),
      topics=data.get("topics"),
      user_id=data.get("user_id"),
      agent_id=data.get("agent_id"),
      input=data.get("input"),
      created_at=data.get("created_at"),
      updated_at=data.get("updated_at"),
    )
