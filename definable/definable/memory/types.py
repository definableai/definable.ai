"""Core data types for the memory system."""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from uuid import uuid4


@dataclass
class MemoryEntry:
  """A single entry in session memory.

  Represents a message, summary, or system note stored for a session.
  Entries are scoped by session_id and user_id.

  Attributes:
    session_id: Session this entry belongs to.
    memory_id: Unique identifier (auto-generated UUID if not provided).
    user_id: User this entry is associated with.
    role: Message role ("user", "assistant", "tool", "system", "summary").
    content: Text content of the entry.
    message_data: Full serialized message data (preserves tool_calls, etc.).
    created_at: Epoch seconds when entry was created.
    updated_at: Epoch seconds when entry was last updated.
  """

  session_id: str
  memory_id: Optional[str] = None
  user_id: str = "default"
  role: str = "user"
  content: str = ""
  message_data: Optional[Dict[str, Any]] = None
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

  def to_dict(self) -> Dict[str, Any]:
    """Serialize to a plain dict."""
    return {
      "memory_id": self.memory_id,
      "session_id": self.session_id,
      "user_id": self.user_id,
      "role": self.role,
      "content": self.content,
      "message_data": self.message_data,
      "created_at": self.created_at,
      "updated_at": self.updated_at,
    }

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
    )
