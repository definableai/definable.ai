"""Scratchpad — always-retrieved mutable belief state.

The scratchpad is a per-user/session key-value store of current beliefs,
preferences, and active context. It is always included in retrieval results
(Layer 1 of the cascade) so the agent has immediate access to the user's
current state without needing a search query.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Scratchpad:
  """Mutable belief state that is always retrieved.

  Attributes:
    session_id: Session scope.
    user_id: User scope.
    beliefs: Key-value belief store.
    active_topics: Currently active conversation topics.
    pending_tasks: Tasks the user has mentioned but not completed.
    updated_at: Last modification timestamp.
  """

  session_id: str = "default"
  user_id: str = "default"
  beliefs: Dict[str, Any] = field(default_factory=dict)
  active_topics: list[str] = field(default_factory=list)
  pending_tasks: list[str] = field(default_factory=list)
  updated_at: float = 0.0

  def __post_init__(self) -> None:
    if self.updated_at == 0.0:
      self.updated_at = time.time()

  def set_belief(self, key: str, value: Any) -> None:
    """Set a belief, updating the timestamp."""
    self.beliefs[key] = value
    self.updated_at = time.time()

  def get_belief(self, key: str, default: Any = None) -> Any:
    """Get a belief value."""
    return self.beliefs.get(key, default)

  def remove_belief(self, key: str) -> None:
    """Remove a belief if it exists."""
    self.beliefs.pop(key, None)
    self.updated_at = time.time()

  def add_topic(self, topic: str) -> None:
    if topic not in self.active_topics:
      self.active_topics.append(topic)
      self.updated_at = time.time()

  def remove_topic(self, topic: str) -> None:
    if topic in self.active_topics:
      self.active_topics.remove(topic)
      self.updated_at = time.time()

  def format_for_prompt(self) -> str:
    """Format scratchpad as XML for system prompt injection."""
    parts = ["<scratchpad>"]
    if self.beliefs:
      parts.append("  <beliefs>")
      for k, v in self.beliefs.items():
        parts.append(f'    <belief key="{k}">{v}</belief>')
      parts.append("  </beliefs>")
    if self.active_topics:
      parts.append(f"  <active_topics>{', '.join(self.active_topics)}</active_topics>")
    if self.pending_tasks:
      parts.append("  <pending_tasks>")
      for task in self.pending_tasks:
        parts.append(f"    <task>{task}</task>")
      parts.append("  </pending_tasks>")
    parts.append("</scratchpad>")
    return "\n".join(parts)

  def to_dict(self) -> Dict[str, Any]:
    return {
      "session_id": self.session_id,
      "user_id": self.user_id,
      "beliefs": self.beliefs,
      "active_topics": self.active_topics,
      "pending_tasks": self.pending_tasks,
      "updated_at": self.updated_at,
    }

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "Scratchpad":
    return cls(
      session_id=data.get("session_id", "default"),
      user_id=data.get("user_id", "default"),
      beliefs=data.get("beliefs", {}),
      active_topics=data.get("active_topics", []),
      pending_tasks=data.get("pending_tasks", []),
      updated_at=data.get("updated_at", 0.0),
    )


def merge_scratchpads(base: Scratchpad, update: Optional[Scratchpad]) -> Scratchpad:
  """Merge two scratchpads. Update takes precedence for conflicting beliefs."""
  if update is None:
    return base
  merged = Scratchpad(
    session_id=base.session_id,
    user_id=base.user_id,
    beliefs={**base.beliefs, **update.beliefs},
    active_topics=list(dict.fromkeys(base.active_topics + update.active_topics)),
    pending_tasks=list(dict.fromkeys(base.pending_tasks + update.pending_tasks)),
  )
  return merged
