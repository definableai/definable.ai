"""Data models for the tool-based memory system."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4


@dataclass
class WorkingMemory:
  """Always-loaded scratchpad for the current user.

  Contains critical context: user facts, active goals, preferences.
  The agent rewrites this fully via the update_working_memory tool.
  """

  user_id: str
  content: str = ""
  updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
  version: int = 0


@dataclass
class IndexEntry:
  """Summary entry in the non-working memory index.

  The LLM scans these summaries to decide which entries to fetch.
  """

  id: str = field(default_factory=lambda: uuid4().hex[:12])
  user_id: str = ""
  summary: str = ""
  category: str = "conversation"
  tags: List[str] = field(default_factory=list)
  created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
  session_id: str = ""


@dataclass
class MemoryEntry:
  """Full content of an archived memory entry."""

  id: str = ""
  user_id: str = ""
  content: str = ""
  category: str = "conversation"
  created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
  updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
  source_turn: Optional[int] = None
