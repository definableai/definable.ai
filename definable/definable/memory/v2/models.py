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
class WarmMemory:
  """Extended memory tier — overflow from hot working memory.

  Not injected into every prompt. Accessible via read_extended_memory tool.
  """

  user_id: str
  content: str = ""
  updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class WorkingMemorySnapshot:
  """Historical snapshot of working memory for rollback."""

  user_id: str
  version: int
  content: str
  updated_at: datetime
  session_id: str = ""


@dataclass
class IndexEntry:
  """Summary entry in the archived memory index.

  The LLM scans these summaries to decide which entries to fetch.
  """

  id: str = field(default_factory=lambda: uuid4().hex[:12])
  user_id: str = ""
  summary: str = ""
  category: str = "conversation"
  tags: List[str] = field(default_factory=list)
  created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
  session_id: str = ""
  # Access tracking
  access_count: int = 0
  last_accessed_at: Optional[datetime] = None
  # Quality metadata
  confidence: float = 1.0  # 1.0 = user stated, 0.7 = inferred, 0.5 = uncertain
  source: str = "user_stated"  # user_stated | user_implied | agent_observed


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
  expires_at: Optional[datetime] = None  # Optional TTL


@dataclass
class MemoryStats:
  """Usage statistics for a user's memory."""

  user_id: str
  wm_chars: int = 0
  wm_version: int = 0
  warm_chars: int = 0
  entry_count: int = 0
  total_content_chars: int = 0
  oldest_entry: Optional[datetime] = None
  newest_entry: Optional[datetime] = None
  categories: dict = field(default_factory=dict)  # category -> count


@dataclass
class ConsolidationReport:
  """Result of a consolidation run."""

  user_id: str
  duplicates_merged: int = 0
  stale_pruned: int = 0
  expired_removed: int = 0
  entries_before: int = 0
  entries_after: int = 0
