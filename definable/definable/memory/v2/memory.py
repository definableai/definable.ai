"""Memory — tool-based memory orchestrator.

The LLM is the memory manager. This class wires the store, tools,
skill, and prompt injection together. The developer just passes it to Agent.
"""

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from definable.memory.v2.models import ConsolidationReport, MemoryStats
from definable.memory.v2.prompt import build_working_memory_block
from definable.memory.v2.stores.base import MemoryStore

if TYPE_CHECKING:
  from definable.skill.base import Skill
  from definable.tool.function import Function

_SKILL_DIR = Path(__file__).parent / "skill"

# Sections that the WM template requires
WM_REQUIRED_SECTIONS = ["## Identity", "## Preferences", "## Projects", "## Team", "## Other"]

# Summaries matching these patterns are too vague to be useful
VAGUE_SUMMARY_PATTERNS = [
  "'s identity",
  "'s background",
  "'s preferences",
  "activities and interests",
  "professional details",
  "personal information",
  "technical background",
  "work details",
]


class Memory:
  """Tool-based memory system with enterprise features.

  The agent gets memory tools auto-injected, a memory-manager skill
  loaded into the system prompt, and working memory injected before
  every LLM call.

  Features:
    - Two-tier working memory (hot + warm)
    - Recency-weighted archive search
    - Access tracking on entries
    - Background consolidation (dedupe, prune, limit enforcement)
    - WM versioning and rollback
    - GDPR delete/export
    - Framework-level validation (WM structure, summary quality)

  Usage:
      from definable.memory.v2 import Memory, SQLiteStore

      agent = Agent(
          model="openai/gpt-4o",
          memory=Memory(store=SQLiteStore("./memory.db")),
      )
  """

  def __init__(
    self,
    store: MemoryStore,
    *,
    working_memory_max_chars: int = 4000,
    warm_memory_max_chars: int = 12000,
    index_max_entries: int = 500,
    categories: Optional[List[str]] = None,
    auto_inject: bool = True,
    half_life_days: float = 30.0,
    stale_days: int = 90,
  ) -> None:
    self.store = store
    self.working_memory_max_chars = working_memory_max_chars
    self.warm_memory_max_chars = warm_memory_max_chars
    self.index_max_entries = index_max_entries
    self.categories = categories or ["user", "feedback", "project", "reference", "conversation"]
    self.auto_inject = auto_inject
    self.half_life_days = half_life_days
    self.stale_days = stale_days
    self._skill: Optional["Skill"] = None

    # Compat flags for agent integration (v1 Memory had these)
    self.enabled = True
    self.description: Optional[str] = None
    self.model: Optional[object] = None
    self.has_semantic_search: bool = False

  def get_skill(self) -> "Skill":
    """Load and return the memory-manager MarkdownSkill."""
    if self._skill is None:
      from definable.skill.markdown import SkillLoader

      self._skill = SkillLoader.load_skill_directory(_SKILL_DIR)
    return self._skill

  def get_tools(self, user_id: str, session_id: str) -> "List[Function]":
    """Build memory tools bound to the given user/session context."""
    from definable.memory.v2.tools import build_memory_tools

    return build_memory_tools(self, user_id, session_id)

  async def get_prompt_injection(self, user_id: str) -> str:
    """Build the working memory block for system prompt injection."""
    wm = await self.store.get_working_memory(user_id)
    content = wm.content if wm else ""
    updated_at = wm.updated_at.isoformat() if wm else ""
    return build_working_memory_block(user_id, content, updated_at)

  async def get_session_preamble(self, user_id: str, limit: int = 5) -> str:
    """Build a brief context from recent archive entries for cold-start continuity.

    Inject alongside WM at the start of each session.
    """
    entries = await self.store.search_index(user_id, limit=limit)
    if not entries:
      return ""
    lines = [f"- {e.summary}" for e in entries]
    return "<recent_context>\n" + "\n".join(lines) + "\n</recent_context>"

  # --- Consolidation ---

  async def consolidate(self, user_id: str) -> ConsolidationReport:
    """Merge duplicates, prune stale/expired, enforce limits.

    Call periodically (e.g., nightly, after every N sessions).
    No LLM calls — pure algorithmic.
    """
    report = ConsolidationReport(user_id=user_id)
    report.entries_before = await self.store.count_entries(user_id)

    # 1. Remove expired entries
    report.expired_removed = await self.store.prune_expired(user_id)

    # 2. Detect and merge near-duplicates (word overlap > 60%)
    entries = await self.store.search_index(user_id, limit=5000)
    seen_ids: set[str] = set()
    for i, a in enumerate(entries):
      if a.id in seen_ids:
        continue
      for b in entries[i + 1 :]:
        if b.id in seen_ids:
          continue
        if a.category == b.category and _word_overlap(a.summary, b.summary) > 0.6:
          # Keep the newer or more-accessed entry
          keep, drop = (a, b) if a.created_at >= b.created_at else (b, a)
          if drop.access_count > keep.access_count:
            keep, drop = drop, keep
          await self.store.delete_entry(drop.id)
          seen_ids.add(drop.id)
          report.duplicates_merged += 1

    # 3. Prune stale entries (not accessed in stale_days, low access count)
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    remaining = await self.store.search_index(user_id, limit=5000)
    for entry in remaining:
      if entry.id in seen_ids:
        continue
      if entry.last_accessed_at is None:
        days_since = (now - entry.created_at).days
      else:
        days_since = (now - entry.last_accessed_at).days
      if days_since > self.stale_days and entry.access_count < 3:
        await self.store.delete_entry(entry.id)
        report.stale_pruned += 1

    # 4. Enforce entry limit
    overflow = await self.store.enforce_limit(user_id, self.index_max_entries)
    report.stale_pruned += overflow

    report.entries_after = await self.store.count_entries(user_id)
    return report

  # --- Post-turn audit ---

  async def post_turn_audit(self, user_id: str) -> List[str]:
    """Check memory integrity after a turn. Returns list of issues."""
    issues: List[str] = []
    wm = await self.store.get_working_memory(user_id)
    if wm:
      for section in WM_REQUIRED_SECTIONS:
        if section not in wm.content:
          issues.append(f"WM missing section: {section}")
      if len(wm.content) < 100 and wm.version > 3:
        issues.append(f"WM suspiciously small ({len(wm.content)} chars at version {wm.version})")
    return issues

  # --- Stats ---

  async def get_stats(self, user_id: str) -> MemoryStats:
    """Get memory usage statistics for a user."""
    return await self.store.get_stats(user_id)

  # --- GDPR ---

  async def delete_user(self, user_id: str) -> int:
    """Delete all memory data for a user. Returns entry count deleted."""
    return await self.store.delete_user(user_id)

  async def export_user(self, user_id: str) -> dict:
    """Export all user data as JSON."""
    return await self.store.export_user(user_id)

  # --- v1 Memory compat (agent pipeline calls these) ---

  async def _ensure_initialized(self) -> None:
    """No-op — v2 store initializes lazily."""

  async def add(self, message: object, *, session_id: str = "", user_id: str = "") -> None:
    """No-op — v2 memory is tool-managed, not pipeline-managed."""

  async def get_entries(self, session_id: str = "", user_id: str = "") -> list:
    """No-op — v2 recall happens via LLM tool calls, not the recall phase."""
    return []

  async def close(self) -> None:
    """Clean up store resources."""
    await self.store.close()


def _word_overlap(a: str, b: str) -> float:
  """Calculate word overlap ratio between two strings."""
  words_a = set(a.lower().split())
  words_b = set(b.lower().split())
  if not words_a or not words_b:
    return 0.0
  intersection = words_a & words_b
  return len(intersection) / min(len(words_a), len(words_b))
