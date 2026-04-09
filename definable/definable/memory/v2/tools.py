"""Memory tools that the LLM calls to manage its own memory.

These are created as closures over the Memory instance so each
tool has access to the store. The agent never calls them directly —
the LLM decides when to use them during its response.
"""

from typing import TYPE_CHECKING, Any, List, Optional

from definable.tool.function import Function

if TYPE_CHECKING:
  from definable.memory.v2.memory import Memory

# Summaries matching these patterns are too vague to be useful
_VAGUE_PATTERNS = [
  "'s identity",
  "'s background",
  "'s preferences",
  "activities and interests",
  "professional details",
  "personal information",
  "technical background",
  "work details",
]

_REQUIRED_SECTIONS = ["## Identity", "## Preferences", "## Projects", "## Team", "## Other"]


def _strip_text(value: Any) -> str:
  if value is None:
    return ""
  return str(value).strip()


def _first_fact_line(content: str) -> str:
  for raw_line in content.splitlines():
    line = raw_line.strip()
    if not line:
      continue
    if line.lower().startswith("fact:"):
      return line[5:].strip()
    if line.lower().startswith("why:") or line.lower().startswith("how to apply:"):
      continue
    return line
  return ""


def _normalize_tags(tags: Any) -> list[str]:
  if tags is None:
    return []
  if isinstance(tags, str):
    return [tag.strip() for tag in tags.split(",") if tag.strip()]
  if isinstance(tags, (list, tuple, set)):
    return [_strip_text(tag) for tag in tags if _strip_text(tag)]
  tag = _strip_text(tags)
  return [tag] if tag else []


def build_memory_tools(memory: "Memory", user_id: str, session_id: str) -> List[Function]:
  """Build all memory tools as closures over the memory instance.

  Called once per run in _prepare_tools_for_run. The user_id and session_id
  are captured from the run context.
  """
  store = memory.store
  max_chars = memory.working_memory_max_chars
  warm_max = memory.warm_memory_max_chars

  async def update_working_memory(content: Optional[str] = None) -> str:
    """Rewrite the user's working memory scratchpad.

    CRITICAL: Include ALL existing facts plus any new information. Copy all
    sections from the current working memory, then add/modify only what changed.
    Never drop existing information — working memory must be comprehensive.

    The scratchpad uses a fixed section template:
      ## Identity | ## Preferences | ## Projects | ## Team | ## Other

    Args:
      content: Full replacement content (markdown). Not a diff — write the
        complete new state with all sections preserved.
    """
    content = _strip_text(content)
    if not content:
      return "REJECTED: Missing required `content` argument. Pass the full working-memory markdown as `content`."

    current_wm = await store.get_working_memory(user_id)
    current_content = current_wm.content if current_wm else ""
    prev_len = len(current_content)

    # Validate: collapse guard
    if prev_len > 200 and len(content) < prev_len * 0.4 and len(content) < 200:
      return (
        f"REJECTED: New content ({len(content)} chars) is dramatically smaller than "
        f"current ({prev_len} chars). This looks like working memory collapse — "
        f"you are losing information. Rewrite to include ALL existing facts "
        f"plus any new information. Current working memory:\n{current_content[:500]}"
      )

    # Validate: section template (only enforce after WM has been established)
    if prev_len > 200:
      missing = [s for s in _REQUIRED_SECTIONS if s not in content]
      if missing:
        return (
          f"REJECTED: Missing required sections: {missing}. Include all 5 sections (## Identity, ## Preferences, ## Projects, ## Team, ## Other)."
        )

    # Handle overflow: move excess to warm memory
    if len(content) > max_chars:
      # Split by sections, move lowest-priority to warm
      hot, warm_overflow = _split_by_priority(content, max_chars)
      if warm_overflow:
        existing_warm = await store.get_warm_memory(user_id)
        warm_content = (existing_warm.content if existing_warm else "") + "\n\n" + warm_overflow
        await store.set_warm_memory(user_id, warm_content[:warm_max])
      wm = await store.set_working_memory(user_id, hot, session_id=session_id)
      return (
        f"Updated working memory ({len(hot)} chars, version {wm.version}). "
        f"{len(warm_overflow)} chars moved to extended memory. Previous was {prev_len} chars."
      )

    wm = await store.set_working_memory(user_id, content, session_id=session_id)
    return f"Updated working memory ({len(content)} chars, version {wm.version}). Previous was {prev_len} chars."

  async def archive_to_memory(
    summary: Optional[str] = None,
    content: Optional[str] = None,
    category: Optional[str] = "conversation",
    tags: Optional[str] = None,
    source: Optional[str] = "user_stated",
  ) -> str:
    """Store ONE fact in long-term memory with a searchable summary.

    Archive one fact per call. If the user shared 3 facts, make 3 separate calls.

    The summary is what search indexes — it MUST contain the specific nouns and
    values from the fact, not vague descriptions.
      GOOD: "Alice graduated from MIT in 2016 with a CS degree"
      BAD:  "Alice's educational background"

    Args:
      summary: Searchable one-liner with key nouns and values. Must be specific
        enough that someone searching for any key term would find this entry.
      content: Full detail (markdown). Structure: fact, then Why: and How to apply: lines.
      category: One of: user, feedback, project, reference, conversation.
      tags: Comma-separated key nouns for filtering (e.g., "MIT,education,2016,CS").
      source: Who asserted this: "user_stated" (user said it directly),
        "user_implied" (inferred from context), "agent_observed" (your conclusion).
    """
    summary = _strip_text(summary)
    content = _strip_text(content)
    category = _strip_text(category) or "conversation"
    source = _strip_text(source) or "user_stated"

    adjustments: list[str] = []

    if not summary and content:
      summary = _first_fact_line(content)[:200]
      if summary:
        adjustments.append("derived summary from content")

    if not content and summary:
      content = (
        f"Fact: {summary}\n"
        "Why: Preserve this concrete fact for future recall.\n"
        "How to apply: Use it when answering later questions about the user or conversation."
      )
      adjustments.append("filled missing content from summary")

    if not summary:
      return "REJECTED: Missing required `summary` argument. Provide a concrete searchable fact summary."

    if not content:
      return "REJECTED: Missing required `content` argument. Include Fact/Why/How to apply details."

    candidate_summary = _first_fact_line(content)
    if candidate_summary and len(summary.split()) < 4 and len(candidate_summary.split()) >= 4:
      summary = candidate_summary[:200]
      adjustments.append("expanded short summary from content")

    # Validate summary quality
    summary_lower = summary.lower()
    for pattern in _VAGUE_PATTERNS:
      if pattern in summary_lower:
        if candidate_summary and candidate_summary.lower() != summary_lower:
          summary = candidate_summary[:200]
          summary_lower = summary.lower()
          adjustments.append("replaced vague summary from content")
          break
        return (
          f"REJECTED: Summary too vague ('{summary}'). "
          f"Use specific nouns and values. Example: 'Alice graduated from MIT in 2016' not 'Alice's background'."
        )
    if len(summary.split()) < 4:
      if candidate_summary and candidate_summary != summary and len(candidate_summary.split()) >= len(summary.split()):
        summary = candidate_summary[:200]
        adjustments.append("replaced short summary from content")
      elif summary:
        summary = f"Conversation fact: {summary}"[:200]
        adjustments.append("prefixed short summary")

    confidence = 1.0 if source == "user_stated" else 0.7 if source == "user_implied" else 0.5
    tag_list = _normalize_tags(tags)
    entry = await store.add_entry(
      user_id=user_id,
      summary=summary[:200],
      content=content,
      category=category,
      tags=tag_list,
      session_id=session_id,
      confidence=confidence,
      source=source,
    )

    # Enforce entry limit after adding
    await store.enforce_limit(user_id, memory.index_max_entries)

    adjustment_note = f" Adjustments: {', '.join(adjustments)}." if adjustments else ""
    return f"Archived (id={entry.id}, category={category}, confidence={confidence}). Summary: {entry.summary}.{adjustment_note}"

  async def recall_memory(query: Optional[str] = None, category: Optional[str] = None, limit: int = 20, timeframe: Optional[str] = None) -> str:
    """Search archived memory. Returns summaries with content previews.

    Searches across summaries, tags, AND full content with recency-weighted
    ranking. Use specific nouns (names, places, technologies) for best results.
    If first search returns nothing, try alternative terms.

    Args:
      query: Search text (optional). Use specific keywords.
      category: Filter by category: user, feedback, project, reference, conversation (optional).
      limit: Max results to return (default 20).
      timeframe: Optional. "recent" (last 30 days), "last_quarter" (90 days), or "YYYY-MM" for a specific month.
    """
    from datetime import datetime, timedelta, timezone

    after = None
    before = None
    if timeframe:
      now = datetime.now(timezone.utc)
      if timeframe == "recent":
        after = now - timedelta(days=30)
      elif timeframe == "last_quarter":
        after = now - timedelta(days=90)
      elif len(timeframe) == 7 and "-" in timeframe:  # YYYY-MM format
        try:
          year, month = int(timeframe[:4]), int(timeframe[5:])
          after = datetime(year, month, 1, tzinfo=timezone.utc)
          if month == 12:
            before = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
          else:
            before = datetime(year, month + 1, 1, tzinfo=timezone.utc)
        except (ValueError, IndexError):
          pass

    entries = await store.search_index(user_id=user_id, query=query, category=category, limit=limit, after=after, before=before)
    if not entries:
      return "No memories found. Try different or broader search terms."

    # Fetch content snippets
    entry_ids = [e.id for e in entries]
    full_entries = await store.get_entries(entry_ids)
    content_map = {e.id: e.content[:120] for e in full_entries}

    lines = []
    for e in entries:
      tags_str = f" [{', '.join(e.tags)}]" if e.tags else ""
      conf_str = f" (conf={e.confidence:.1f})" if e.confidence < 1.0 else ""
      snippet = content_map.get(e.id, "")
      snippet_str = f"\n      Preview: {snippet}..." if snippet else ""
      lines.append(f"- [{e.id}] ({e.category}{tags_str}{conf_str}) {e.summary}{snippet_str}")
    return f"Found {len(entries)} entries:\n" + "\n".join(lines)

  async def fetch_memory_entries(entry_ids: str) -> str:
    """Load full content of specific memory entries by ID.

    Call after recall_memory to get the full details of entries you need.

    Args:
      entry_ids: Comma-separated entry IDs to fetch.
    """
    ids = [i.strip() for i in entry_ids.split(",") if i.strip()]
    if not ids:
      return "No entry IDs provided."
    entries = await store.get_entries(ids)
    if not entries:
      return "No entries found for the given IDs."
    parts = []
    for e in entries:
      parts.append(f"### [{e.id}] ({e.category})\n{e.content}")
    return "\n\n".join(parts)

  async def forget_memory(entry_id: str) -> str:
    """Delete a specific memory entry.

    Use when the user asks to forget something, when you detect stale/incorrect
    info, or when cleaning up old entries after a correction (Rule 11).

    Args:
      entry_id: The ID of the entry to delete.
    """
    deleted = await store.delete_entry(entry_id.strip())
    if deleted:
      return f"Deleted entry {entry_id}."
    return f"Entry {entry_id} not found."

  async def read_extended_memory() -> str:
    """Read the extended (warm) memory tier — overflow context not loaded by default.

    Call this when working memory doesn't have enough context and recall_memory
    isn't finding what you need. Extended memory contains facts that overflowed
    from working memory.
    """
    warm = await store.get_warm_memory(user_id)
    if warm and warm.content.strip():
      return f"Extended memory ({len(warm.content)} chars):\n{warm.content}"
    return "No extended memory stored."

  async def refresh_preferences() -> str:
    """Check for stale preferences that haven't been referenced recently.

    Call periodically to identify preferences that may be outdated.
    """
    entries = await store.search_index(user_id, category="feedback", limit=50)
    if not entries:
      return "No preferences stored."
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    stale = []
    for e in entries:
      ref_date = e.last_accessed_at or e.created_at
      days = (now - ref_date).days
      if days > 60:
        stale.append(f"- [{e.id}] {e.summary} (last referenced {days} days ago)")
    if stale:
      return f"{len(stale)} preferences may be outdated:\n" + "\n".join(stale)
    return "All preferences appear current."

  # Build Function objects
  tools = []
  fns = [
    update_working_memory,
    archive_to_memory,
    recall_memory,
    fetch_memory_entries,
    forget_memory,
    read_extended_memory,
    refresh_preferences,
  ]
  for fn in fns:
    f = Function(name=fn.__name__, entrypoint=fn)  # type: ignore[arg-type]
    f.process_entrypoint()
    tools.append(f)

  return tools


def _split_by_priority(content: str, max_chars: int) -> tuple[str, str]:
  """Split WM content into hot (high priority) and warm (overflow).

  Priority order: Identity > Projects > Preferences > Team > Other
  """
  sections: dict[str, str] = {}
  current_section = ""
  current_lines: list[str] = []

  for line in content.split("\n"):
    if line.startswith("## "):
      if current_section:
        sections[current_section] = "\n".join(current_lines)
      current_section = line
      current_lines = [line]
    else:
      current_lines.append(line)
  if current_section:
    sections[current_section] = "\n".join(current_lines)

  # Priority order — first sections are kept in hot
  priority = ["## Identity", "## Projects", "## Preferences", "## Team", "## Other"]

  hot_parts: list[str] = []
  warm_parts: list[str] = []
  hot_len = 0

  for section_name in priority:
    section_content = sections.get(section_name, f"{section_name}\n")
    if hot_len + len(section_content) <= max_chars:
      hot_parts.append(section_content)
      hot_len += len(section_content)
    else:
      warm_parts.append(section_content)

  return "\n\n".join(hot_parts), "\n\n".join(warm_parts)
