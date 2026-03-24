"""Memory tools that the LLM calls to manage its own memory.

These are created as closures over the Memory instance so each
tool has access to the store. The agent never calls them directly —
the LLM decides when to use them during its response.
"""

from typing import TYPE_CHECKING, List, Optional

from definable.tool.function import Function

if TYPE_CHECKING:
  from definable.memory.v2.memory import Memory


def build_memory_tools(memory: "Memory", user_id: str, session_id: str) -> List[Function]:
  """Build all memory tools as closures over the memory instance.

  Called once per run in _prepare_tools_for_run. The user_id and session_id
  are captured from the run context.
  """
  store = memory.store
  max_chars = memory.working_memory_max_chars

  async def update_working_memory(content: str) -> str:
    """Rewrite the user's working memory scratchpad.

    Call this after turns where important context changed — user shared
    personal info, goals shifted, preferences were stated, or items
    became irrelevant.

    Args:
      content: Full replacement content (markdown). Not a diff — write the complete new state.
    """
    if len(content) > max_chars:
      wm = await store.set_working_memory(user_id, content[:max_chars])
      return (
        f"Updated (WARNING: truncated from {len(content)} to {max_chars} chars). Archive overflow items using archive_to_memory first, then trim."
      )
    wm = await store.set_working_memory(user_id, content)
    return f"Updated working memory ({len(content)} chars, version {wm.version})."

  async def archive_to_memory(summary: str, content: str, category: str = "conversation", tags: Optional[str] = None) -> str:
    """Store information in long-term memory with a one-line summary.

    Use when evicting from working memory, preserving conversation insights,
    or when the user explicitly asks you to remember something.

    Args:
      summary: Searchable one-liner with key terms. Use nouns, not abstractions.
      content: Full detail (markdown). Be specific and self-contained.
      category: One of: user, feedback, project, reference, conversation.
      tags: Comma-separated tags for filtering (optional).
    """
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    entry = await store.add_entry(
      user_id=user_id,
      summary=summary[:200],
      content=content,
      category=category,
      tags=tag_list,
      session_id=session_id,
    )
    return f"Archived (id={entry.id}, category={category}). Summary: {entry.summary}"

  async def recall_memory(query: Optional[str] = None, category: Optional[str] = None, limit: int = 20) -> str:
    """Search archived memory. Returns summaries only — use fetch_memory_entries for full details.

    Args:
      query: Search text (optional). Searches over summaries.
      category: Filter by category: user, feedback, project, reference, conversation (optional).
      limit: Max results to return (default 20).
    """
    entries = await store.search_index(user_id=user_id, query=query, category=category, limit=limit)
    if not entries:
      return "No memories found."
    lines = []
    for e in entries:
      tags_str = f" [{', '.join(e.tags)}]" if e.tags else ""
      lines.append(f"- [{e.id}] ({e.category}{tags_str}) {e.summary}")
    return f"Found {len(entries)} entries:\n" + "\n".join(lines)

  async def fetch_memory_entries(entry_ids: str) -> str:
    """Load full content of specific memory entries by ID.

    Call after recall_memory to get the details of entries you need.

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

    Use when the user asks to forget something or when you detect stale/incorrect info.

    Args:
      entry_id: The ID of the entry to delete.
    """
    deleted = await store.delete_entry(entry_id.strip())
    if deleted:
      return f"Deleted entry {entry_id}."
    return f"Entry {entry_id} not found."

  # Build Function objects using the auto-schema pattern
  tools = []
  fns = [update_working_memory, archive_to_memory, recall_memory, fetch_memory_entries, forget_memory]
  for fn in fns:
    f = Function(name=fn.__name__, entrypoint=fn)  # type: ignore[arg-type]
    f.process_entrypoint()
    tools.append(f)

  return tools
