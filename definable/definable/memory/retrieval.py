"""Retrieval pipeline: parallel search, scoring, and token budget packing."""

import asyncio
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union

from definable.memory.config import MemoryConfig
from definable.memory.scorer import ScoredMemory, score_candidate
from definable.memory.topics import extract_topics, get_transition_map, predict_next_topics
from definable.memory.types import Episode, KnowledgeAtom, MemoryPayload, Procedure
from definable.utils.log import log_debug

if TYPE_CHECKING:
  from definable.knowledge.embedders.base import Embedder


async def recall_memories(
  store,
  query: str,
  *,
  token_budget: int = 500,
  embedder: Optional["Embedder"] = None,
  user_id: Optional[str] = None,
  session_id: Optional[str] = None,
  config: Optional[MemoryConfig] = None,
) -> Optional[MemoryPayload]:
  """Run the full recall pipeline: retrieve, score, pack, format.

  Args:
      store: MemoryStore instance.
      query: The user's query to match against.
      token_budget: Maximum tokens for the memory payload.
      embedder: Optional embedder for semantic search.
      user_id: Optional user ID filter.
      session_id: Optional session ID for recent context.
      config: Memory configuration.

  Returns:
      MemoryPayload with formatted context, or None if no relevant memories.
  """
  cfg = config or MemoryConfig()

  # 1. Extract topics from query
  current_topics = extract_topics(query)
  log_debug(f"Memory recall topics: {current_topics}", log_level=2)

  # 2. Build transition map and predict next topics
  transition_map: Dict[str, list] = {}
  predicted_topics: List[str] = []
  if current_topics:
    transition_map = await get_transition_map(
      store,
      current_topics,
      user_id=user_id,
      min_count=cfg.topic_transition_min_count,
    )
    predicted_topics = predict_next_topics(
      current_topics,
      transition_map,
      min_probability=cfg.topic_transition_min_probability,
    )

  # 3. Compute query embedding if embedder available
  query_embedding: Optional[List[float]] = None
  if embedder:
    try:
      query_embedding = await embedder.async_get_embedding(query)
    except Exception:
      log_debug("Failed to compute query embedding for memory recall", log_level=2)

  # 4. Parallel retrieval (5 paths)
  candidates: List[Union[Episode, KnowledgeAtom, Procedure]] = []

  tasks = []

  # Path 1: Vector search over atoms
  if query_embedding:
    tasks.append(_search_atoms_by_embedding(store, query_embedding, user_id, cfg.retrieval_top_k))
  # Path 2: Vector search over episodes
  if query_embedding:
    tasks.append(_search_episodes_by_embedding(store, query_embedding, user_id, cfg.retrieval_top_k))
  # Path 3: Topic-based atom lookup
  tasks.append(_get_atoms_by_topic(store, current_topics + predicted_topics, user_id))
  # Path 4: Procedure matching
  tasks.append(_get_matching_procedures(store, query, user_id))
  # Path 5: Recent session episodes
  if session_id:
    tasks.append(_get_recent_session_episodes(store, session_id, cfg.recent_episodes_limit))

  results = await asyncio.gather(*tasks, return_exceptions=True)

  seen_ids: set = set()
  for result in results:
    if isinstance(result, Exception):
      log_debug(f"Memory retrieval path failed: {result}", log_level=2)
      continue
    for item in result:
      item_id = getattr(item, "id", None)
      if item_id and item_id not in seen_ids:
        seen_ids.add(item_id)
        candidates.append(item)

  if not candidates:
    return None

  # 5. Score all candidates
  max_access = max((_get_access_count(c) for c in candidates), default=1)
  predicted_set: Set[str] = set(predicted_topics)

  scored: List[ScoredMemory] = []
  for candidate in candidates:
    candidate_embedding = getattr(candidate, "embedding", None)
    s = score_candidate(
      candidate,
      query_embedding=query_embedding,
      candidate_embedding=candidate_embedding,
      predicted_topics=predicted_set,
      max_access_count=max_access,
      config=cfg,
    )
    token_count = _get_token_count(candidate)
    formatted = _format_candidate(candidate)
    scored.append(ScoredMemory(memory=candidate, score=s, token_count=token_count, formatted=formatted))

  # 6. Token budget packing (greedy knapsack, sorted by score)
  scored.sort(key=lambda x: x.score, reverse=True)

  selected: List[ScoredMemory] = []
  tokens_used = 0
  for item in scored:
    # Estimate token count: approximate 4 chars per token if not pre-counted
    tc = item.token_count if item.token_count > 0 else max(1, len(item.formatted) // 4)
    if tokens_used + tc <= token_budget:
      selected.append(item)
      tokens_used += tc

  if not selected:
    return None

  # 7. Format payload
  context = _format_payload(selected)

  # Update access timestamps (fire-and-forget)
  now = time.time()
  for item in selected:
    mem = item.memory
    mem_id = getattr(mem, "id", None)
    if not mem_id:
      continue
    try:
      if isinstance(mem, Episode):
        await store.update_episode(mem_id, last_accessed_at=now, access_count=(mem.access_count or 0) + 1)
      elif isinstance(mem, KnowledgeAtom):
        await store.update_atom(mem_id, last_accessed_at=now, access_count=(mem.access_count or 0) + 1)
      elif isinstance(mem, Procedure):
        await store.update_procedure(mem_id, last_accessed_at=now)
    except Exception:
      pass  # Access tracking is best-effort

  return MemoryPayload(
    context=context,
    tokens_used=tokens_used,
    chunks_included=len(selected),
    chunks_available=len(candidates),
  )


# --- Retrieval helpers ---


async def _search_atoms_by_embedding(store, embedding, user_id, top_k):
  return await store.search_atoms_by_embedding(embedding, user_id=user_id, top_k=top_k)


async def _search_episodes_by_embedding(store, embedding, user_id, top_k):
  return await store.search_episodes_by_embedding(embedding, user_id=user_id, top_k=top_k)


async def _get_atoms_by_topic(store, topics, user_id):
  """Get atoms matching any of the given topics."""
  atoms = await store.get_atoms(user_id=user_id, limit=100)
  matched = []
  topic_set = set(topics)
  for atom in atoms:
    if topic_set & set(atom.topics or []):
      matched.append(atom)
  return matched


async def _get_matching_procedures(store, query, user_id):
  """Get procedures matching the query."""
  proc = await store.find_similar_procedure(query, user_id=user_id)
  return [proc] if proc else []


async def _get_recent_session_episodes(store, session_id, limit):
  return await store.get_episodes(session_id=session_id, limit=limit)


def _get_access_count(candidate) -> int:
  return getattr(candidate, "access_count", 0) or 0


def _get_token_count(candidate) -> int:
  return getattr(candidate, "token_count", 0) or 0


def _format_candidate(candidate: Union[Episode, KnowledgeAtom, Procedure]) -> str:
  """Format a single memory candidate as a human-readable string."""
  if isinstance(candidate, KnowledgeAtom):
    conf = candidate.confidence
    return f"- {candidate.content} (confidence: {conf:.2f})"
  elif isinstance(candidate, Procedure):
    return f"- When {candidate.trigger}: {candidate.action}"
  elif isinstance(candidate, Episode):
    role = candidate.role
    content = candidate.content
    # Truncate long episode content
    if len(content) > 200:
      content = content[:197] + "..."
    return f"- [{role}] {content}"
  return f"- {candidate}"


def _format_payload(selected: List[ScoredMemory]) -> str:
  """Format selected memories into structured XML sections."""
  atoms: List[str] = []
  procedures: List[str] = []
  episodes: List[str] = []

  for item in selected:
    mem = item.memory
    if isinstance(mem, KnowledgeAtom):
      # Separate high-confidence atoms as "user profile" vs general "known facts"
      if mem.confidence >= 0.8:
        atoms.insert(0, item.formatted)  # High confidence first
      else:
        atoms.append(item.formatted)
    elif isinstance(mem, Procedure):
      procedures.append(item.formatted)
    elif isinstance(mem, Episode):
      episodes.append(item.formatted)

  sections = []
  if atoms:
    high_conf = [a for a in atoms if "(confidence: 0.8" in a or "(confidence: 0.9" in a or "(confidence: 1.0" in a]
    low_conf = [a for a in atoms if a not in high_conf]
    if high_conf:
      sections.append("  <user_profile>\n" + "\n".join(f"    {a}" for a in high_conf) + "\n  </user_profile>")
    if low_conf:
      sections.append("  <known_facts>\n" + "\n".join(f"    {a}" for a in low_conf) + "\n  </known_facts>")
  if episodes:
    sections.append("  <recent_context>\n" + "\n".join(f"    {a}" for a in episodes) + "\n  </recent_context>")
  if procedures:
    sections.append("  <preferences>\n" + "\n".join(f"    {a}" for a in procedures) + "\n  </preferences>")

  if not sections:
    return ""

  return "<memory_context>\n" + "\n".join(sections) + "\n</memory_context>"
