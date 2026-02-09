"""Progressive distillation engine for memory compression.

Four stages:
  0 -> 1: Raw -> Summary (LLM call)
  1 -> 2: Summary -> Facts (LLM extracts SPO triples)
  2 -> 3: Facts -> Atoms (create/reinforce KnowledgeAtom records)
  3 -> archive: Merge duplicate atoms, prune low-confidence
"""

import contextlib
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional
from uuid import uuid4

from definable.memory.config import MemoryConfig
from definable.memory.topics import extract_topics
from definable.memory.types import Episode, KnowledgeAtom
from definable.utils.log import log_debug, log_warning

if TYPE_CHECKING:
  from definable.knowledge.embedders.base import Embedder


@dataclass
class DistillationResult:
  """Result of a distillation run."""

  episodes_processed: int = 0
  atoms_created: int = 0
  atoms_reinforced: int = 0
  atoms_pruned: int = 0


async def run_distillation(
  store,
  *,
  model=None,
  embedder: Optional["Embedder"] = None,
  user_id: Optional[str] = None,
  config: Optional[MemoryConfig] = None,
) -> DistillationResult:
  """Run progressive distillation across all stages.

  Args:
      store: MemoryStore instance.
      model: Optional model for LLM-based summarization (stages 0->1, 1->2).
      embedder: Optional embedder for computing atom embeddings.
      user_id: Optional user ID filter.
      config: Memory configuration.

  Returns:
      DistillationResult with counts of processed items.
  """
  cfg = config or MemoryConfig()
  now = time.time()
  result = DistillationResult()

  # Stage 0 -> 1: Raw -> Summary
  threshold_0 = now - cfg.distillation_stage_0_age
  episodes_0 = await store.get_episodes_for_distillation(stage=0, older_than=threshold_0)
  episodes_0 = episodes_0[: cfg.distillation_batch_size]

  for episode in episodes_0:
    summary = await _summarize_episode(episode, model)
    if summary:
      await store.update_episode(episode.id, content=summary, compression_stage=1)
      result.episodes_processed += 1

  # Stage 1 -> 2: Summary -> Facts
  threshold_1 = now - cfg.distillation_stage_1_age
  episodes_1 = await store.get_episodes_for_distillation(stage=1, older_than=threshold_1)
  episodes_1 = episodes_1[: cfg.distillation_batch_size]

  for episode in episodes_1:
    facts = await _extract_facts(episode, model)
    if facts:
      content = "; ".join(facts)
      await store.update_episode(episode.id, content=content, compression_stage=2)
      result.episodes_processed += 1

  # Stage 2 -> 3: Facts -> Atoms
  threshold_2 = now - cfg.distillation_stage_2_age
  episodes_2 = await store.get_episodes_for_distillation(stage=2, older_than=threshold_2)
  episodes_2 = episodes_2[: cfg.distillation_batch_size]

  for episode in episodes_2:
    atoms_result = await _create_atoms_from_facts(
      store,
      episode,
      embedder=embedder,
      config=cfg,
    )
    result.atoms_created += atoms_result["created"]
    result.atoms_reinforced += atoms_result["reinforced"]
    await store.update_episode(episode.id, compression_stage=3)
    result.episodes_processed += 1

  # Stage 3 -> archive: Prune low-confidence atoms
  threshold_3 = now - cfg.distillation_stage_3_age
  episodes_3 = await store.get_episodes_for_distillation(stage=3, older_than=threshold_3)
  if episodes_3:
    pruned = await store.prune_atoms(min_confidence=0.1)
    result.atoms_pruned = pruned

  log_debug(
    f"Distillation complete: {result.episodes_processed} episodes, "
    f"{result.atoms_created} atoms created, {result.atoms_reinforced} reinforced, "
    f"{result.atoms_pruned} pruned",
    log_level=2,
  )
  return result


async def _summarize_episode(episode: Episode, model=None) -> Optional[str]:
  """Summarize a raw episode using the distillation model.

  Falls back to simple truncation if no model is available.
  """
  if model:
    try:
      from definable.models.message import Message

      messages = [
        Message(role="system", content="Summarize the following conversation turn in 1-2 concise sentences. Preserve key facts and intent."),
        Message(role="user", content=f"[{episode.role}]: {episode.content}"),
      ]
      assistant_msg = Message(role="assistant")
      response = await model.ainvoke(messages=messages, assistant_message=assistant_msg)
      if response.content:
        return response.content
    except Exception as e:
      log_warning(f"Distillation model call failed for summary: {e}")

  # Fallback: truncate to first 200 chars
  content = episode.content
  if len(content) > 200:
    return content[:197] + "..."
  return content


async def _extract_facts(episode: Episode, model=None) -> Optional[List[str]]:
  """Extract factual statements from a summary.

  Returns a list of fact strings (subject-predicate-object form).
  Falls back to splitting on sentence boundaries.
  """
  if model:
    try:
      from definable.models.message import Message

      messages = [
        Message(
          role="system",
          content=(
            "Extract key facts from the following text as a list of simple statements. "
            "Each fact should be in the form 'subject predicate object' (e.g., 'user lives in San Francisco'). "
            "Return one fact per line. If there are no clear facts, return 'NO_FACTS'."
          ),
        ),
        Message(role="user", content=episode.content),
      ]
      assistant_msg = Message(role="assistant")
      response = await model.ainvoke(messages=messages, assistant_message=assistant_msg)
      if response.content and response.content.strip() != "NO_FACTS":
        lines = [line.strip().lstrip("- ").strip() for line in response.content.strip().split("\n")]
        return [line for line in lines if line and len(line) > 5]
    except Exception as e:
      log_warning(f"Distillation model call failed for fact extraction: {e}")

  # Fallback: split on periods, filter short segments
  sentences = [s.strip() for s in episode.content.split(".") if s.strip()]
  return [s for s in sentences if len(s) > 10]


async def _create_atoms_from_facts(
  store,
  episode: Episode,
  *,
  embedder: Optional["Embedder"] = None,
  config: Optional[MemoryConfig] = None,
) -> dict:
  """Convert fact strings in an episode's content into KnowledgeAtom records.

  If a matching atom (same subject+predicate) exists, reinforce it instead.
  """
  cfg = config or MemoryConfig()
  facts = [f.strip() for f in episode.content.split(";") if f.strip()]

  created = 0
  reinforced = 0
  now = time.time()

  for fact in facts:
    # Parse fact into subject/predicate/object
    parts = _parse_spo(fact)
    if not parts:
      continue
    subject, predicate, obj = parts

    # Check if similar atom exists
    existing = await store.find_similar_atom(subject, predicate, user_id=episode.user_id)

    if existing:
      # Reinforce existing atom
      new_confidence = min(1.0, existing.confidence + cfg.reinforcement_boost)
      new_count = existing.reinforcement_count + 1
      source_ids = list(existing.source_episode_ids or [])
      if episode.id not in source_ids:
        source_ids.append(episode.id)
      await store.update_atom(
        existing.id,
        confidence=new_confidence,
        reinforcement_count=new_count,
        last_reinforced_at=now,
        source_episode_ids=source_ids,
      )
      reinforced += 1
    else:
      # Create new atom
      topics = extract_topics(fact)
      embedding = None
      if embedder:
        with contextlib.suppress(Exception):
          embedding = await embedder.async_get_embedding(fact)

      from definable.tokens import count_text_tokens

      atom = KnowledgeAtom(
        id=str(uuid4()),
        user_id=episode.user_id,
        subject=subject,
        predicate=predicate,
        object=obj,
        content=fact,
        embedding=embedding,
        confidence=0.5,
        topics=topics,
        token_count=count_text_tokens(fact),
        source_episode_ids=[episode.id],
        created_at=now,
        last_accessed_at=now,
      )
      await store.store_atom(atom)
      created += 1

  return {"created": created, "reinforced": reinforced}


def _parse_spo(fact: str) -> Optional[tuple]:
  """Parse a fact string into (subject, predicate, object).

  Uses simple heuristic: first word is subject, second is predicate, rest is object.
  Falls back if the fact is too short.
  """
  words = fact.strip().split()
  if len(words) < 3:
    return None

  subject = words[0]
  predicate = words[1]
  obj = " ".join(words[2:])
  return (subject, predicate, obj)
