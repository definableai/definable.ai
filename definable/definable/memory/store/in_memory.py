"""Pure-Python in-memory store implementing the MemoryStore protocol."""

import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from definable.memory.store._utils import cosine_similarity
from definable.memory.types import Episode, KnowledgeAtom, Procedure, TopicTransition
from definable.utils.log import log_debug


class InMemoryStore:
  """In-memory memory store backed by plain dicts.

  Useful for testing, development, and short-lived processes that do not
  require persistence.  All data is lost when the process exits.
  """

  def __init__(self) -> None:
    self._episodes: Dict[str, Episode] = {}
    self._atoms: Dict[str, KnowledgeAtom] = {}
    self._procedures: Dict[str, Procedure] = {}
    # Keyed by (user_id, from_topic, to_topic) -> count
    self._topic_transitions: Dict[Tuple[Optional[str], str, str], int] = {}
    self._initialized = False

  # --- Lifecycle ---

  async def initialize(self) -> None:
    if self._initialized:
      return
    self._initialized = True
    log_debug("InMemoryStore initialized", log_level=2)

  async def close(self) -> None:
    self._episodes.clear()
    self._atoms.clear()
    self._procedures.clear()
    self._topic_transitions.clear()
    self._initialized = False

  async def _ensure_initialized(self) -> None:
    if not self._initialized:
      await self.initialize()

  # --- Episodes ---

  async def store_episode(self, episode: Episode) -> str:
    await self._ensure_initialized()
    if not episode.id:
      episode.id = str(uuid4())
    now = time.time()
    if episode.created_at == 0.0:
      episode.created_at = now
    if episode.last_accessed_at == 0.0:
      episode.last_accessed_at = now

    self._episodes[episode.id] = deepcopy(episode)
    return episode.id

  async def get_episodes(
    self,
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = 50,
    min_stage: Optional[int] = None,
    max_stage: Optional[int] = None,
  ) -> List[Episode]:
    await self._ensure_initialized()
    results: List[Episode] = []

    for ep in self._episodes.values():
      if user_id is not None and ep.user_id != user_id:
        continue
      if session_id is not None and ep.session_id != session_id:
        continue
      if min_stage is not None and ep.compression_stage < min_stage:
        continue
      if max_stage is not None and ep.compression_stage > max_stage:
        continue
      results.append(deepcopy(ep))

    results.sort(key=lambda e: e.created_at, reverse=True)
    return results[:limit]

  async def update_episode(self, episode_id: str, **fields: Any) -> None:
    await self._ensure_initialized()
    if not fields:
      return
    ep = self._episodes.get(episode_id)
    if ep is None:
      return
    for key, value in fields.items():
      setattr(ep, key, value)

  async def get_episodes_for_distillation(self, stage: int, older_than: float) -> List[Episode]:
    await self._ensure_initialized()
    results: List[Episode] = []

    for ep in self._episodes.values():
      if ep.compression_stage == stage and ep.created_at < older_than:
        results.append(deepcopy(ep))

    results.sort(key=lambda e: e.created_at)
    return results[:50]

  # --- Knowledge Atoms ---

  async def store_atom(self, atom: KnowledgeAtom) -> str:
    await self._ensure_initialized()
    if not atom.id:
      atom.id = str(uuid4())
    now = time.time()
    if atom.created_at == 0.0:
      atom.created_at = now
    if atom.last_accessed_at == 0.0:
      atom.last_accessed_at = now

    self._atoms[atom.id] = deepcopy(atom)
    return atom.id

  async def get_atoms(
    self,
    *,
    user_id: Optional[str] = None,
    min_confidence: float = 0.1,
    limit: int = 50,
  ) -> List[KnowledgeAtom]:
    await self._ensure_initialized()
    results: List[KnowledgeAtom] = []

    for atom in self._atoms.values():
      if atom.confidence < min_confidence:
        continue
      if user_id is not None and atom.user_id != user_id:
        continue
      results.append(deepcopy(atom))

    results.sort(key=lambda a: a.last_accessed_at, reverse=True)
    return results[:limit]

  async def find_similar_atom(
    self,
    subject: str,
    predicate: str,
    user_id: Optional[str] = None,
  ) -> Optional[KnowledgeAtom]:
    await self._ensure_initialized()
    subject_lower = subject.lower()
    predicate_lower = predicate.lower()

    for atom in self._atoms.values():
      if user_id is not None and atom.user_id != user_id:
        continue
      if atom.subject.lower() == subject_lower and atom.predicate.lower() == predicate_lower:
        return deepcopy(atom)
    return None

  async def update_atom(self, atom_id: str, **fields: Any) -> None:
    await self._ensure_initialized()
    if not fields:
      return
    atom = self._atoms.get(atom_id)
    if atom is None:
      return
    for key, value in fields.items():
      setattr(atom, key, value)

  async def prune_atoms(self, min_confidence: float) -> int:
    await self._ensure_initialized()
    to_delete = [aid for aid, atom in self._atoms.items() if atom.confidence < min_confidence]
    for aid in to_delete:
      del self._atoms[aid]
    return len(to_delete)

  # --- Procedures ---

  async def store_procedure(self, procedure: Procedure) -> str:
    await self._ensure_initialized()
    if not procedure.id:
      procedure.id = str(uuid4())
    now = time.time()
    if procedure.created_at == 0.0:
      procedure.created_at = now
    if procedure.last_accessed_at == 0.0:
      procedure.last_accessed_at = now

    self._procedures[procedure.id] = deepcopy(procedure)
    return procedure.id

  async def get_procedures(
    self,
    *,
    user_id: Optional[str] = None,
    min_confidence: float = 0.3,
  ) -> List[Procedure]:
    await self._ensure_initialized()
    results: List[Procedure] = []

    for proc in self._procedures.values():
      if proc.confidence < min_confidence:
        continue
      if user_id is not None and proc.user_id != user_id:
        continue
      results.append(deepcopy(proc))

    results.sort(key=lambda p: p.confidence, reverse=True)
    return results

  async def find_similar_procedure(
    self,
    trigger: str,
    user_id: Optional[str] = None,
  ) -> Optional[Procedure]:
    await self._ensure_initialized()
    trigger_words = set(trigger.lower().split())

    # Get all procedures with a low confidence floor (same as SQLite impl)
    procedures = await self.get_procedures(user_id=user_id, min_confidence=0.1)
    best_match: Optional[Procedure] = None
    best_overlap = 0

    for proc in procedures:
      proc_words = set(proc.trigger.lower().split())
      overlap = len(trigger_words & proc_words)
      if overlap > best_overlap:
        best_overlap = overlap
        best_match = proc

    return best_match if best_overlap > 0 else None

  async def update_procedure(self, procedure_id: str, **fields: Any) -> None:
    await self._ensure_initialized()
    if not fields:
      return
    proc = self._procedures.get(procedure_id)
    if proc is None:
      return
    for key, value in fields.items():
      setattr(proc, key, value)

  # --- Topics ---

  async def store_topic_transition(
    self,
    from_topic: str,
    to_topic: str,
    user_id: Optional[str] = None,
  ) -> None:
    await self._ensure_initialized()
    key = (user_id, from_topic, to_topic)
    self._topic_transitions[key] = self._topic_transitions.get(key, 0) + 1

  async def get_topic_transitions(
    self,
    from_topic: str,
    user_id: Optional[str] = None,
    min_count: int = 3,
  ) -> List[TopicTransition]:
    await self._ensure_initialized()
    matching: List[Tuple[str, str, int]] = []

    for (uid, ft, tt), count in self._topic_transitions.items():
      if ft != from_topic:
        continue
      if user_id is not None and uid != user_id:
        continue
      if count < min_count:
        continue
      matching.append((ft, tt, count))

    # Sort by count descending
    matching.sort(key=lambda x: x[2], reverse=True)

    # Compute probabilities
    total = sum(m[2] for m in matching)
    transitions: List[TopicTransition] = []
    for ft, tt, count in matching:
      prob = count / total if total > 0 else 0.0
      transitions.append(TopicTransition(from_topic=ft, to_topic=tt, count=count, probability=prob))
    return transitions

  # --- Vector Search ---

  async def search_episodes_by_embedding(
    self,
    embedding: List[float],
    *,
    user_id: Optional[str] = None,
    top_k: int = 20,
  ) -> List[Episode]:
    await self._ensure_initialized()
    scored: List[Tuple[float, Episode]] = []

    for ep in self._episodes.values():
      if ep.embedding is None:
        continue
      if user_id is not None and ep.user_id != user_id:
        continue
      sim = cosine_similarity(embedding, ep.embedding)
      scored.append((sim, deepcopy(ep)))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [ep for _, ep in scored[:top_k]]

  async def search_atoms_by_embedding(
    self,
    embedding: List[float],
    *,
    user_id: Optional[str] = None,
    top_k: int = 20,
  ) -> List[KnowledgeAtom]:
    await self._ensure_initialized()
    scored: List[Tuple[float, KnowledgeAtom]] = []

    for atom in self._atoms.values():
      if atom.embedding is None:
        continue
      if user_id is not None and atom.user_id != user_id:
        continue
      sim = cosine_similarity(embedding, atom.embedding)
      scored.append((sim, deepcopy(atom)))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [atom for _, atom in scored[:top_k]]

  # --- Deletion ---

  async def delete_user_data(self, user_id: str) -> None:
    await self._ensure_initialized()
    self._episodes = {eid: ep for eid, ep in self._episodes.items() if ep.user_id != user_id}
    self._atoms = {aid: a for aid, a in self._atoms.items() if a.user_id != user_id}
    self._procedures = {pid: p for pid, p in self._procedures.items() if p.user_id != user_id}
    self._topic_transitions = {k: v for k, v in self._topic_transitions.items() if k[0] != user_id}

  async def delete_session_data(self, session_id: str) -> None:
    await self._ensure_initialized()
    self._episodes = {eid: ep for eid, ep in self._episodes.items() if ep.session_id != session_id}

  # --- Context manager ---

  async def __aenter__(self) -> "InMemoryStore":
    await self.initialize()
    return self

  async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    await self.close()
