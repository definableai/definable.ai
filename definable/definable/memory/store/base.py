"""MemoryStore protocol — structural subtyping for memory backends."""

from typing import List, Optional, Protocol, runtime_checkable

from definable.memory.types import Episode, KnowledgeAtom, Procedure, TopicTransition


@runtime_checkable
class MemoryStore(Protocol):
  """Protocol for memory storage backends.

  Implementations must provide async methods for storing and retrieving
  episodes, knowledge atoms, procedures, and topic transitions.
  """

  # Lifecycle

  async def initialize(self) -> None: ...

  async def close(self) -> None: ...

  # Episodes

  async def store_episode(self, episode: Episode) -> str: ...

  async def get_episodes(
    self,
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = 50,
    min_stage: Optional[int] = None,
    max_stage: Optional[int] = None,
  ) -> List[Episode]: ...

  async def update_episode(self, episode_id: str, **fields) -> None: ...

  async def get_episodes_for_distillation(self, stage: int, older_than: float) -> List[Episode]: ...

  # Knowledge Atoms

  async def store_atom(self, atom: KnowledgeAtom) -> str: ...

  async def get_atoms(
    self,
    *,
    user_id: Optional[str] = None,
    min_confidence: float = 0.1,
    limit: int = 50,
  ) -> List[KnowledgeAtom]: ...

  async def find_similar_atom(
    self,
    subject: str,
    predicate: str,
    user_id: Optional[str] = None,
  ) -> Optional[KnowledgeAtom]: ...

  async def update_atom(self, atom_id: str, **fields) -> None: ...

  async def prune_atoms(self, min_confidence: float) -> int: ...

  # Procedures

  async def store_procedure(self, procedure: Procedure) -> str: ...

  async def get_procedures(
    self,
    *,
    user_id: Optional[str] = None,
    min_confidence: float = 0.3,
  ) -> List[Procedure]: ...

  async def find_similar_procedure(
    self,
    trigger: str,
    user_id: Optional[str] = None,
  ) -> Optional[Procedure]: ...

  async def update_procedure(self, procedure_id: str, **fields) -> None: ...

  # Topics

  async def store_topic_transition(
    self,
    from_topic: str,
    to_topic: str,
    user_id: Optional[str] = None,
  ) -> None: ...

  async def get_topic_transitions(
    self,
    from_topic: str,
    user_id: Optional[str] = None,
    min_count: int = 3,
  ) -> List[TopicTransition]: ...

  # Vector search

  async def search_episodes_by_embedding(
    self,
    embedding: List[float],
    *,
    user_id: Optional[str] = None,
    top_k: int = 20,
  ) -> List[Episode]: ...

  async def search_atoms_by_embedding(
    self,
    embedding: List[float],
    *,
    user_id: Optional[str] = None,
    top_k: int = 20,
  ) -> List[KnowledgeAtom]: ...

  # Deletion

  async def delete_user_data(self, user_id: str) -> None: ...

  async def delete_session_data(self, session_id: str) -> None: ...
