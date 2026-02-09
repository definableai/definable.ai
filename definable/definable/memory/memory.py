"""CognitiveMemory — main entry point for the memory system."""

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, List, Optional
from uuid import uuid4

from definable.memory.config import MemoryConfig
from definable.memory.distillation import DistillationResult, run_distillation
from definable.memory.retrieval import recall_memories
from definable.memory.topics import extract_topics
from definable.memory.types import Episode, MemoryPayload
from definable.utils.log import log_debug, log_warning

if TYPE_CHECKING:
  from definable.knowledge.embedders.base import Embedder
  from definable.memory.store.base import MemoryStore
  from definable.models.base import Model


class CognitiveMemory:
  """Multi-tier cognitive memory with token-budget-aware retrieval.

  Provides:
    - Automatic recall of relevant memories before model calls
    - Automatic storage of new conversation turns
    - Progressive distillation (raw -> summary -> facts -> atoms)
    - Predictive pre-loading via topic transition model
    - Token budget optimization for context injection

  Example:
      from definable.memory import CognitiveMemory, SQLiteMemoryStore

      memory = CognitiveMemory(
          store=SQLiteMemoryStore("./memory.db"),
          token_budget=500,
      )

      # Used by Agent automatically:
      # agent = Agent(model=model, memory=memory)
  """

  def __init__(
    self,
    store: "MemoryStore",
    token_budget: int = 500,
    embedder: Optional["Embedder"] = None,
    distillation_model: Optional["Model"] = None,
    config: Optional[MemoryConfig] = None,
  ):
    """Initialize CognitiveMemory.

    Args:
        store: Backend storage (e.g. SQLiteMemoryStore("./memory.db")).
            The store is initialized lazily on the first recall/store call.
        token_budget: Maximum tokens of memory context to inject into the
            system prompt. Higher values provide more context but use more
            of the model's context window. Default: 500.
        embedder: Optional embedder for semantic (vector) search over
            memories. When provided, enables the two vector-search retrieval
            paths. Without it, retrieval falls back to topic matching and
            recency. Example: OpenAIEmbedder(id="text-embedding-3-small").
        distillation_model: Optional model for progressive distillation
            (summarizing old episodes, extracting facts). Without it,
            distillation uses simple truncation and heuristic extraction.
        config: Fine-tune scoring weights, decay rates, distillation
            thresholds, and retrieval parameters. See MemoryConfig for
            defaults.
    """
    self.store = store
    self.token_budget = token_budget
    self.embedder = embedder
    self.distillation_model = distillation_model
    self.config = config or MemoryConfig()
    self._initialized = False

  async def _ensure_initialized(self) -> None:
    if not self._initialized:
      await self.store.initialize()
      self._initialized = True

  async def recall(
    self,
    query: str,
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
  ) -> Optional[MemoryPayload]:
    """Retrieve relevant memories for the given query.

    Runs the full recall pipeline: topic extraction, parallel retrieval,
    scoring, budget packing, and formatting.

    Args:
        query: The user's query to match against.
        user_id: Optional user ID for filtering.
        session_id: Optional session ID for recent context.

    Returns:
        MemoryPayload with formatted context string, or None.
    """
    try:
      await self._ensure_initialized()
      return await recall_memories(
        self.store,
        query,
        token_budget=self.token_budget,
        embedder=self.embedder,
        user_id=user_id,
        session_id=session_id,
        config=self.config,
      )
    except Exception as e:
      log_warning(f"Memory recall failed (non-fatal): {e}")
      return None

  async def store_messages(
    self,
    messages: List,
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
  ) -> None:
    """Store conversation messages as episodes.

    Extracts topics, computes embeddings (if available), computes sentiment,
    stores episodes, updates topic transitions, and schedules distillation.

    Args:
        messages: List of Message objects from the conversation turn.
        user_id: Optional user ID.
        session_id: Optional session ID.
    """
    try:
      await self._ensure_initialized()
      session_id = session_id or str(uuid4())
      now = time.time()

      prev_topics: List[str] = []

      for msg in messages:
        role = getattr(msg, "role", "user")
        content = getattr(msg, "content", None)
        if not content or role not in ("user", "assistant"):
          continue

        # Extract topics
        topics = extract_topics(content)

        # Compute embedding if available
        embedding = None
        if self.embedder:
          with contextlib.suppress(Exception):
            embedding = await self.embedder.async_get_embedding(content)

        # Compute simple sentiment
        sentiment = _compute_sentiment(content)

        # Count tokens
        from definable.tokens import count_text_tokens

        token_count = count_text_tokens(content)

        # Create and store episode
        episode = Episode(
          id=str(uuid4()),
          user_id=user_id,
          session_id=session_id,
          role=role,
          content=content,
          embedding=embedding,
          topics=topics,
          sentiment=sentiment,
          token_count=token_count,
          compression_stage=0,
          created_at=now,
          last_accessed_at=now,
          access_count=0,
        )
        await self.store.store_episode(episode)

        # Update topic transitions
        if prev_topics and topics:
          for from_t in prev_topics:
            for to_t in topics:
              if from_t != to_t:
                await self.store.store_topic_transition(from_t, to_t, user_id=user_id)
        prev_topics = topics

      # Schedule distillation (non-blocking)
      self._schedule_distillation(user_id=user_id)

    except Exception as e:
      log_warning(f"Memory store failed (non-fatal): {e}")

  async def run_distillation(self, *, user_id: Optional[str] = None) -> DistillationResult:
    """Manually trigger distillation.

    Args:
        user_id: Optional user ID filter.

    Returns:
        DistillationResult with processing counts.
    """
    await self._ensure_initialized()
    return await run_distillation(
      self.store,
      model=self.distillation_model,
      embedder=self.embedder,
      user_id=user_id,
      config=self.config,
    )

  async def forget(self, *, user_id: Optional[str] = None, session_id: Optional[str] = None) -> None:
    """Delete stored memories.

    Args:
        user_id: Delete all data for this user.
        session_id: Delete all data for this session.
    """
    await self._ensure_initialized()
    if user_id:
      await self.store.delete_user_data(user_id)
    if session_id:
      await self.store.delete_session_data(session_id)

  async def close(self) -> None:
    """Close the memory store."""
    if self._initialized:
      await self.store.close()
      self._initialized = False

  def _schedule_distillation(self, *, user_id: Optional[str] = None) -> None:
    """Schedule a non-blocking distillation run."""
    try:
      loop = asyncio.get_running_loop()
      loop.create_task(self._safe_distillation(user_id=user_id))
    except RuntimeError:
      pass  # No running loop — skip background distillation

  async def _safe_distillation(self, *, user_id: Optional[str] = None) -> None:
    """Run distillation with error suppression."""
    try:
      await run_distillation(
        self.store,
        model=self.distillation_model,
        embedder=self.embedder,
        user_id=user_id,
        config=self.config,
      )
    except Exception as e:
      log_debug(f"Background distillation failed: {e}", log_level=2)

  async def __aenter__(self):
    await self._ensure_initialized()
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    await self.close()


# --- Sentiment heuristic ---

_POSITIVE_WORDS = {
  "good",
  "great",
  "excellent",
  "amazing",
  "wonderful",
  "fantastic",
  "love",
  "like",
  "enjoy",
  "happy",
  "pleased",
  "glad",
  "thanks",
  "thank",
  "perfect",
  "awesome",
  "nice",
  "brilliant",
  "best",
  "beautiful",
  "helpful",
  "appreciate",
  "excited",
  "impressive",
}

_NEGATIVE_WORDS = {
  "bad",
  "terrible",
  "awful",
  "horrible",
  "hate",
  "dislike",
  "angry",
  "frustrated",
  "annoyed",
  "disappointed",
  "sad",
  "wrong",
  "broken",
  "error",
  "fail",
  "failed",
  "bug",
  "worst",
  "ugly",
  "useless",
  "stupid",
  "problem",
  "issue",
  "crash",
  "slow",
  "confusing",
  "difficult",
  "hard",
}


def _compute_sentiment(text: str) -> float:
  """Compute simple keyword-based sentiment score.

  Returns a value between -1.0 (very negative) and 1.0 (very positive).
  """
  words = set(text.lower().split())
  pos = len(words & _POSITIVE_WORDS)
  neg = len(words & _NEGATIVE_WORDS)
  total = pos + neg
  if total == 0:
    return 0.0
  return (pos - neg) / total
