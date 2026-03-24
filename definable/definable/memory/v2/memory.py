"""Memory — tool-based memory orchestrator.

The LLM is the memory manager. This class wires the store, tools,
skill, and prompt injection together. The developer just passes it to Agent.
"""

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from definable.memory.v2.prompt import build_working_memory_block
from definable.memory.v2.stores.base import MemoryStore

if TYPE_CHECKING:
  from definable.skill.base import Skill
  from definable.tool.function import Function

_SKILL_DIR = Path(__file__).parent / "skill"


class Memory:
  """Tool-based memory system.

  The agent gets memory tools auto-injected, a memory-manager skill
  loaded into the system prompt, and working memory injected before
  every LLM call.

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
    index_max_entries: int = 200,
    categories: Optional[List[str]] = None,
    auto_inject: bool = True,
  ) -> None:
    self.store = store
    self.working_memory_max_chars = working_memory_max_chars
    self.index_max_entries = index_max_entries
    self.categories = categories or ["user", "feedback", "project", "reference", "conversation"]
    self.auto_inject = auto_inject
    self._skill: Optional["Skill"] = None

    # Compat flags for agent integration (v1 Memory had these)
    self.enabled = True
    self.description: Optional[str] = None
    self.model: Optional[object] = None
    self.has_semantic_search: bool = False

  def get_skill(self) -> "Skill":
    """Load and return the memory-manager MarkdownSkill.

    The skill contains all memory management instructions and is
    automatically attached to the agent alongside memory tools.
    """
    if self._skill is None:
      from definable.skill.markdown import SkillLoader

      self._skill = SkillLoader.load_skill_directory(_SKILL_DIR)
    return self._skill

  def get_tools(self, user_id: str, session_id: str) -> "List[Function]":
    """Build memory tools bound to the given user/session context."""
    from definable.memory.v2.tools import build_memory_tools

    return build_memory_tools(self, user_id, session_id)

  async def get_prompt_injection(self, user_id: str) -> str:
    """Build the working memory block for system prompt injection.

    The memory management instructions come from the skill (SKILL.md),
    which is injected separately via the agent's skill system.
    This method only provides the <working_memory> block.
    """
    wm = await self.store.get_working_memory(user_id)
    content = wm.content if wm else ""
    updated_at = wm.updated_at.isoformat() if wm else ""
    return build_working_memory_block(user_id, content, updated_at)

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
