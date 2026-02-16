"""SkillRegistry — discover, search, and inject markdown skills.

The registry loads markdown skills from the built-in library and/or
custom directories, and provides two injection modes:

- **Eager**: All skill instructions are injected into the system prompt
  (best for small collections).
- **Lazy**: A catalog table + ``read_skill`` tool are injected, letting
  the LLM load skills on demand (best for large collections).

Example::

    from definable.skills import SkillRegistry

    registry = SkillRegistry()           # loads built-in library
    print(registry.list_skills())        # list available skills

    # Eager: inject all skills
    agent = Agent(model=model, skills=registry.as_eager())

    # Lazy: inject catalog + read_skill tool
    agent = Agent(model=model, skills=[registry.as_lazy()])
"""

from pathlib import Path
from typing import List, Optional

from definable.skills.base import Skill
from definable.skills.markdown import MarkdownSkill, MarkdownSkillMeta, SkillLoader


# Built-in library directory
_LIBRARY_DIR = Path(__file__).parent / "library"


class SkillRegistry:
  """Registry for discovering and managing markdown skills.

  Args:
    skills: Explicit list of MarkdownSkill instances to register.
    directories: Additional directories to scan for ``.md`` skill files.
    include_library: Whether to load the built-in skill library (default True).
  """

  def __init__(
    self,
    *,
    skills: Optional[List[MarkdownSkill]] = None,
    directories: Optional[List[Path]] = None,
    include_library: bool = True,
  ):
    self._skills: dict[str, MarkdownSkill] = {}

    # 1. Load built-in library
    if include_library and _LIBRARY_DIR.is_dir():
      for skill in SkillLoader.load_directory(_LIBRARY_DIR):
        self._skills[skill.meta.name] = skill

    # 2. Load from custom directories
    if directories:
      for directory in directories:
        for skill in SkillLoader.load_directory(directory):
          self._skills[skill.meta.name] = skill

    # 3. Add explicit skills (last wins for dedup)
    if skills:
      for skill in skills:
        self._skills[skill.meta.name] = skill

  def list_skills(self) -> List[MarkdownSkillMeta]:
    """Return metadata for all registered skills.

    Returns:
      List of MarkdownSkillMeta, sorted by name.
    """
    return [s.meta for s in sorted(self._skills.values(), key=lambda s: s.meta.name)]

  def get_skill(self, name: str) -> Optional[MarkdownSkill]:
    """Look up a skill by name.

    Args:
      name: Skill name (case-sensitive).

    Returns:
      The MarkdownSkill, or None if not found.
    """
    return self._skills.get(name)

  def search_skills(self, query: str) -> List[MarkdownSkill]:
    """Search skills by keyword/tag matching.

    Scores each skill based on matches in name, description, and tags.
    Tag matches receive a bonus.  Results are returned in descending
    score order.

    Args:
      query: Search query string.

    Returns:
      List of matching skills, best matches first.
    """
    query_lower = query.lower()
    query_terms = query_lower.split()
    scored: List[tuple[float, MarkdownSkill]] = []

    for skill in self._skills.values():
      score = 0.0
      name_lower = skill.meta.name.lower()
      desc_lower = skill.meta.description.lower()
      tags_lower = [t.lower() for t in skill.meta.tags]

      for term in query_terms:
        # Name match (highest signal)
        if term in name_lower:
          score += 3.0
        # Description match
        if term in desc_lower:
          score += 1.0
        # Tag match (bonus)
        for tag in tags_lower:
          if term in tag:
            score += 2.0

      if score > 0:
        scored.append((score, skill))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [skill for _, skill in scored]

  def as_eager(self) -> List[Skill]:
    """Return all skills for eager injection.

    Each skill's ``get_instructions()`` will be called by the agent's
    ``_build_skill_instructions()`` during run.

    Returns:
      List of Skill instances (MarkdownSkill is a Skill subclass).
    """
    return list(self._skills.values())

  def as_lazy(self) -> Skill:
    """Return a single wrapper skill with a catalog and ``read_skill`` tool.

    The wrapper skill provides:
    - Instructions: a markdown table listing all available skills.
    - Tools: a ``read_skill`` tool that loads a skill's full content.

    Returns:
      A Skill instance with catalog instructions and read_skill tool.
    """
    return _build_lazy_skill(self)

  def __len__(self) -> int:
    return len(self._skills)

  def __contains__(self, name: str) -> bool:
    return name in self._skills

  def __repr__(self) -> str:
    return f"SkillRegistry(skills={len(self._skills)})"


def _build_lazy_skill(registry: SkillRegistry) -> Skill:
  """Build a wrapper Skill with catalog table + read_skill tool.

  Args:
    registry: The SkillRegistry to build the lazy skill from.

  Returns:
    A Skill with catalog instructions and a read_skill tool.
  """
  from definable.tools.decorator import tool

  # Build catalog table
  lines = [
    "# Available Skills",
    "",
    "You have access to a library of skills. Use the `read_skill` tool to load a skill's full methodology before tackling a task.",
    "",
    "| Name | Description | Tags |",
    "|------|-------------|------|",
  ]
  for meta in registry.list_skills():
    tags = ", ".join(meta.tags) if meta.tags else ""
    lines.append(f"| {meta.name} | {meta.description} | {tags} |")

  catalog = "\n".join(lines)

  # Create read_skill tool via closure
  @tool(description="Load a skill's full methodology by name. Returns the skill content or an error if not found.")
  def read_skill(skill_name: str) -> str:
    """Load a skill's full methodology by name.

    Args:
      skill_name: The name of the skill to load (from the catalog table).

    Returns:
      The skill's full content, or an error message if not found.
    """
    skill = registry.get_skill(skill_name)
    if skill is None:
      available = ", ".join(s.name for s in registry.list_skills())
      return f"Skill '{skill_name}' not found. Available skills: {available}"
    return skill.get_instructions()

  return Skill(
    name="skill_library",
    instructions=catalog,
    tools=[read_skill],
  )
