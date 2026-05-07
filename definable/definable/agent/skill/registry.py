"""SkillRegistry — discover, search, and inject markdown skills.

The registry loads markdown skills from the built-in library and/or
custom directories, and provides three injection modes:

- **On-demand** (default): An XML catalog + ``activate_skill``,
  ``read_skill_file``, and ``run_skill_script`` tools are injected.
  The LLM picks which skill to activate based on the user's query.
- **Eager**: All skill instructions are injected into the system prompt
  (best for small collections, or explicit user choice).
- **Lazy**: A catalog table + ``read_skill`` tool (legacy, kept for
  backward compatibility).

Example::

    from definable.agent.skill import SkillRegistry

    registry = SkillRegistry()           # loads built-in library
    print(registry.list_skills())        # list available skills

    # On-demand (recommended): inject catalog + 3 tools
    agent = Agent(model=model, skill_registry=registry)

    # Eager: inject all skills
    agent = Agent(model=model, skills=registry.as_eager())

    # Legacy lazy: inject catalog + read_skill tool
    agent = Agent(model=model, skills=[registry.as_lazy()])
"""

from pathlib import Path
from typing import List, Optional
from xml.sax.saxutils import escape as xml_escape

from definable.agent.skill.base import Skill
from definable.agent.skill.markdown import MarkdownSkill, MarkdownSkillMeta, SkillLoader


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

  def to_prompt(self) -> str:
    """Generate an XML catalog of all skills for system prompt injection.

    Only name and description are included — this is the compact
    catalog that lets the LLM decide which skill to activate.

    Returns:
      XML string with ``<available_skills>`` root element.
    """
    lines = ["<available_skills>"]
    for meta in self.list_skills():
      lines.append("  <skill>")
      lines.append(f"    <name>{xml_escape(meta.name)}</name>")
      lines.append(f"    <description>{xml_escape(meta.description)}</description>")
      lines.append("  </skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)

  def as_on_demand(self) -> Skill:
    """Return a Skill with XML catalog + 3 on-demand tools.

    This is the recommended mode for ``skill_registry=`` on Agent.
    Only name+description are injected at startup; the model calls
    ``activate_skill`` to load full content on demand.

    Returns:
      A Skill with catalog instructions and activate/read/run tools.
    """
    return _build_on_demand_skill(self)

  def as_eager(self) -> List[Skill]:
    """Return all skills for eager injection.

    Each skill's ``get_instructions()`` will be called by the agent's
    ``_build_skill_instructions()`` during run.

    Returns:
      List of Skill instances (MarkdownSkill is a Skill subclass).
    """
    return list(self._skills.values())

  def __len__(self) -> int:
    return len(self._skills)

  def __contains__(self, name: str) -> bool:
    return name in self._skills

  def __repr__(self) -> str:
    return f"SkillRegistry(skills={len(self._skills)})"


def _build_on_demand_skill(registry: SkillRegistry) -> Skill:
  """Build a wrapper Skill with XML catalog + 3 on-demand tools.

  Tools:
    - ``activate_skill``: Load full SKILL.md content for a skill.
    - ``read_skill_file``: Read a bundled resource file.
    - ``run_skill_script``: Execute a bundled script.

  Args:
    registry: The SkillRegistry to build from.

  Returns:
    A Skill with catalog instructions and 3 tools.
  """
  from definable.tool.decorator import tool

  catalog_xml = registry.to_prompt()
  instructions = (
    "You have access to specialized skills. Review the catalog below and activate "
    "any skill relevant to the user's request using the `activate_skill` tool.\n\n"
    f"{catalog_xml}\n\n"
    "When a skill is activated, follow its instructions carefully. Use "
    "`read_skill_file` to access referenced documents and `run_skill_script` "
    "to execute bundled scripts."
  )

  @tool(description="Activate a skill by name. Returns the skill's full instructions and lists any bundled resources.")
  def activate_skill(skill_name: str) -> str:
    """Activate a skill by name to load its full instructions.

    Args:
      skill_name: The name of the skill to activate (from the catalog).

    Returns:
      The skill's full content and resource listing.
    """
    skill = registry.get_skill(skill_name)
    if skill is None:
      available = ", ".join(s.name for s in registry.list_skills())
      return f"Skill '{skill_name}' not found. Available skills: {available}"

    result = skill.get_instructions()

    # List bundled resources if this is a directory skill
    if skill.is_directory_skill:
      files = skill.list_files()
      if files:
        result += "\n\n<bundled_resources>\n"
        for f in files:
          result += f"  {f}\n"
        result += "</bundled_resources>"

    return result

  @tool(description="Read a bundled resource file from a skill directory (e.g. references, forms, examples).")
  def read_skill_file(skill_name: str, file_path: str) -> str:
    """Read a bundled resource file from an activated skill.

    Args:
      skill_name: The name of the skill.
      file_path: Relative path within the skill directory (e.g. "references/api.md").

    Returns:
      The file content, or an error message.
    """
    skill = registry.get_skill(skill_name)
    if skill is None:
      return f"Skill '{skill_name}' not found."
    try:
      return skill.read_file(file_path)
    except (ValueError, FileNotFoundError) as e:
      return f"Error: {e}"

  @tool(description="Run a bundled script from a skill directory. Returns stdout/stderr output.")
  def run_skill_script(skill_name: str, script_path: str, args: str = "") -> str:
    """Run a bundled script from a skill directory.

    Args:
      skill_name: The name of the skill.
      script_path: Relative path to the script (e.g. "scripts/extract.py").
      args: Space-separated command-line arguments (optional).

    Returns:
      Script output (stdout + stderr).
    """
    import asyncio

    from definable.agent.skill.executor import SkillScriptExecutor

    skill = registry.get_skill(skill_name)
    if skill is None:
      return f"Skill '{skill_name}' not found."

    executor = SkillScriptExecutor()
    arg_list = args.split() if args.strip() else []

    # Run the async executor — handle both running and no-loop contexts
    try:
      loop = asyncio.get_running_loop()
    except RuntimeError:
      loop = None

    if loop and loop.is_running():
      # We're inside an async context — create a task via the loop
      import concurrent.futures

      with concurrent.futures.ThreadPoolExecutor() as pool:
        result = pool.submit(asyncio.run, executor.run(skill, script_path, arg_list)).result()
      return result  # type: ignore[return-value]
    else:
      return asyncio.run(executor.run(skill, script_path, arg_list))

  return Skill(
    name="skill_library",
    instructions=instructions,
    tools=[activate_skill, read_skill_file, run_skill_script],
  )
