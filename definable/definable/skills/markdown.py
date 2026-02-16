"""Markdown-based skills — teach agents methodology via .md files.

A MarkdownSkill is a Skill defined by a markdown file with YAML
frontmatter.  The frontmatter provides searchable metadata (name,
description, tags) and the body provides domain expertise (methodology,
steps, rules) injected into the agent's system prompt.

Example markdown skill file::

    ---
    name: code-review
    description: Systematic code review with severity-ranked findings
    tags: [code, review, quality]
    ---

    ## When to Use
    Use this skill when reviewing code changes...

    ## Steps
    1. Read the diff carefully
    2. ...

Usage::

    from definable.skills import MarkdownSkill, SkillLoader

    # Parse from string
    skill = SkillLoader.parse(markdown_text)

    # Load from file
    skill = SkillLoader.load_file(Path("skills/code-review.md"))

    # Load a directory of skills
    skills = SkillLoader.load_directory(Path("skills/"))
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from definable.skills.base import Skill


@dataclass
class MarkdownSkillMeta:
  """Metadata parsed from a markdown skill's YAML frontmatter.

  Attributes:
    name: Unique skill identifier (required).
    description: Short human-readable description.
    version: Semver string (e.g. "1.0.0").
    requires_tools: Tool names this skill expects the agent to have.
    tags: Searchable tags for discovery.
    author: Skill author.
  """

  name: str
  description: str = ""
  version: str = "1.0.0"
  requires_tools: List[str] = field(default_factory=list)
  tags: List[str] = field(default_factory=list)
  author: str = ""


class MarkdownSkill(Skill):
  """A skill defined by markdown content with YAML frontmatter.

  MarkdownSkill extends :class:`Skill` to carry structured metadata
  (``meta``) and wrap its instructions in ``<skill>`` tags for clear
  prompt delineation.

  Args:
    meta: Parsed frontmatter metadata.
    content: Markdown body (after frontmatter).
    source_path: Optional path the skill was loaded from.
  """

  def __init__(
    self,
    *,
    meta: MarkdownSkillMeta,
    content: str,
    source_path: Optional[Path] = None,
  ):
    super().__init__(
      name=meta.name,
      instructions=content,
    )
    self.meta = meta
    self._raw_content = content
    self._source_path = source_path

  def get_instructions(self) -> str:
    """Wrap content in ``<skill>`` tags for clear prompt delineation."""
    text = self._raw_content.strip()
    if not text:
      return ""
    return f'<skill name="{self.meta.name}">\n{text}\n</skill>'

  @property
  def tools(self) -> list:
    """Markdown skills provide no tools."""
    return []

  def __repr__(self) -> str:
    tags = ", ".join(self.meta.tags) if self.meta.tags else "none"
    return f"MarkdownSkill({self.meta.name!r}, tags=[{tags}])"


class SkillLoader:
  """Static methods for parsing and loading markdown skills."""

  @staticmethod
  def parse_frontmatter(text: str) -> Tuple[Dict[str, object], str]:
    """Parse YAML frontmatter from markdown text.

    Handles simple ``key: value`` pairs, ``key: "quoted"`` values,
    and ``key: [list, items]`` syntax.  No pyyaml dependency.

    Args:
      text: Full markdown text with optional ``---`` delimited frontmatter.

    Returns:
      Tuple of (frontmatter dict, body text).

    Raises:
      ValueError: If frontmatter delimiters are malformed.
    """
    stripped = text.lstrip("\n")
    if not stripped.startswith("---"):
      return {}, text

    # Find the closing ---
    first_newline = stripped.index("\n")
    rest = stripped[first_newline + 1 :]
    end_idx = rest.find("\n---")
    if end_idx == -1:
      raise ValueError("Frontmatter opening '---' found but no closing '---'")

    fm_block = rest[:end_idx]
    body = rest[end_idx + 4 :].lstrip("\n")  # skip \n---

    # Parse YAML-like key: value lines
    data: Dict[str, object] = {}
    for line in fm_block.splitlines():
      line = line.strip()
      if not line or line.startswith("#"):
        continue

      colon_idx = line.find(":")
      if colon_idx == -1:
        continue

      key = line[:colon_idx].strip()
      raw_val = line[colon_idx + 1 :].strip()

      # Parse value
      if raw_val.startswith("[") and raw_val.endswith("]"):
        # List: [item1, item2, item3]
        inner = raw_val[1:-1].strip()
        if not inner:
          data[key] = []
        else:
          items = [_strip_quotes(item.strip()) for item in inner.split(",")]
          data[key] = [item for item in items if item]
      elif raw_val.startswith('"') and raw_val.endswith('"'):
        data[key] = raw_val[1:-1]
      elif raw_val.startswith("'") and raw_val.endswith("'"):
        data[key] = raw_val[1:-1]
      else:
        data[key] = raw_val

    return data, body

  @staticmethod
  def parse(text: str, source_path: Optional[Path] = None) -> "MarkdownSkill":
    """Parse markdown text into a MarkdownSkill.

    Args:
      text: Full markdown content with YAML frontmatter.
      source_path: Optional file path for debugging.

    Returns:
      A MarkdownSkill instance.

    Raises:
      ValueError: If required ``name`` field is missing from frontmatter.
    """
    fm, body = SkillLoader.parse_frontmatter(text)

    name = fm.get("name")
    if not name or not isinstance(name, str):
      src = f" in {source_path}" if source_path else ""
      raise ValueError(f"Markdown skill frontmatter must include 'name'{src}")

    # Build requires_tools list
    requires_tools_raw = fm.get("requires_tools", [])
    if isinstance(requires_tools_raw, str):
      requires_tools = [requires_tools_raw]
    elif isinstance(requires_tools_raw, list):
      requires_tools = [str(t) for t in requires_tools_raw]
    else:
      requires_tools = []

    # Build tags list
    tags_raw = fm.get("tags", [])
    if isinstance(tags_raw, str):
      tags = [tags_raw]
    elif isinstance(tags_raw, list):
      tags = [str(t) for t in tags_raw]
    else:
      tags = []

    meta = MarkdownSkillMeta(
      name=str(name),
      description=str(fm.get("description", "")),
      version=str(fm.get("version", "1.0.0")),
      requires_tools=requires_tools,
      tags=tags,
      author=str(fm.get("author", "")),
    )

    return MarkdownSkill(
      meta=meta,
      content=body,
      source_path=source_path,
    )

  @staticmethod
  def load_file(path: Path) -> "MarkdownSkill":
    """Load a markdown skill from a file.

    Args:
      path: Path to a ``.md`` file with YAML frontmatter.

    Returns:
      A MarkdownSkill instance.

    Raises:
      FileNotFoundError: If the file does not exist.
      ValueError: If parsing fails.
    """
    text = path.read_text(encoding="utf-8")
    return SkillLoader.parse(text, source_path=path)

  @staticmethod
  def load_directory(directory: Path) -> List["MarkdownSkill"]:
    """Recursively load all ``.md`` skills from a directory.

    Files that fail to parse are logged as warnings and skipped.

    Args:
      directory: Directory to scan for ``.md`` files.

    Returns:
      List of successfully parsed MarkdownSkill instances.
    """
    from definable.utils.log import log_warning

    skills: List[MarkdownSkill] = []
    if not directory.is_dir():
      log_warning(f"Skill directory does not exist: {directory}")
      return skills

    for md_path in sorted(directory.rglob("*.md")):
      try:
        skill = SkillLoader.load_file(md_path)
        skills.append(skill)
      except Exception as e:
        log_warning(f"Failed to load skill from {md_path}: {e}")

    return skills


def _strip_quotes(s: str) -> str:
  """Strip surrounding quotes from a string."""
  if len(s) >= 2:
    if (s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'"):
      return s[1:-1]
  return s
