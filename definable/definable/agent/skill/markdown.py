"""Markdown-based skills — teach agents methodology via .md files.

A MarkdownSkill is a Skill defined by a markdown file with YAML
frontmatter.  The frontmatter provides searchable metadata (name,
description, tags) and the body provides domain expertise (methodology,
steps, rules) injected into the agent's system prompt.

Supports two formats:

**Flat file** (original)::

    skills/code-review.md
    ---
    name: code-review
    description: Systematic code review
    tags: [code, review]
    ---
    ## Steps ...

**Directory skill** (Agent Skills spec)::

    skills/pdf/SKILL.md
    skills/pdf/scripts/extract.py
    skills/pdf/references/api.md

    ---
    name: pdf
    description: Use when working with PDF files...
    license: Proprietary
    metadata:
      version: 2.0.0
      author: Anthropic
    ---
    # PDF Processing Guide ...

Usage::

    from definable.agent.skill import MarkdownSkill, SkillLoader

    # Parse from string
    skill = SkillLoader.parse(markdown_text)

    # Load a directory skill
    skill = SkillLoader.load_skill_directory(Path("skills/pdf"))

    # Load a mixed directory (flat + directory skills)
    skills = SkillLoader.load_directory(Path("skills/"))
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from definable.agent.skill.base import Skill


# Agent Skills spec name pattern: kebab-case, lowercase, 1-64 chars
_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")


def validate_agent_skills_name(name: str) -> List[str]:
  """Validate a skill name against the Agent Skills spec.

  Names must be kebab-case, lowercase, 1-64 characters. No consecutive
  hyphens, no leading/trailing hyphens.

  Args:
    name: The skill name to validate.

  Returns:
    List of validation error strings. Empty list means valid.
  """
  errors: List[str] = []
  if not name:
    errors.append("Name must not be empty")
    return errors
  if len(name) > 64:
    errors.append(f"Name must be 64 characters or fewer (got {len(name)})")
  if name != name.lower():
    errors.append("Name must be lowercase")
  if "--" in name:
    errors.append("Name must not contain consecutive hyphens")
  if name.startswith("-"):
    errors.append("Name must not start with a hyphen")
  if name.endswith("-"):
    errors.append("Name must not end with a hyphen")
  if not _NAME_PATTERN.match(name) and not errors:
    errors.append("Name must be kebab-case (lowercase letters, digits, single hyphens)")
  return errors


@dataclass
class MarkdownSkillMeta:
  """Metadata parsed from a markdown skill's YAML frontmatter.

  Supports both the original flat format and the Agent Skills spec
  format with nested ``metadata:`` blocks.

  Attributes:
    name: Unique skill identifier (required).
    description: Short human-readable description.
    version: Semver string (e.g. "1.0.0").
    requires_tools: Tool names this skill expects the agent to have.
    tags: Searchable tags for discovery.
    author: Skill author.
    license: License string (Agent Skills spec).
    compatibility: Compatibility info (Agent Skills spec).
    metadata: Additional key-value metadata (Agent Skills spec).
    allowed_tools: Tools this skill is allowed to invoke (Agent Skills spec).
  """

  name: str
  description: str = ""
  version: str = "1.0.0"
  requires_tools: List[str] = field(default_factory=list)
  tags: List[str] = field(default_factory=list)
  author: str = ""
  license: str = ""
  compatibility: str = ""
  metadata: Dict[str, str] = field(default_factory=dict)
  allowed_tools: List[str] = field(default_factory=list)


class MarkdownSkill(Skill):
  """A skill defined by markdown content with YAML frontmatter.

  MarkdownSkill extends :class:`Skill` to carry structured metadata
  (``meta``) and wrap its instructions in ``<skill>`` tags for clear
  prompt delineation.

  Supports both flat file skills and directory-based Agent Skills
  with bundled resources (scripts, references, assets).

  Args:
    meta: Parsed frontmatter metadata.
    content: Markdown body (after frontmatter).
    source_path: Optional path the skill was loaded from.
    skill_dir: Optional directory path for directory-based skills.
  """

  def __init__(
    self,
    *,
    meta: MarkdownSkillMeta,
    content: str,
    source_path: Optional[Path] = None,
    skill_dir: Optional[Path] = None,
  ):
    super().__init__(
      name=meta.name,
      instructions=content,
    )
    self.meta = meta
    self._raw_content = content
    self._source_path = source_path
    self._skill_dir = skill_dir

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

  # ── Directory skill resource access ─────────────────────────

  @property
  def skill_directory(self) -> Optional[Path]:
    """The directory path for directory-based skills, or None."""
    return self._skill_dir

  @property
  def is_directory_skill(self) -> bool:
    """Whether this skill was loaded from a directory (vs flat file)."""
    return self._skill_dir is not None

  def list_files(self) -> List[str]:
    """List all bundled files in the skill directory (excluding SKILL.md).

    Returns:
      Sorted list of relative file paths. Empty if not a directory skill.
    """
    if self._skill_dir is None or not self._skill_dir.is_dir():
      return []
    files: List[str] = []
    for path in sorted(self._skill_dir.rglob("*")):
      if path.is_file() and path.name.upper() != "SKILL.MD":
        files.append(str(path.relative_to(self._skill_dir)))
    return files

  def read_file(self, relative_path: str) -> str:
    """Read a bundled resource file from the skill directory.

    Args:
      relative_path: Path relative to skill directory (e.g. "scripts/extract.py").

    Returns:
      File content as string.

    Raises:
      ValueError: If not a directory skill or path escapes the skill directory.
      FileNotFoundError: If the file does not exist.
    """
    if self._skill_dir is None:
      raise ValueError(f"Skill '{self.meta.name}' is not a directory skill")
    target = (self._skill_dir / relative_path).resolve()
    skill_root = self._skill_dir.resolve()
    if not str(target).startswith(str(skill_root)):
      raise ValueError(f"Path '{relative_path}' escapes skill directory")
    if not target.is_file():
      raise FileNotFoundError(f"File not found: {relative_path}")
    return target.read_text(encoding="utf-8")

  def has_scripts(self) -> bool:
    """Whether the skill has a scripts/ subdirectory with files."""
    if self._skill_dir is None:
      return False
    scripts_dir = self._skill_dir / "scripts"
    return scripts_dir.is_dir() and any(scripts_dir.iterdir())

  def has_references(self) -> bool:
    """Whether the skill has a references/ subdirectory with files."""
    if self._skill_dir is None:
      return False
    refs_dir = self._skill_dir / "references"
    return refs_dir.is_dir() and any(refs_dir.iterdir())

  def has_assets(self) -> bool:
    """Whether the skill has an assets/ subdirectory with files."""
    if self._skill_dir is None:
      return False
    assets_dir = self._skill_dir / "assets"
    return assets_dir.is_dir() and any(assets_dir.iterdir())

  def __repr__(self) -> str:
    tags = ", ".join(self.meta.tags) if self.meta.tags else "none"
    kind = "dir" if self.is_directory_skill else "file"
    return f"MarkdownSkill({self.meta.name!r}, tags=[{tags}], {kind})"


class SkillLoader:
  """Static methods for parsing and loading markdown skills."""

  @staticmethod
  def parse_frontmatter(text: str) -> Tuple[Dict[str, object], str]:
    """Parse YAML frontmatter from markdown text.

    Handles simple ``key: value`` pairs, ``key: "quoted"`` values,
    ``key: [list, items]`` syntax, and one level of nested blocks
    (e.g. ``metadata:`` with indented children).

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

    # Parse YAML-like key: value lines with one level of nesting
    data: Dict[str, object] = {}
    current_nested_key: Optional[str] = None
    current_nested_dict: Dict[str, str] = {}

    for line in fm_block.splitlines():
      # Blank or comment lines
      stripped_line = line.strip()
      if not stripped_line or stripped_line.startswith("#"):
        continue

      # Check if this is an indented line (part of a nested block)
      if line[0] in (" ", "\t") and current_nested_key is not None:
        colon_idx = stripped_line.find(":")
        if colon_idx != -1:
          nested_key = stripped_line[:colon_idx].strip()
          nested_val = stripped_line[colon_idx + 1 :].strip()
          current_nested_dict[nested_key] = _strip_quotes(nested_val) if nested_val else ""
        continue

      # Save any accumulated nested dict
      if current_nested_key is not None:
        data[current_nested_key] = dict(current_nested_dict)
        current_nested_key = None
        current_nested_dict = {}

      # Parse top-level key: value
      colon_idx = stripped_line.find(":")
      if colon_idx == -1:
        continue

      key = stripped_line[:colon_idx].strip()
      raw_val = stripped_line[colon_idx + 1 :].strip()

      # Check if this starts a nested block (value is empty → dict follows)
      if not raw_val:
        current_nested_key = key
        current_nested_dict = {}
        continue

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

    # Flush any remaining nested dict
    if current_nested_key is not None:
      data[current_nested_key] = dict(current_nested_dict)

    return data, body

  @staticmethod
  def parse(text: str, source_path: Optional[Path] = None) -> "MarkdownSkill":
    """Parse markdown text into a MarkdownSkill.

    Handles both the original flat frontmatter and the Agent Skills
    spec format with nested ``metadata:`` blocks and hyphenated keys.

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

    # Build requires_tools list (support hyphenated key too)
    requires_tools_raw = fm.get("requires_tools") or fm.get("requires-tools", [])
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

    # Build allowed_tools list (hyphenated key)
    allowed_tools_raw = fm.get("allowed_tools") or fm.get("allowed-tools", [])
    if isinstance(allowed_tools_raw, str):
      allowed_tools = [allowed_tools_raw]
    elif isinstance(allowed_tools_raw, list):
      allowed_tools = [str(t) for t in allowed_tools_raw]
    else:
      allowed_tools = []

    # Parse metadata dict (Agent Skills spec nested block)
    metadata_raw = fm.get("metadata", {})
    metadata: Dict[str, str] = {}
    if isinstance(metadata_raw, dict):
      metadata = {str(k): str(v) for k, v in metadata_raw.items()}

    # Bidirectional mapping: explicit top-level takes precedence over metadata
    version = str(fm.get("version", metadata.get("version", "1.0.0")))
    author = str(fm.get("author", metadata.get("author", "")))
    description = str(fm.get("description", ""))
    license_str = str(fm.get("license", metadata.get("license", "")))
    compatibility = str(fm.get("compatibility", metadata.get("compatibility", "")))

    # Tags from top-level or metadata
    if not tags and "tags" in metadata:
      tag_str = metadata["tags"]
      tags = [t.strip() for t in tag_str.split(",") if t.strip()]

    meta = MarkdownSkillMeta(
      name=str(name),
      description=description,
      version=version,
      requires_tools=requires_tools,
      tags=tags,
      author=author,
      license=license_str,
      compatibility=compatibility,
      metadata=metadata,
      allowed_tools=allowed_tools,
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
  def load_skill_directory(directory: Path) -> "MarkdownSkill":
    """Load a directory-based skill (Agent Skills spec format).

    Discovers ``SKILL.md`` (case-insensitive) in the directory, parses it,
    and sets the skill_dir for resource access.

    Args:
      directory: Directory containing a ``SKILL.md`` file.

    Returns:
      A MarkdownSkill with ``skill_dir`` set.

    Raises:
      FileNotFoundError: If no ``SKILL.md`` found in the directory.
      ValueError: If parsing fails.
    """
    from definable.utils.log import log_warning

    # Find SKILL.md case-insensitively
    skill_md: Optional[Path] = None
    for child in directory.iterdir():
      if child.is_file() and child.name.upper() == "SKILL.MD":
        skill_md = child
        break

    if skill_md is None:
      raise FileNotFoundError(f"No SKILL.md found in {directory}")

    text = skill_md.read_text(encoding="utf-8")
    skill = SkillLoader.parse(text, source_path=skill_md)
    skill._skill_dir = directory

    # Validate name matches directory name (warning, not error)
    dir_name = directory.name
    if skill.meta.name != dir_name:
      name_errors = validate_agent_skills_name(skill.meta.name)
      if name_errors:
        log_warning(f"Skill '{skill.meta.name}' name validation: {', '.join(name_errors)}")

    return skill

  @staticmethod
  def load_directory(directory: Path) -> List["MarkdownSkill"]:
    """Load skills from a directory, supporting both flat and directory formats.

    First pass: scan subdirectories containing ``SKILL.md`` → directory skills.
    Second pass: scan loose ``*.md`` files (excluding SKILL.md) → flat skills.
    Directory skills win on name collision (dedup).

    Args:
      directory: Directory to scan.

    Returns:
      List of successfully parsed MarkdownSkill instances.
    """
    from definable.utils.log import log_warning

    skills_by_name: Dict[str, MarkdownSkill] = {}

    if not directory.is_dir():
      log_warning(f"Skill directory does not exist: {directory}")
      return []

    # Pass 1: directory skills (subdirectories with SKILL.md)
    for child in sorted(directory.iterdir()):
      if not child.is_dir():
        continue
      # Check for SKILL.md case-insensitively
      has_skill_md = any(f.name.upper() == "SKILL.MD" for f in child.iterdir() if f.is_file())
      if has_skill_md:
        try:
          skill = SkillLoader.load_skill_directory(child)
          skills_by_name[skill.meta.name] = skill
        except Exception as e:
          log_warning(f"Failed to load skill directory {child}: {e}")

    # Pass 2: flat .md files (excluding SKILL.md itself)
    for md_path in sorted(directory.glob("*.md")):
      if md_path.name.upper() == "SKILL.MD":
        continue
      try:
        skill = SkillLoader.load_file(md_path)
        # Directory skills win on collision
        if skill.meta.name not in skills_by_name:
          skills_by_name[skill.meta.name] = skill
      except Exception as e:
        log_warning(f"Failed to load skill from {md_path}: {e}")

    return list(skills_by_name.values())


def _strip_quotes(s: str) -> str:
  """Strip surrounding quotes from a string."""
  if len(s) >= 2:
    if (s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'"):
      return s[1:-1]
  return s
