"""Unit tests for Agent Skills spec integration.

Tests cover: name validation, expanded meta, nested metadata parsing,
directory skill loading, resource access, mixed format loading, XML
prompt generation, on-demand mode, code executor, backward compat,
and full library discovery.
"""

import asyncio
import textwrap
from pathlib import Path

import pytest

from definable.skill.markdown import (
  MarkdownSkill,
  MarkdownSkillMeta,
  SkillLoader,
  validate_agent_skills_name,
)
from definable.skill.registry import SkillRegistry


# ── Helpers ──────────────────────────────────────────────────────


def _make_skill(
  name: str,
  description: str = "",
  tags: list | None = None,
  content: str = "body",
  skill_dir: Path | None = None,
) -> MarkdownSkill:
  """Build a MarkdownSkill without file I/O."""
  return MarkdownSkill(
    meta=MarkdownSkillMeta(name=name, description=description, tags=tags or []),
    content=content,
    skill_dir=skill_dir,
  )


def _make_skill_dir(tmp_path: Path, name: str, content: str, extra_files: dict[str, str] | None = None) -> Path:
  """Create a directory skill on disk with SKILL.md and optional extra files."""
  skill_dir = tmp_path / name
  skill_dir.mkdir()
  (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
  if extra_files:
    for rel_path, file_content in extra_files.items():
      full = skill_dir / rel_path
      full.parent.mkdir(parents=True, exist_ok=True)
      full.write_text(file_content, encoding="utf-8")
  return skill_dir


# ── Name Validation ──────────────────────────────────────────────


@pytest.mark.unit
class TestNameValidation:
  """Tests for validate_agent_skills_name()."""

  def test_valid_simple_name(self):
    assert validate_agent_skills_name("pdf") == []

  def test_valid_kebab_case(self):
    assert validate_agent_skills_name("code-review") == []

  def test_valid_with_digits(self):
    assert validate_agent_skills_name("gpt-4o-mini") == []

  def test_empty_name(self):
    errors = validate_agent_skills_name("")
    assert len(errors) == 1
    assert "empty" in errors[0].lower()

  def test_too_long(self):
    errors = validate_agent_skills_name("a" * 65)
    assert any("64" in e for e in errors)

  def test_uppercase_rejected(self):
    errors = validate_agent_skills_name("Code-Review")
    assert any("lowercase" in e.lower() for e in errors)

  def test_consecutive_hyphens(self):
    errors = validate_agent_skills_name("code--review")
    assert any("consecutive" in e.lower() for e in errors)

  def test_leading_hyphen(self):
    errors = validate_agent_skills_name("-code")
    assert any("start" in e.lower() for e in errors)

  def test_trailing_hyphen(self):
    errors = validate_agent_skills_name("code-")
    assert any("end" in e.lower() for e in errors)

  def test_single_char(self):
    assert validate_agent_skills_name("a") == []

  def test_max_length(self):
    assert validate_agent_skills_name("a" * 64) == []


# ── Meta Expansion ───────────────────────────────────────────────


@pytest.mark.unit
class TestMetaExpansion:
  """Tests for expanded MarkdownSkillMeta fields."""

  def test_default_values(self):
    meta = MarkdownSkillMeta(name="test")
    assert meta.license == ""
    assert meta.compatibility == ""
    assert meta.metadata == {}
    assert meta.allowed_tools == []

  def test_all_new_fields(self):
    meta = MarkdownSkillMeta(
      name="test",
      license="MIT",
      compatibility="claude-code>=1.0",
      metadata={"custom": "value"},
      allowed_tools=["bash", "read"],
    )
    assert meta.license == "MIT"
    assert meta.compatibility == "claude-code>=1.0"
    assert meta.metadata["custom"] == "value"
    assert meta.allowed_tools == ["bash", "read"]

  def test_backward_compat_fields_unchanged(self):
    meta = MarkdownSkillMeta(name="t", version="2.0.0", tags=["a"], author="me")
    assert meta.version == "2.0.0"
    assert meta.tags == ["a"]
    assert meta.author == "me"


# ── Nested Metadata Parsing ─────────────────────────────────────


@pytest.mark.unit
class TestNestedMetadataParsing:
  """Tests for frontmatter parser with nested metadata blocks."""

  def test_nested_metadata_block(self):
    text = textwrap.dedent("""\
      ---
      name: my-skill
      description: A test skill
      metadata:
        version: 2.0.0
        author: Someone
      ---
      Body content.
    """)
    fm, body = SkillLoader.parse_frontmatter(text)
    assert fm["name"] == "my-skill"
    assert isinstance(fm["metadata"], dict)
    assert fm["metadata"]["version"] == "2.0.0"
    assert fm["metadata"]["author"] == "Someone"
    assert "Body content." in body

  def test_hyphenated_keys(self):
    text = textwrap.dedent("""\
      ---
      name: my-skill
      allowed-tools: [bash, read]
      ---
      Body.
    """)
    skill = SkillLoader.parse(text)
    assert skill.meta.allowed_tools == ["bash", "read"]

  def test_top_level_version_overrides_metadata(self):
    text = textwrap.dedent("""\
      ---
      name: my-skill
      version: 3.0.0
      metadata:
        version: 1.0.0
      ---
      Body.
    """)
    skill = SkillLoader.parse(text)
    assert skill.meta.version == "3.0.0"

  def test_metadata_version_used_when_no_top_level(self):
    text = textwrap.dedent("""\
      ---
      name: my-skill
      metadata:
        version: 2.5.0
      ---
      Body.
    """)
    skill = SkillLoader.parse(text)
    assert skill.meta.version == "2.5.0"

  def test_license_parsed(self):
    text = textwrap.dedent("""\
      ---
      name: my-skill
      license: MIT
      ---
      Body.
    """)
    skill = SkillLoader.parse(text)
    assert skill.meta.license == "MIT"

  def test_old_format_still_works(self):
    text = textwrap.dedent("""\
      ---
      name: code-review
      description: Review code
      version: 1.0.0
      tags: [code, review]
      ---
      Use this skill when...
    """)
    skill = SkillLoader.parse(text)
    assert skill.meta.name == "code-review"
    assert skill.meta.version == "1.0.0"
    assert skill.meta.tags == ["code", "review"]

  def test_empty_metadata_block(self):
    text = textwrap.dedent("""\
      ---
      name: my-skill
      metadata:
      ---
      Body.
    """)
    fm, _ = SkillLoader.parse_frontmatter(text)
    # Empty nested block → empty dict
    assert fm["metadata"] == {}


# ── Directory Skill Loading ──────────────────────────────────────


@pytest.mark.unit
class TestDirectorySkillLoading:
  """Tests for load_skill_directory()."""

  def test_load_basic_directory(self, tmp_path):
    content = textwrap.dedent("""\
      ---
      name: my-skill
      description: Test skill
      ---
      Instructions here.
    """)
    skill_dir = _make_skill_dir(tmp_path, "my-skill", content)
    skill = SkillLoader.load_skill_directory(skill_dir)
    assert skill.meta.name == "my-skill"
    assert skill.is_directory_skill is True
    assert skill.skill_directory == skill_dir

  def test_case_insensitive_skill_md(self, tmp_path):
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    (skill_dir / "skill.md").write_text("---\nname: test-skill\n---\nBody.", encoding="utf-8")
    skill = SkillLoader.load_skill_directory(skill_dir)
    assert skill.meta.name == "test-skill"

  def test_missing_skill_md_raises(self, tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No SKILL.md"):
      SkillLoader.load_skill_directory(empty_dir)

  def test_directory_with_resources(self, tmp_path):
    content = "---\nname: pdf\ndescription: PDF tools\n---\nPDF guide."
    extras = {
      "scripts/extract.py": "print('hello')",
      "references/api.md": "# API ref",
      "assets/logo.png": "fake-image-data",
    }
    skill_dir = _make_skill_dir(tmp_path, "pdf", content, extras)
    skill = SkillLoader.load_skill_directory(skill_dir)
    assert skill.has_scripts()
    assert skill.has_references()
    assert skill.has_assets()


# ── Resource Access ──────────────────────────────────────────────


@pytest.mark.unit
class TestResourceAccess:
  """Tests for MarkdownSkill resource methods."""

  def test_list_files(self, tmp_path):
    content = "---\nname: s\n---\nBody."
    extras = {"scripts/a.py": "code", "refs/b.md": "text"}
    skill_dir = _make_skill_dir(tmp_path, "s", content, extras)
    skill = SkillLoader.load_skill_directory(skill_dir)
    files = skill.list_files()
    assert "scripts/a.py" in files
    assert "refs/b.md" in files
    # SKILL.md should be excluded
    assert not any("SKILL" in f.upper() for f in files)

  def test_read_file(self, tmp_path):
    content = "---\nname: s\n---\nBody."
    extras = {"scripts/hello.py": "print('world')"}
    skill_dir = _make_skill_dir(tmp_path, "s", content, extras)
    skill = SkillLoader.load_skill_directory(skill_dir)
    result = skill.read_file("scripts/hello.py")
    assert "print('world')" in result

  def test_path_escape_rejected(self, tmp_path):
    content = "---\nname: s\n---\nBody."
    skill_dir = _make_skill_dir(tmp_path, "s", content)
    skill = SkillLoader.load_skill_directory(skill_dir)
    with pytest.raises(ValueError, match="escapes"):
      skill.read_file("../../etc/passwd")

  def test_read_file_not_found(self, tmp_path):
    content = "---\nname: s\n---\nBody."
    skill_dir = _make_skill_dir(tmp_path, "s", content)
    skill = SkillLoader.load_skill_directory(skill_dir)
    with pytest.raises(FileNotFoundError):
      skill.read_file("nonexistent.py")

  def test_read_file_on_flat_skill_raises(self):
    skill = _make_skill("flat")
    with pytest.raises(ValueError, match="not a directory"):
      skill.read_file("anything.py")

  def test_list_files_on_flat_skill(self):
    skill = _make_skill("flat")
    assert skill.list_files() == []

  def test_has_scripts_false_for_flat(self):
    skill = _make_skill("flat")
    assert skill.has_scripts() is False

  def test_has_references_false_for_flat(self):
    skill = _make_skill("flat")
    assert skill.has_references() is False

  def test_has_assets_false_for_flat(self):
    skill = _make_skill("flat")
    assert skill.has_assets() is False


# ── Mixed Format Loading ─────────────────────────────────────────


@pytest.mark.unit
class TestMixedFormatLoading:
  """Tests for load_directory() with both flat and directory skills."""

  def test_loads_directory_skills(self, tmp_path):
    content = "---\nname: dir-skill\ndescription: D\n---\nBody."
    _make_skill_dir(tmp_path, "dir-skill", content)
    skills = SkillLoader.load_directory(tmp_path)
    names = [s.meta.name for s in skills]
    assert "dir-skill" in names

  def test_loads_flat_skills(self, tmp_path):
    (tmp_path / "flat.md").write_text("---\nname: flat\n---\nBody.", encoding="utf-8")
    skills = SkillLoader.load_directory(tmp_path)
    names = [s.meta.name for s in skills]
    assert "flat" in names

  def test_directory_wins_on_collision(self, tmp_path):
    # Both directory and flat have name "collision"
    _make_skill_dir(tmp_path, "collision", "---\nname: collision\n---\nDirectory version.")
    (tmp_path / "collision.md").write_text("---\nname: collision\n---\nFlat version.", encoding="utf-8")
    skills = SkillLoader.load_directory(tmp_path)
    collision_skills = [s for s in skills if s.meta.name == "collision"]
    assert len(collision_skills) == 1
    assert collision_skills[0].is_directory_skill is True

  def test_mixed_directory(self, tmp_path):
    _make_skill_dir(tmp_path, "dir-a", "---\nname: dir-a\n---\nA.")
    (tmp_path / "flat-b.md").write_text("---\nname: flat-b\n---\nB.", encoding="utf-8")
    skills = SkillLoader.load_directory(tmp_path)
    names = sorted(s.meta.name for s in skills)
    assert names == ["dir-a", "flat-b"]

  def test_nonexistent_directory(self, tmp_path):
    skills = SkillLoader.load_directory(tmp_path / "nope")
    assert skills == []

  def test_skill_md_in_root_ignored(self, tmp_path):
    """SKILL.md at root level (not in a subdirectory) should be ignored as a flat file."""
    (tmp_path / "SKILL.md").write_text("---\nname: root-skill\n---\nBody.", encoding="utf-8")
    skills = SkillLoader.load_directory(tmp_path)
    names = [s.meta.name for s in skills]
    assert "root-skill" not in names


# ── XML Prompt Generation ────────────────────────────────────────


@pytest.mark.unit
class TestXmlPrompt:
  """Tests for SkillRegistry.to_prompt()."""

  def test_basic_xml_output(self):
    s = _make_skill("test-skill", description="Test desc")
    reg = SkillRegistry(skills=[s], include_library=False)
    xml = reg.to_prompt()
    assert "<available_skills>" in xml
    assert "</available_skills>" in xml
    assert "<name>test-skill</name>" in xml
    assert "<description>Test desc</description>" in xml

  def test_xml_escaping(self):
    s = _make_skill("esc", description='Has <special> & "chars"')
    reg = SkillRegistry(skills=[s], include_library=False)
    xml = reg.to_prompt()
    assert "&lt;special&gt;" in xml
    assert "&amp;" in xml

  def test_multiple_skills_sorted(self):
    s1 = _make_skill("zebra", description="Last")
    s2 = _make_skill("alpha", description="First")
    reg = SkillRegistry(skills=[s1, s2], include_library=False)
    xml = reg.to_prompt()
    # alpha should appear before zebra
    alpha_pos = xml.index("alpha")
    zebra_pos = xml.index("zebra")
    assert alpha_pos < zebra_pos

  def test_empty_registry(self):
    reg = SkillRegistry(include_library=False)
    xml = reg.to_prompt()
    assert "<available_skills>" in xml
    assert "</available_skills>" in xml


# ── On-Demand Mode ───────────────────────────────────────────────


@pytest.mark.unit
class TestOnDemandMode:
  """Tests for SkillRegistry.as_on_demand()."""

  def test_returns_skill_with_three_tools(self):
    s = _make_skill("test", description="Test skill", content="Full instructions here.")
    reg = SkillRegistry(skills=[s], include_library=False)
    on_demand = reg.as_on_demand()
    assert on_demand.name == "skill_library"
    assert len(on_demand.tools) == 3
    tool_names = {t.name for t in on_demand.tools}
    assert tool_names == {"activate_skill", "read_skill_file", "run_skill_script"}

  def test_instructions_contain_xml_catalog(self):
    s = _make_skill("my-skill", description="Does things")
    reg = SkillRegistry(skills=[s], include_library=False)
    on_demand = reg.as_on_demand()
    instructions = on_demand.get_instructions()
    assert "<available_skills>" in instructions
    assert "<name>my-skill</name>" in instructions

  def test_activate_skill_tool_returns_content(self):
    s = _make_skill("target", description="Target skill", content="Full methodology.")
    reg = SkillRegistry(skills=[s], include_library=False)
    on_demand = reg.as_on_demand()
    activate = next(t for t in on_demand.tools if t.name == "activate_skill")
    result = activate.entrypoint(skill_name="target")
    assert 'name="target"' in result
    assert "Full methodology." in result

  def test_activate_nonexistent_skill(self):
    reg = SkillRegistry(skills=[_make_skill("x")], include_library=False)
    on_demand = reg.as_on_demand()
    activate = next(t for t in on_demand.tools if t.name == "activate_skill")
    result = activate.entrypoint(skill_name="nonexistent")
    assert "not found" in result.lower()

  def test_read_skill_file_tool(self, tmp_path):
    content = "---\nname: s\n---\nBody."
    extras = {"refs/guide.md": "# Guide\nContent here."}
    skill_dir = _make_skill_dir(tmp_path, "s", content, extras)
    skill = SkillLoader.load_skill_directory(skill_dir)
    reg = SkillRegistry(skills=[skill], include_library=False)
    on_demand = reg.as_on_demand()
    read_file = next(t for t in on_demand.tools if t.name == "read_skill_file")
    result = read_file.entrypoint(skill_name="s", file_path="refs/guide.md")
    assert "# Guide" in result

  def test_read_skill_file_not_found(self):
    s = _make_skill("no-dir")
    reg = SkillRegistry(skills=[s], include_library=False)
    on_demand = reg.as_on_demand()
    read_file = next(t for t in on_demand.tools if t.name == "read_skill_file")
    result = read_file.entrypoint(skill_name="no-dir", file_path="nope.txt")
    assert "error" in result.lower()

  def test_activate_skill_with_resources(self, tmp_path):
    content = "---\nname: rich\n---\nInstructions."
    extras = {"scripts/run.py": "print(1)", "refs/api.md": "API doc"}
    skill_dir = _make_skill_dir(tmp_path, "rich", content, extras)
    skill = SkillLoader.load_skill_directory(skill_dir)
    reg = SkillRegistry(skills=[skill], include_library=False)
    on_demand = reg.as_on_demand()
    activate = next(t for t in on_demand.tools if t.name == "activate_skill")
    result = activate.entrypoint(skill_name="rich")
    assert "<bundled_resources>" in result
    assert "scripts/run.py" in result


# ── Code Executor ────────────────────────────────────────────────


@pytest.mark.unit
class TestCodeExecutor:
  """Tests for SkillScriptExecutor."""

  def test_run_python_script(self, tmp_path):
    from definable.skill.executor import SkillScriptExecutor

    content = "---\nname: s\n---\nBody."
    extras = {"scripts/hello.py": "print('Hello from script')"}
    skill_dir = _make_skill_dir(tmp_path, "s", content, extras)
    skill = SkillLoader.load_skill_directory(skill_dir)
    executor = SkillScriptExecutor()
    result = asyncio.run(executor.run(skill, "scripts/hello.py"))
    assert "Hello from script" in result

  def test_run_bash_script(self, tmp_path):
    from definable.skill.executor import SkillScriptExecutor

    content = "---\nname: s\n---\nBody."
    extras = {"scripts/greet.sh": "#!/bin/bash\necho 'Hello bash'"}
    skill_dir = _make_skill_dir(tmp_path, "s", content, extras)
    skill = SkillLoader.load_skill_directory(skill_dir)
    executor = SkillScriptExecutor()
    result = asyncio.run(executor.run(skill, "scripts/greet.sh"))
    assert "Hello bash" in result

  def test_script_with_args(self, tmp_path):
    from definable.skill.executor import SkillScriptExecutor

    content = "---\nname: s\n---\nBody."
    extras = {"scripts/echo.py": "import sys; print(' '.join(sys.argv[1:]))"}
    skill_dir = _make_skill_dir(tmp_path, "s", content, extras)
    skill = SkillLoader.load_skill_directory(skill_dir)
    executor = SkillScriptExecutor()
    result = asyncio.run(executor.run(skill, "scripts/echo.py", ["foo", "bar"]))
    assert "foo bar" in result

  def test_timeout(self, tmp_path):
    from definable.skill.executor import SkillScriptExecutor

    content = "---\nname: s\n---\nBody."
    extras = {"scripts/slow.py": "import time; time.sleep(10)"}
    skill_dir = _make_skill_dir(tmp_path, "s", content, extras)
    skill = SkillLoader.load_skill_directory(skill_dir)
    executor = SkillScriptExecutor(timeout=1)
    result = asyncio.run(executor.run(skill, "scripts/slow.py"))
    assert "timed out" in result.lower()

  def test_path_escape(self, tmp_path):
    from definable.skill.executor import SkillScriptExecutor

    content = "---\nname: s\n---\nBody."
    skill_dir = _make_skill_dir(tmp_path, "s", content)
    skill = SkillLoader.load_skill_directory(skill_dir)
    executor = SkillScriptExecutor()
    result = asyncio.run(executor.run(skill, "../../etc/passwd"))
    assert "error" in result.lower()

  def test_output_truncation(self, tmp_path):
    from definable.skill.executor import SkillScriptExecutor

    content = "---\nname: s\n---\nBody."
    extras = {"scripts/big.py": "print('x' * 50000)"}
    skill_dir = _make_skill_dir(tmp_path, "s", content, extras)
    skill = SkillLoader.load_skill_directory(skill_dir)
    executor = SkillScriptExecutor(max_output=100)
    result = asyncio.run(executor.run(skill, "scripts/big.py"))
    assert "truncated" in result.lower()
    assert len(result) < 200  # truncated + message

  def test_nonexistent_script(self, tmp_path):
    from definable.skill.executor import SkillScriptExecutor

    content = "---\nname: s\n---\nBody."
    skill_dir = _make_skill_dir(tmp_path, "s", content)
    skill = SkillLoader.load_skill_directory(skill_dir)
    executor = SkillScriptExecutor()
    result = asyncio.run(executor.run(skill, "scripts/nope.py"))
    assert "not found" in result.lower()

  def test_unsupported_extension(self, tmp_path):
    from definable.skill.executor import SkillScriptExecutor

    content = "---\nname: s\n---\nBody."
    extras = {"scripts/file.xyz": "data"}
    skill_dir = _make_skill_dir(tmp_path, "s", content, extras)
    skill = SkillLoader.load_skill_directory(skill_dir)
    executor = SkillScriptExecutor()
    result = asyncio.run(executor.run(skill, "scripts/file.xyz"))
    assert "unsupported" in result.lower()

  def test_not_directory_skill(self):
    from definable.skill.executor import SkillScriptExecutor

    skill = _make_skill("flat")
    executor = SkillScriptExecutor()
    result = asyncio.run(executor.run(skill, "scripts/x.py"))
    assert "not a directory" in result.lower()


# ── Backward Compatibility ───────────────────────────────────────


@pytest.mark.unit
class TestBackwardCompat:
  """Tests ensuring the old format and old methods still work."""

  def test_old_flat_format_loads(self, tmp_path):
    (tmp_path / "old.md").write_text(
      "---\nname: old-skill\ndescription: Old format\nversion: 1.0.0\ntags: [a, b]\n---\nOld body.",
      encoding="utf-8",
    )
    skill = SkillLoader.load_file(tmp_path / "old.md")
    assert skill.meta.name == "old-skill"
    assert skill.meta.version == "1.0.0"
    assert skill.meta.tags == ["a", "b"]
    assert skill.is_directory_skill is False

  def test_as_eager_still_works(self):
    s = _make_skill("e1", content="Content 1")
    reg = SkillRegistry(skills=[s], include_library=False)
    eager = reg.as_eager()
    assert len(eager) == 1
    assert eager[0].get_instructions() != ""

  def test_as_lazy_still_works(self):
    s = _make_skill("l1", description="Lazy test")
    reg = SkillRegistry(skills=[s], include_library=False)
    lazy = reg.as_lazy()
    assert lazy.name == "skill_library"
    assert len(lazy.tools) == 1
    assert lazy.tools[0].name == "read_skill"

  def test_flat_skill_repr(self):
    skill = _make_skill("test", tags=["a"])
    assert "file" in repr(skill)

  def test_directory_skill_repr(self, tmp_path):
    content = "---\nname: d\n---\nBody."
    skill_dir = _make_skill_dir(tmp_path, "d", content)
    skill = SkillLoader.load_skill_directory(skill_dir)
    assert "dir" in repr(skill)


# ── Full Library ─────────────────────────────────────────────────


@pytest.mark.unit
class TestFullLibrary:
  """Tests for the complete built-in library (24 skills)."""

  def test_library_loads(self):
    reg = SkillRegistry(include_library=True)
    assert len(reg) >= 24

  def test_all_expected_skills_present(self):
    reg = SkillRegistry(include_library=True)
    expected = {
      "code-review",
      "data-analysis",
      "debug-code",
      "explain-concept",
      "plan-project",
      "summarize-document",
      "web-research",
      "write-report",
      "pdf",
      "xlsx",
      "docx",
      "pptx",
      "skill-creator",
      "mcp-builder",
      "algorithmic-art",
      "brand-guidelines",
      "canvas-design",
      "doc-coauthoring",
      "frontend-design",
      "internal-comms",
      "slack-gif-creator",
      "theme-factory",
      "web-artifacts-builder",
      "webapp-testing",
    }
    actual = {s.name for s in reg.list_skills()}
    missing = expected - actual
    assert not missing, f"Missing skills: {missing}"

  def test_to_prompt_includes_all_skills(self):
    reg = SkillRegistry(include_library=True)
    xml = reg.to_prompt()
    for meta in reg.list_skills():
      assert f"<name>{meta.name}</name>" in xml

  def test_on_demand_mode_with_full_library(self):
    reg = SkillRegistry(include_library=True)
    on_demand = reg.as_on_demand()
    assert len(on_demand.tools) == 3
    # Activate a known skill
    activate = next(t for t in on_demand.tools if t.name == "activate_skill")
    result = activate.entrypoint(skill_name="code-review")
    assert "code-review" in result


# ── Imports ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestImports:
  """Tests for public exports."""

  def test_validate_agent_skills_name_importable(self):
    from definable.skill import validate_agent_skills_name

    assert callable(validate_agent_skills_name)

  def test_skill_script_executor_importable(self):
    from definable.skill import SkillScriptExecutor

    assert SkillScriptExecutor is not None
