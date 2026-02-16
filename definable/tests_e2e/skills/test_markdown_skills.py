"""Tests for the markdown-based skills layer.

Covers:
- SkillLoader: parse valid/invalid markdown, frontmatter edge cases, load file/directory
- MarkdownSkill: is Skill subclass, get_instructions wraps in tags, meta accessible
- SkillRegistry: loads builtins, custom dirs, list/get/search, as_eager, as_lazy, len/contains
- Lazy mode: read_skill tool exists and works, catalog in instructions, not-found handling
- Agent integration: skill_registry param works (eager/lazy), combines with skills=
- Built-in library: all .md files load, required frontmatter present, unique names
"""

import tempfile
from pathlib import Path

import pytest

from definable.agents.agent import Agent
from definable.agents.config import AgentConfig, TracingConfig
from definable.agents.testing import MockModel
from definable.skills.base import Skill
from definable.skills.markdown import MarkdownSkill, MarkdownSkillMeta, SkillLoader
from definable.skills.registry import SkillRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_model():
  return MockModel(responses=["Mock response"])


@pytest.fixture
def no_trace_config():
  return AgentConfig(tracing=TracingConfig(enabled=False))


@pytest.fixture
def sample_md():
  return """---
name: test-skill
description: A test skill for unit tests
version: 2.0.0
tags: [test, example]
author: Test Author
requires_tools: [search_web]
---

## When to Use
Use this when testing.

## Steps
1. Do the thing.
2. Check the result.
"""


@pytest.fixture
def tmp_skill_dir(sample_md):
  """Create a temp directory with sample skill files."""
  with tempfile.TemporaryDirectory() as tmpdir:
    p = Path(tmpdir)

    # Valid skill
    (p / "test-skill.md").write_text(sample_md, encoding="utf-8")

    # Another valid skill
    (p / "other.md").write_text(
      "---\nname: other-skill\ndescription: Another skill\ntags: [other]\n---\n\nOther content.",
      encoding="utf-8",
    )

    # Invalid skill (no name)
    (p / "bad.md").write_text(
      "---\ndescription: No name field\n---\n\nBroken.",
      encoding="utf-8",
    )

    yield p


# ---------------------------------------------------------------------------
# Test: SkillLoader — parse_frontmatter
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
  """Test the frontmatter parser."""

  def test_parse_basic_frontmatter(self):
    text = "---\nname: my-skill\ndescription: A skill\n---\n\nBody text."
    fm, body = SkillLoader.parse_frontmatter(text)
    assert fm["name"] == "my-skill"
    assert fm["description"] == "A skill"
    assert body == "Body text."

  def test_parse_list_values(self):
    text = "---\ntags: [foo, bar, baz]\n---\n\nBody."
    fm, body = SkillLoader.parse_frontmatter(text)
    assert fm["tags"] == ["foo", "bar", "baz"]

  def test_parse_empty_list(self):
    text = "---\ntags: []\n---\n\nBody."
    fm, body = SkillLoader.parse_frontmatter(text)
    assert fm["tags"] == []

  def test_parse_quoted_values(self):
    text = '---\nname: "my skill"\ndescription: \'quoted\'\n---\n\nBody.'
    fm, body = SkillLoader.parse_frontmatter(text)
    assert fm["name"] == "my skill"
    assert fm["description"] == "quoted"

  def test_no_frontmatter_returns_full_text(self):
    text = "Just plain markdown.\nNo frontmatter here."
    fm, body = SkillLoader.parse_frontmatter(text)
    assert fm == {}
    assert body == text

  def test_missing_closing_delimiter_raises(self):
    text = "---\nname: broken\n\nNo closing delimiter."
    with pytest.raises(ValueError, match="no closing"):
      SkillLoader.parse_frontmatter(text)

  def test_comments_in_frontmatter_ignored(self):
    text = "---\nname: my-skill\n# this is a comment\ntags: [a]\n---\n\nBody."
    fm, body = SkillLoader.parse_frontmatter(text)
    assert fm["name"] == "my-skill"
    assert fm["tags"] == ["a"]
    assert "#" not in fm

  def test_leading_newlines_stripped(self):
    text = "\n\n---\nname: test\n---\n\nBody."
    fm, body = SkillLoader.parse_frontmatter(text)
    assert fm["name"] == "test"


# ---------------------------------------------------------------------------
# Test: SkillLoader — parse
# ---------------------------------------------------------------------------


class TestSkillLoaderParse:
  """Test parsing markdown text into MarkdownSkill."""

  def test_parse_valid_skill(self, sample_md):
    skill = SkillLoader.parse(sample_md)
    assert skill.meta.name == "test-skill"
    assert skill.meta.description == "A test skill for unit tests"
    assert skill.meta.version == "2.0.0"
    assert skill.meta.tags == ["test", "example"]
    assert skill.meta.author == "Test Author"
    assert skill.meta.requires_tools == ["search_web"]

  def test_parse_missing_name_raises(self):
    text = "---\ndescription: no name\n---\n\nBody."
    with pytest.raises(ValueError, match="must include 'name'"):
      SkillLoader.parse(text)

  def test_parse_minimal_skill(self):
    text = "---\nname: minimal\n---\n\nJust a body."
    skill = SkillLoader.parse(text)
    assert skill.meta.name == "minimal"
    assert skill.meta.description == ""
    assert skill.meta.tags == []
    assert skill.meta.version == "1.0.0"

  def test_parse_preserves_source_path(self):
    text = "---\nname: test\n---\n\nBody."
    path = Path("/tmp/test.md")
    skill = SkillLoader.parse(text, source_path=path)
    assert skill._source_path == path


# ---------------------------------------------------------------------------
# Test: SkillLoader — load_file / load_directory
# ---------------------------------------------------------------------------


class TestSkillLoaderIO:
  """Test file and directory loading."""

  def test_load_file(self, tmp_skill_dir):
    skill = SkillLoader.load_file(tmp_skill_dir / "test-skill.md")
    assert skill.meta.name == "test-skill"

  def test_load_file_not_found(self):
    with pytest.raises(FileNotFoundError):
      SkillLoader.load_file(Path("/nonexistent/skill.md"))

  def test_load_directory(self, tmp_skill_dir):
    skills = SkillLoader.load_directory(tmp_skill_dir)
    # 2 valid out of 3 files (bad.md has no name)
    assert len(skills) == 2
    names = {s.meta.name for s in skills}
    assert "test-skill" in names
    assert "other-skill" in names

  def test_load_directory_nonexistent(self):
    skills = SkillLoader.load_directory(Path("/nonexistent/dir"))
    assert skills == []


# ---------------------------------------------------------------------------
# Test: MarkdownSkill
# ---------------------------------------------------------------------------


class TestMarkdownSkill:
  """Test MarkdownSkill behavior."""

  def test_is_skill_subclass(self, sample_md):
    skill = SkillLoader.parse(sample_md)
    assert isinstance(skill, Skill)
    assert isinstance(skill, MarkdownSkill)

  def test_get_instructions_wraps_in_tags(self, sample_md):
    skill = SkillLoader.parse(sample_md)
    instructions = skill.get_instructions()
    assert instructions.startswith('<skill name="test-skill">')
    assert instructions.endswith("</skill>")
    assert "When to Use" in instructions

  def test_empty_content_returns_empty_instructions(self):
    meta = MarkdownSkillMeta(name="empty")
    skill = MarkdownSkill(meta=meta, content="")
    assert skill.get_instructions() == ""

  def test_tools_returns_empty_list(self, sample_md):
    skill = SkillLoader.parse(sample_md)
    assert skill.tools == []

  def test_name_matches_meta(self, sample_md):
    skill = SkillLoader.parse(sample_md)
    assert skill.name == skill.meta.name == "test-skill"

  def test_repr(self, sample_md):
    skill = SkillLoader.parse(sample_md)
    r = repr(skill)
    assert "test-skill" in r
    assert "MarkdownSkill" in r


# ---------------------------------------------------------------------------
# Test: SkillRegistry
# ---------------------------------------------------------------------------


class TestSkillRegistry:
  """Test SkillRegistry functionality."""

  def test_loads_builtin_library(self):
    registry = SkillRegistry()
    assert len(registry) > 0
    assert "web-research" in registry
    assert "code-review" in registry

  def test_include_library_false(self):
    registry = SkillRegistry(include_library=False)
    assert len(registry) == 0

  def test_custom_directory(self, tmp_skill_dir):
    registry = SkillRegistry(
      include_library=False,
      directories=[tmp_skill_dir],
    )
    assert len(registry) == 2
    assert "test-skill" in registry

  def test_explicit_skills(self, sample_md):
    skill = SkillLoader.parse(sample_md)
    registry = SkillRegistry(
      include_library=False,
      skills=[skill],
    )
    assert len(registry) == 1
    assert "test-skill" in registry

  def test_dedup_last_wins(self, sample_md):
    """When skills have the same name, the last one added wins."""
    skill1 = SkillLoader.parse(sample_md)
    text2 = "---\nname: test-skill\ndescription: Override\n---\n\nNew content."
    skill2 = SkillLoader.parse(text2)
    registry = SkillRegistry(
      include_library=False,
      skills=[skill1, skill2],
    )
    assert len(registry) == 1
    assert registry.get_skill("test-skill").meta.description == "Override"

  def test_list_skills(self):
    registry = SkillRegistry()
    metas = registry.list_skills()
    assert len(metas) == len(registry)
    assert all(isinstance(m, MarkdownSkillMeta) for m in metas)
    # Should be sorted by name
    names = [m.name for m in metas]
    assert names == sorted(names)

  def test_get_skill_found(self):
    registry = SkillRegistry()
    skill = registry.get_skill("web-research")
    assert skill is not None
    assert skill.meta.name == "web-research"

  def test_get_skill_not_found(self):
    registry = SkillRegistry()
    assert registry.get_skill("nonexistent") is None

  def test_search_skills(self):
    registry = SkillRegistry()
    results = registry.search_skills("code")
    assert len(results) > 0
    # code-review should rank high
    names = [s.meta.name for s in results]
    assert "code-review" in names

  def test_search_skills_by_tag(self):
    registry = SkillRegistry()
    results = registry.search_skills("debug")
    names = [s.meta.name for s in results]
    assert "debug-code" in names

  def test_search_no_results(self):
    registry = SkillRegistry()
    results = registry.search_skills("xyznonexistent123")
    assert results == []

  def test_as_eager(self):
    registry = SkillRegistry()
    skills = registry.as_eager()
    assert len(skills) == len(registry)
    assert all(isinstance(s, Skill) for s in skills)

  def test_as_lazy(self):
    registry = SkillRegistry()
    wrapper = registry.as_lazy()
    assert isinstance(wrapper, Skill)
    assert wrapper.name == "skill_library"
    # Should have instructions (catalog) and tools (read_skill)
    assert "Available Skills" in wrapper.get_instructions()
    assert len(wrapper.tools) == 1
    assert wrapper.tools[0].name == "read_skill"

  def test_contains(self):
    registry = SkillRegistry()
    assert "web-research" in registry
    assert "nonexistent" not in registry

  def test_repr(self):
    registry = SkillRegistry()
    r = repr(registry)
    assert "SkillRegistry" in r
    assert "skills=" in r


# ---------------------------------------------------------------------------
# Test: Lazy mode
# ---------------------------------------------------------------------------


class TestLazyMode:
  """Test lazy mode with read_skill tool."""

  def test_read_skill_returns_content(self):
    registry = SkillRegistry()
    wrapper = registry.as_lazy()
    read_tool = wrapper.tools[0]
    # Call the tool's entrypoint directly
    result = read_tool.entrypoint(skill_name="web-research")
    assert '<skill name="web-research">' in result
    assert "When to Use" in result

  def test_read_skill_not_found(self):
    registry = SkillRegistry()
    wrapper = registry.as_lazy()
    read_tool = wrapper.tools[0]
    result = read_tool.entrypoint(skill_name="nonexistent")
    assert "not found" in result
    assert "Available skills:" in result

  def test_catalog_contains_all_skills(self):
    registry = SkillRegistry()
    wrapper = registry.as_lazy()
    catalog = wrapper.get_instructions()
    for meta in registry.list_skills():
      assert meta.name in catalog


# ---------------------------------------------------------------------------
# Test: Agent integration
# ---------------------------------------------------------------------------


class TestAgentIntegration:
  """Test skill_registry param on Agent."""

  def test_skill_registry_eager(self, mock_model, no_trace_config):
    """Small registry injects skills eagerly."""
    registry = SkillRegistry()
    assert len(registry) <= 15  # should be 8 builtins

    agent = Agent(
      model=mock_model,
      skill_registry=registry,
      config=no_trace_config,
    )
    # All registry skills should be in agent.skills
    instructions = agent._build_skill_instructions()
    assert "When to Use" in instructions

  def test_skill_registry_lazy_threshold(self, mock_model, no_trace_config, tmp_skill_dir):
    """Large registry switches to lazy mode."""
    # Create enough skills to exceed threshold
    skills = []
    for i in range(20):
      text = f"---\nname: skill-{i}\ndescription: Skill number {i}\ntags: [test]\n---\n\nContent {i}."
      skills.append(SkillLoader.parse(text))

    registry = SkillRegistry(
      include_library=False,
      skills=skills,
    )
    assert len(registry) > 15

    agent = Agent(
      model=mock_model,
      skill_registry=registry,
      config=no_trace_config,
    )
    # Should have the lazy skill_library skill with read_skill tool
    assert "read_skill" in agent.tool_names
    instructions = agent._build_skill_instructions()
    assert "Available Skills" in instructions

  def test_skill_registry_combines_with_skills(self, mock_model, no_trace_config):
    """skill_registry skills combine with explicit skills=."""
    from definable.skills import Calculator

    registry = SkillRegistry()
    agent = Agent(
      model=mock_model,
      skills=[Calculator()],
      skill_registry=registry,
      config=no_trace_config,
    )
    # Should have calculator tool + registry skills
    assert "calculate" in agent.tool_names
    instructions = agent._build_skill_instructions()
    assert "When to Use" in instructions

  @pytest.mark.asyncio
  async def test_skill_instructions_in_system_prompt(self, mock_model, no_trace_config):
    """Registry skill instructions appear in the system message."""
    registry = SkillRegistry()
    agent = Agent(
      model=mock_model,
      skill_registry=registry,
      instructions="Base instructions.",
      config=no_trace_config,
    )
    await agent.arun("Hello")

    last_call = mock_model.call_history[-1]
    messages = last_call["messages"]
    system_msg = messages[0]
    assert system_msg.role == "system"
    assert "Base instructions" in system_msg.content
    assert '<skill name="' in system_msg.content


# ---------------------------------------------------------------------------
# Test: Built-in library
# ---------------------------------------------------------------------------


class TestBuiltinLibrary:
  """Test the built-in markdown skill library."""

  def test_all_md_files_load(self):
    """All .md files in the library directory parse successfully."""
    library_dir = Path(__file__).parent.parent.parent / "definable" / "skills" / "library"
    if not library_dir.is_dir():
      pytest.skip("Library directory not found")

    skills = SkillLoader.load_directory(library_dir)
    md_count = len(list(library_dir.glob("*.md")))
    assert len(skills) == md_count
    assert md_count == 8

  def test_all_have_required_frontmatter(self):
    """Every built-in skill has name, description, and tags."""
    registry = SkillRegistry(include_library=True)
    for meta in registry.list_skills():
      assert meta.name, f"Skill missing name: {meta}"
      assert meta.description, f"Skill {meta.name} missing description"
      assert meta.tags, f"Skill {meta.name} missing tags"

  def test_unique_names(self):
    """All built-in skills have unique names."""
    registry = SkillRegistry(include_library=True)
    metas = registry.list_skills()
    names = [m.name for m in metas]
    assert len(names) == len(set(names)), f"Duplicate skill names: {names}"

  def test_expected_skills_present(self):
    """All 8 expected skills are in the library."""
    registry = SkillRegistry(include_library=True)
    expected = {
      "web-research",
      "code-review",
      "debug-code",
      "summarize-document",
      "write-report",
      "data-analysis",
      "plan-project",
      "explain-concept",
    }
    actual = {m.name for m in registry.list_skills()}
    assert expected == actual
