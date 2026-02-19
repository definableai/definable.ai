"""Unit tests for SkillRegistry.

Tests cover creation, skill lookup, listing, search, and
container protocol methods (__len__, __contains__).
"""

import pytest

from definable.skill.markdown import MarkdownSkill, MarkdownSkillMeta
from definable.skill.registry import SkillRegistry


def _make_skill(name: str, description: str = "", tags: list | None = None, content: str = "body") -> MarkdownSkill:
  """Helper to build a MarkdownSkill without file I/O."""
  return MarkdownSkill(
    meta=MarkdownSkillMeta(
      name=name,
      description=description,
      tags=tags or [],
    ),
    content=content,
  )


@pytest.mark.unit
class TestSkillRegistryCreation:
  """Tests for SkillRegistry instantiation."""

  def test_registry_can_be_created_empty(self):
    """SkillRegistry with no library and no skills is empty."""
    reg = SkillRegistry(include_library=False)
    assert len(reg) == 0

  def test_registry_with_explicit_skills(self):
    """SkillRegistry populated with explicit skills contains them."""
    s1 = _make_skill("alpha")
    s2 = _make_skill("beta")
    reg = SkillRegistry(skills=[s1, s2], include_library=False)
    assert len(reg) == 2

  def test_registry_repr(self):
    """__repr__ shows the skill count."""
    reg = SkillRegistry(include_library=False)
    assert "SkillRegistry(skills=0)" in repr(reg)


@pytest.mark.unit
class TestGetSkill:
  """Tests for SkillRegistry.get_skill()."""

  def test_get_skill_returns_skill_by_name(self):
    """get_skill returns the correct MarkdownSkill for a known name."""
    s = _make_skill("lookup_me", description="found")
    reg = SkillRegistry(skills=[s], include_library=False)
    result = reg.get_skill("lookup_me")
    assert result is not None
    assert result.meta.name == "lookup_me"

  def test_get_nonexistent_skill_returns_none(self):
    """get_skill returns None for an unknown name."""
    reg = SkillRegistry(include_library=False)
    assert reg.get_skill("nonexistent") is None


@pytest.mark.unit
class TestListSkills:
  """Tests for SkillRegistry.list_skills()."""

  def test_list_skills_returns_all_metadata(self):
    """list_skills returns MarkdownSkillMeta for every registered skill."""
    s1 = _make_skill("aaa")
    s2 = _make_skill("bbb")
    reg = SkillRegistry(skills=[s1, s2], include_library=False)
    metas = reg.list_skills()
    names = [m.name for m in metas]
    assert "aaa" in names
    assert "bbb" in names

  def test_list_skills_sorted_by_name(self):
    """list_skills returns results sorted alphabetically by name."""
    s1 = _make_skill("zebra")
    s2 = _make_skill("apple")
    reg = SkillRegistry(skills=[s1, s2], include_library=False)
    metas = reg.list_skills()
    assert metas[0].name == "apple"
    assert metas[1].name == "zebra"

  def test_list_skills_empty_registry(self):
    """list_skills on an empty registry returns an empty list."""
    reg = SkillRegistry(include_library=False)
    assert reg.list_skills() == []


@pytest.mark.unit
class TestSearchSkills:
  """Tests for SkillRegistry.search_skills()."""

  def test_search_by_name(self):
    """search_skills finds skills whose name matches the query."""
    s = _make_skill("code-review", description="Review code changes", tags=["code"])
    reg = SkillRegistry(skills=[s], include_library=False)
    results = reg.search_skills("code")
    assert len(results) >= 1
    assert results[0].meta.name == "code-review"

  def test_search_no_match(self):
    """search_skills returns empty when no skill matches."""
    s = _make_skill("alpha")
    reg = SkillRegistry(skills=[s], include_library=False)
    results = reg.search_skills("zzz_no_match")
    assert results == []


@pytest.mark.unit
class TestContainerProtocol:
  """Tests for __len__ and __contains__."""

  def test_len(self):
    """len(registry) matches the number of skills."""
    reg = SkillRegistry(
      skills=[_make_skill("x"), _make_skill("y")],
      include_library=False,
    )
    assert len(reg) == 2

  def test_contains(self):
    """'name' in registry returns True for registered names."""
    reg = SkillRegistry(skills=[_make_skill("present")], include_library=False)
    assert "present" in reg
    assert "absent" not in reg


@pytest.mark.unit
class TestAsEager:
  """Tests for SkillRegistry.as_eager()."""

  def test_as_eager_returns_all_skills(self):
    """as_eager returns a list of all registered skills."""
    s1 = _make_skill("e1")
    s2 = _make_skill("e2")
    reg = SkillRegistry(skills=[s1, s2], include_library=False)
    eager = reg.as_eager()
    assert len(eager) == 2


@pytest.mark.unit
class TestAsLazy:
  """Tests for SkillRegistry.as_lazy()."""

  def test_as_lazy_returns_single_skill(self):
    """as_lazy returns a single Skill with catalog instructions and a read_skill tool."""
    s = _make_skill("lazy_target", description="A lazy skill")
    reg = SkillRegistry(skills=[s], include_library=False)
    lazy = reg.as_lazy()
    assert lazy.name == "skill_library"
    assert "lazy_target" in lazy.get_instructions()
    assert len(lazy.tools) == 1
    assert lazy.tools[0].name == "read_skill"
