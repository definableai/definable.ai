"""Unit tests for the Skill base class.

Tests cover instantiation, tool resolution, instructions,
naming defaults, and lifecycle hooks.
"""

import pytest

from definable.skill.base import Skill
from definable.tool.decorator import tool
from definable.tool.function import Function


@pytest.mark.unit
class TestSkillInstantiation:
  """Tests for Skill.__init__ and attribute storage."""

  def test_skill_with_name_and_tools(self):
    """Skill stores name and explicit tools."""

    @tool
    def dummy(x: str) -> str:
      """A dummy tool."""
      return x

    s = Skill(name="my_skill", tools=[dummy])
    assert s.name == "my_skill"
    assert len(s.tools) == 1
    assert isinstance(s.tools[0], Function)

  def test_skill_name_defaults_to_class_name(self):
    """When no name is given the class name is used."""
    s = Skill()
    assert s.name == "Skill"

  def test_skill_name_override(self):
    """Explicit name kwarg overrides the class-level attribute."""

    class MySkill(Skill):
      name = "class_level"

    s = MySkill(name="override_name")
    assert s.name == "override_name"

  def test_skill_class_attribute_name(self):
    """Subclass with a class-level name attribute uses that name."""

    class Named(Skill):
      name = "from_class"

    s = Named()
    assert s.name == "from_class"

  def test_skill_name_stored_correctly(self):
    """Name passed to init is stored as the .name attribute."""
    s = Skill(name="precise_name")
    assert s.name == "precise_name"


@pytest.mark.unit
class TestSkillTools:
  """Tests for Skill.tools property."""

  def test_tools_returns_list_of_function_objects(self):
    """Explicit tools are returned as a list of Function objects."""

    @tool
    def alpha(a: int) -> int:
      """Add one."""
      return a + 1

    @tool
    def beta(b: str) -> str:
      """Echo."""
      return b

    s = Skill(name="multi", tools=[alpha, beta])
    assert len(s.tools) == 2
    assert all(isinstance(t, Function) for t in s.tools)

  def test_empty_tools_list(self):
    """Skill created with an empty tools list returns empty list."""
    s = Skill(name="empty", tools=[])
    assert s.tools == []

  def test_no_explicit_tools_auto_discovers_none_on_base(self):
    """Base Skill with no explicit tools discovers no tools."""
    s = Skill(name="bare")
    assert s.tools == []

  def test_tools_returns_copy(self):
    """Modifying the returned list does not affect the skill."""

    @tool
    def t(x: str) -> str:
      """T."""
      return x

    s = Skill(name="copy_test", tools=[t])
    returned = s.tools
    returned.clear()
    assert len(s.tools) == 1


@pytest.mark.unit
class TestSkillInstructions:
  """Tests for Skill.instructions and get_instructions()."""

  def test_instructions_from_init(self):
    """Instructions passed at init are accessible via .instructions."""
    s = Skill(name="inst", instructions="Be helpful.")
    assert s.instructions == "Be helpful."

  def test_get_instructions_returns_instructions(self):
    """get_instructions() returns the same string as .instructions."""
    s = Skill(name="inst2", instructions="Do things.")
    assert s.get_instructions() == "Do things."

  def test_default_instructions_are_empty(self):
    """Skill with no instructions has empty-string instructions."""
    s = Skill(name="no_inst")
    assert s.instructions == ""
    assert s.get_instructions() == ""

  def test_class_level_instructions_override(self):
    """init instructions override class-level instructions."""

    class MySkill(Skill):
      instructions = "class level"

    s = MySkill(instructions="init level")
    assert s.instructions == "init level"


@pytest.mark.unit
class TestSkillDependencies:
  """Tests for Skill.dependencies."""

  def test_dependencies_default_empty(self):
    """Dependencies default to an empty dict."""
    s = Skill(name="dep")
    assert s.dependencies == {}

  def test_dependencies_stored(self):
    """Explicit dependencies are stored and accessible."""
    s = Skill(name="dep2", dependencies={"db_url": "sqlite:///test.db"})
    assert s.dependencies["db_url"] == "sqlite:///test.db"


@pytest.mark.unit
class TestSkillLifecycle:
  """Tests for setup/teardown hooks."""

  def test_setup_is_noop_by_default(self):
    """Base Skill.setup() does not raise."""
    s = Skill(name="lc")
    s.setup()  # should not raise

  def test_teardown_is_noop_by_default(self):
    """Base Skill.teardown() does not raise."""
    s = Skill(name="lc2")
    s.teardown()  # should not raise


@pytest.mark.unit
class TestSkillRepr:
  """Tests for Skill.__repr__."""

  def test_repr_includes_name_and_tool_count(self):
    """__repr__ shows name, tool count, and instructions flag."""

    @tool
    def t(x: str) -> str:
      """T."""
      return x

    s = Skill(name="my_repr", tools=[t], instructions="yes")
    r = repr(s)
    assert "my_repr" in r
    assert "tools=1" in r
    assert "instructions=True" in r

  def test_repr_no_instructions(self):
    """__repr__ shows instructions=False when no instructions."""
    s = Skill(name="bare_repr")
    r = repr(s)
    assert "instructions=False" in r
