"""Tests for the Skills layer integration with Agent.

These tests verify that skills are properly integrated into the agent:
- Tools from skills are flattened alongside toolkit/direct tools
- Skill instructions are merged into the system prompt
- Skill lifecycle (setup/teardown) is called
- Skill dependencies are injected into tools
- Priority ordering: direct tools > toolkit tools > skill tools
- Built-in skills work correctly
- Custom skills (class-based and inline) work correctly
"""

import pytest

from definable.agents.agent import Agent
from definable.agents.config import AgentConfig, TracingConfig
from definable.agents.testing import MockModel
from definable.agents.toolkit import Toolkit
from definable.skills.base import Skill
from definable.tools.decorator import tool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_model():
  return MockModel(responses=["Mock response"])


@pytest.fixture
def no_trace_config():
  return AgentConfig(tracing=TracingConfig(enabled=False))


# ---------------------------------------------------------------------------
# Helper skills for testing
# ---------------------------------------------------------------------------


@tool
def greet(name: str) -> str:
  """Greet someone by name."""
  return f"Hello, {name}!"


@tool
def farewell(name: str) -> str:
  """Say goodbye to someone."""
  return f"Goodbye, {name}!"


@tool
def add(a: int, b: int) -> str:
  """Add two numbers."""
  return str(a + b)


class GreeterSkill(Skill):
  """Test skill that provides greeting tools."""

  name = "greeter"
  instructions = "You are a friendly greeter. Always use the greet tool."

  def __init__(self):
    super().__init__()

  @property
  def tools(self):
    return [greet, farewell]


class MathSkill(Skill):
  """Test skill for math with dependencies."""

  name = "math"
  instructions = "Use the add tool for arithmetic."

  def __init__(self, precision: int = 2):
    super().__init__(dependencies={"precision": precision})

  @property
  def tools(self):
    return [add]


class LifecycleSkill(Skill):
  """Skill that tracks setup/teardown calls."""

  name = "lifecycle"
  instructions = "Lifecycle test skill."

  def __init__(self):
    super().__init__()
    self.setup_called = False
    self.teardown_called = False
    self.setup_count = 0
    self.teardown_count = 0

  @property
  def tools(self):
    return []

  def setup(self):
    self.setup_called = True
    self.setup_count += 1

  def teardown(self):
    self.teardown_called = True
    self.teardown_count += 1


class FailingSetupSkill(Skill):
  """Skill whose setup() raises an exception."""

  name = "failing_setup"
  instructions = "This skill fails on setup."

  def setup(self):
    raise RuntimeError("Setup failed intentionally")

  @property
  def tools(self):
    return [greet]


class DynamicInstructionsSkill(Skill):
  """Skill with dynamic instructions based on config."""

  name = "dynamic"

  def __init__(self, mode: str = "polite"):
    super().__init__()
    self._mode = mode

  def get_instructions(self) -> str:
    if self._mode == "polite":
      return "Always be polite and formal."
    return "Be casual and direct."

  @property
  def tools(self):
    return []


# ---------------------------------------------------------------------------
# Test: Agent constructor with skills
# ---------------------------------------------------------------------------


class TestSkillsIntegration:
  """Test that skills are properly wired into the Agent."""

  def test_agent_accepts_skills_parameter(self, mock_model, no_trace_config):
    """Agent constructor accepts skills=[] without error."""
    agent = Agent(
      model=mock_model,
      skills=[GreeterSkill()],
      config=no_trace_config,
    )
    assert len(agent.skills) == 1
    assert agent.skills[0].name == "greeter"

  def test_agent_defaults_to_empty_skills(self, mock_model, no_trace_config):
    """Without skills param, agent.skills is empty list."""
    agent = Agent(model=mock_model, config=no_trace_config)
    assert agent.skills == []

  def test_multiple_skills(self, mock_model, no_trace_config):
    """Agent accepts multiple skills."""
    agent = Agent(
      model=mock_model,
      skills=[GreeterSkill(), MathSkill()],
      config=no_trace_config,
    )
    assert len(agent.skills) == 2

  def test_skills_with_tools_and_toolkits(self, mock_model, no_trace_config):
    """Skills work alongside direct tools and toolkits."""

    @tool
    def direct_tool(x: str) -> str:
      """A direct tool."""
      return x

    agent = Agent(
      model=mock_model,
      tools=[direct_tool],
      skills=[GreeterSkill()],
      config=no_trace_config,
    )
    # Should have: greet, farewell (from skill) + direct_tool
    assert "greet" in agent.tool_names
    assert "farewell" in agent.tool_names
    assert "direct_tool" in agent.tool_names

  def test_repr_includes_skills(self, mock_model, no_trace_config):
    """Agent repr shows skill count when skills are present."""
    agent = Agent(
      model=mock_model,
      skills=[GreeterSkill()],
      config=no_trace_config,
    )
    repr_str = repr(agent)
    assert "skills=1" in repr_str

  def test_repr_without_skills(self, mock_model, no_trace_config):
    """Agent repr omits skills when none are present."""
    agent = Agent(model=mock_model, config=no_trace_config)
    repr_str = repr(agent)
    assert "skills=" not in repr_str


# ---------------------------------------------------------------------------
# Test: Tool flattening
# ---------------------------------------------------------------------------


class TestToolFlattening:
  """Test that skill tools are properly flattened into the agent's tool dict."""

  def test_skill_tools_available(self, mock_model, no_trace_config):
    """Skill tools appear in agent.tool_names."""
    agent = Agent(
      model=mock_model,
      skills=[GreeterSkill()],
      config=no_trace_config,
    )
    assert "greet" in agent.tool_names
    assert "farewell" in agent.tool_names

  def test_skill_tools_combined_with_direct(self, mock_model, no_trace_config):
    """Skill tools and direct tools are all available."""

    @tool
    def my_tool(x: str) -> str:
      """My tool."""
      return x

    agent = Agent(
      model=mock_model,
      tools=[my_tool],
      skills=[MathSkill()],
      config=no_trace_config,
    )
    assert "add" in agent.tool_names
    assert "my_tool" in agent.tool_names

  def test_direct_tools_override_skill_tools(self, mock_model, no_trace_config):
    """Direct tools with same name override skill tools (highest priority)."""

    @tool
    def greet(name: str) -> str:
      """Custom greet that overrides skill's greet."""
      return f"Custom hello, {name}!"

    agent = Agent(
      model=mock_model,
      tools=[greet],
      skills=[GreeterSkill()],
      config=no_trace_config,
    )
    # greet should be the direct tool's version
    assert "greet" in agent.tool_names
    # farewell comes from skill (no override)
    assert "farewell" in agent.tool_names

  def test_toolkit_tools_override_skill_tools(self, mock_model, no_trace_config):
    """Toolkit tools override skill tools of the same name."""

    class GreeterToolkit(Toolkit):
      @property
      def tools(self):
        @tool
        def greet(name: str) -> str:
          """Toolkit greet."""
          return f"Toolkit hello, {name}!"

        return [greet]

    agent = Agent(
      model=mock_model,
      toolkits=[GreeterToolkit()],
      skills=[GreeterSkill()],
      config=no_trace_config,
    )
    # greet from toolkit overrides greet from skill
    assert "greet" in agent.tool_names
    # farewell from skill still available
    assert "farewell" in agent.tool_names

  def test_tool_count_from_multiple_skills(self, mock_model, no_trace_config):
    """Tools from multiple skills are all collected."""
    agent = Agent(
      model=mock_model,
      skills=[GreeterSkill(), MathSkill()],
      config=no_trace_config,
    )
    # greet + farewell from GreeterSkill + add from MathSkill
    assert len(agent.tool_names) == 3


# ---------------------------------------------------------------------------
# Test: Instruction merging
# ---------------------------------------------------------------------------


class TestInstructionMerging:
  """Test that skill instructions are merged into the system prompt."""

  def test_skill_instructions_built(self, mock_model, no_trace_config):
    """_build_skill_instructions returns combined instructions."""
    agent = Agent(
      model=mock_model,
      skills=[GreeterSkill(), MathSkill()],
      config=no_trace_config,
    )
    merged = agent._build_skill_instructions()
    assert "friendly greeter" in merged
    assert "add tool" in merged

  def test_empty_skills_no_instructions(self, mock_model, no_trace_config):
    """No skills means empty skill instructions."""
    agent = Agent(model=mock_model, config=no_trace_config)
    assert agent._build_skill_instructions() == ""

  def test_skill_without_instructions_skipped(self, mock_model, no_trace_config):
    """Skills with empty instructions don't add blank lines."""
    empty_skill = Skill(name="empty", instructions="")
    agent = Agent(
      model=mock_model,
      skills=[empty_skill, GreeterSkill()],
      config=no_trace_config,
    )
    merged = agent._build_skill_instructions()
    assert "friendly greeter" in merged
    assert not merged.startswith("\n")

  def test_dynamic_instructions(self, mock_model, no_trace_config):
    """Skills with get_instructions() override produce dynamic content."""
    agent = Agent(
      model=mock_model,
      skills=[DynamicInstructionsSkill(mode="polite")],
      config=no_trace_config,
    )
    assert "polite and formal" in agent._build_skill_instructions()

    agent2 = Agent(
      model=mock_model,
      skills=[DynamicInstructionsSkill(mode="casual")],
      config=no_trace_config,
    )
    assert "casual and direct" in agent2._build_skill_instructions()

  @pytest.mark.asyncio
  async def test_instructions_injected_into_system_prompt(self, mock_model, no_trace_config):
    """Skill instructions are in the system message sent to the model."""
    agent = Agent(
      model=mock_model,
      instructions="You are a test bot.",
      skills=[GreeterSkill()],
      config=no_trace_config,
    )
    await agent.arun("Hello")

    # Check the messages sent to the model
    assert mock_model.call_count >= 1
    last_call = mock_model.call_history[-1]
    messages = last_call["messages"]
    system_msg = messages[0]
    assert system_msg.role == "system"
    # Agent instructions + skill instructions
    assert "test bot" in system_msg.content
    assert "friendly greeter" in system_msg.content


# ---------------------------------------------------------------------------
# Test: Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
  """Test skill setup() and teardown() lifecycle hooks."""

  def test_setup_called_on_init(self, mock_model, no_trace_config):
    """Skill.setup() is called when agent is initialized."""
    skill = LifecycleSkill()
    assert not skill.setup_called

    Agent(model=mock_model, skills=[skill], config=no_trace_config)

    assert skill.setup_called  # type: ignore[unreachable]
    assert skill.setup_count == 1  # type: ignore[unreachable]

  def test_teardown_called_on_context_exit(self, mock_model, no_trace_config):
    """Skill.teardown() is called when agent context manager exits."""
    skill = LifecycleSkill()
    agent = Agent(model=mock_model, skills=[skill], config=no_trace_config)

    assert not skill.teardown_called

    with agent:
      pass

    assert skill.teardown_called  # type: ignore[unreachable]
    assert skill.teardown_count == 1  # type: ignore[unreachable]

  def test_failing_setup_does_not_crash_agent(self, mock_model, no_trace_config):
    """A skill whose setup() raises does not prevent agent creation."""
    # Should not raise
    agent = Agent(
      model=mock_model,
      skills=[FailingSetupSkill()],
      config=no_trace_config,
    )
    # The skill's tools should still be available
    assert "greet" in agent.tool_names

  def test_multiple_skills_all_setup(self, mock_model, no_trace_config):
    """All skills get their setup() called."""
    skill1 = LifecycleSkill()
    skill2 = LifecycleSkill()
    skill2.name = "lifecycle2"

    Agent(model=mock_model, skills=[skill1, skill2], config=no_trace_config)

    assert skill1.setup_called
    assert skill2.setup_called


# ---------------------------------------------------------------------------
# Test: Dependencies
# ---------------------------------------------------------------------------


class TestSkillDependencies:
  """Test that skill dependencies are injected into tools."""

  def test_skill_dependencies_on_tools(self, mock_model, no_trace_config):
    """Skill dependencies are merged into tool's _dependencies."""
    agent = Agent(
      model=mock_model,
      skills=[MathSkill(precision=4)],
      config=no_trace_config,
    )
    # The add tool should have precision dependency
    add_tool = agent._tools_dict["add"]
    assert add_tool._dependencies.get("precision") == 4


# ---------------------------------------------------------------------------
# Test: Built-in skills
# ---------------------------------------------------------------------------


class TestBuiltinSkills:
  """Test that built-in skills can be instantiated and used."""

  def test_calculator_skill(self, mock_model, no_trace_config):
    from definable.skills import Calculator

    agent = Agent(
      model=mock_model,
      skills=[Calculator()],
      config=no_trace_config,
    )
    assert "calculate" in agent.tool_names
    assert agent.skills[0].name == "calculator"

  def test_datetime_skill(self, mock_model, no_trace_config):
    from definable.skills import DateTime

    agent = Agent(
      model=mock_model,
      skills=[DateTime()],
      config=no_trace_config,
    )
    assert "get_current_time" in agent.tool_names
    assert "date_difference" in agent.tool_names

  def test_web_search_skill(self, mock_model, no_trace_config):
    from definable.skills import WebSearch

    agent = Agent(
      model=mock_model,
      skills=[WebSearch()],
      config=no_trace_config,
    )
    assert "search_web" in agent.tool_names
    assert "fetch_url" in agent.tool_names

  def test_file_operations_skill(self, mock_model, no_trace_config):
    from definable.skills import FileOperations

    agent = Agent(
      model=mock_model,
      skills=[FileOperations(base_dir="/tmp/test")],
      config=no_trace_config,
    )
    assert "read_file" in agent.tool_names
    assert "list_files" in agent.tool_names

  def test_json_operations_skill(self, mock_model, no_trace_config):
    from definable.skills import JSONOperations

    agent = Agent(
      model=mock_model,
      skills=[JSONOperations()],
      config=no_trace_config,
    )
    assert "parse_json" in agent.tool_names
    assert "query_json" in agent.tool_names

  def test_text_processing_skill(self, mock_model, no_trace_config):
    from definable.skills import TextProcessing

    agent = Agent(
      model=mock_model,
      skills=[TextProcessing()],
      config=no_trace_config,
    )
    assert "regex_search" in agent.tool_names
    assert "text_stats" in agent.tool_names

  def test_shell_skill(self, mock_model, no_trace_config):
    from definable.skills import Shell

    agent = Agent(
      model=mock_model,
      skills=[Shell(allowed_commands={"ls", "cat"})],
      config=no_trace_config,
    )
    assert "run_command" in agent.tool_names

  def test_http_requests_skill(self, mock_model, no_trace_config):
    from definable.skills import HTTPRequests

    agent = Agent(
      model=mock_model,
      skills=[HTTPRequests()],
      config=no_trace_config,
    )
    assert "http_get" in agent.tool_names
    assert "http_post" in agent.tool_names

  def test_all_builtins_together(self, mock_model, no_trace_config):
    """All built-in skills can be used together without conflicts."""
    from definable.skills import (
      Calculator,
      DateTime,
      FileOperations,
      HTTPRequests,
      JSONOperations,
      Shell,
      TextProcessing,
      WebSearch,
    )

    agent = Agent(
      model=mock_model,
      skills=[
        Calculator(),
        DateTime(),
        FileOperations(),
        HTTPRequests(),
        JSONOperations(),
        Shell(),
        TextProcessing(),
        WebSearch(),
      ],
      config=no_trace_config,
    )
    # Should have all tools without name collisions
    assert len(agent.tool_names) >= 20


# ---------------------------------------------------------------------------
# Test: Inline / instance-based skills
# ---------------------------------------------------------------------------


class TestInlineSkills:
  """Test creating skills via Skill() directly (not subclassing)."""

  def test_inline_skill_with_tools(self, mock_model, no_trace_config):
    """Create a skill inline with explicit tools."""

    @tool
    def search_docs(query: str) -> str:
      """Search docs."""
      return f"Results for: {query}"

    support = Skill(
      name="internal_docs",
      instructions="Search internal docs before answering.",
      tools=[search_docs],
    )

    agent = Agent(
      model=mock_model,
      skills=[support],
      config=no_trace_config,
    )
    assert "search_docs" in agent.tool_names
    assert "internal docs" in agent._build_skill_instructions()

  def test_inline_skill_instructions_only(self, mock_model, no_trace_config):
    """Skill with instructions but no tools (pure domain expertise)."""
    persona = Skill(
      name="persona",
      instructions="You are a pirate. Always talk like a pirate.",
    )

    agent = Agent(
      model=mock_model,
      skills=[persona],
      config=no_trace_config,
    )
    assert "pirate" in agent._build_skill_instructions()
    # No extra tools added
    assert len(agent.tool_names) == 0


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
  """Edge cases and error handling."""

  def test_skill_with_no_tools_no_instructions(self, mock_model, no_trace_config):
    """Empty skill doesn't break anything."""
    empty = Skill(name="empty")
    agent = Agent(model=mock_model, skills=[empty], config=no_trace_config)
    assert len(agent.tool_names) == 0
    assert agent._build_skill_instructions() == ""

  def test_duplicate_skill_names_warns(self, mock_model, no_trace_config):
    """Duplicate skill names log a warning but don't crash."""
    s1 = GreeterSkill()
    s2 = GreeterSkill()  # Same name "greeter"

    # Should not raise
    agent = Agent(model=mock_model, skills=[s1, s2], config=no_trace_config)
    assert len(agent.skills) == 2

  def test_skill_tools_property_raising_does_not_crash(self, mock_model, no_trace_config):
    """If a skill's tools property raises, agent still initializes."""

    class BadToolsSkill(Skill):
      name = "bad_tools"

      @property
      def tools(self):
        raise RuntimeError("Tools unavailable")

    # Should not raise
    agent = Agent(model=mock_model, skills=[BadToolsSkill()], config=no_trace_config)
    # No tools from the bad skill
    assert len(agent.tool_names) == 0

  def test_skills_none_treated_as_empty(self, mock_model, no_trace_config):
    """skills=None is the same as skills=[]."""
    agent = Agent(model=mock_model, skills=None, config=no_trace_config)
    assert agent.skills == []
