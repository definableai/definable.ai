"""Skill base class — bundles domain expertise with tools.

A Skill is a higher-level abstraction than a Toolkit. While Toolkits only
group related tools, Skills combine **instructions** (domain expertise
the agent should follow) with **tools** (capabilities the agent can use)
and optional **dependencies** (shared config injected into tools).

Two usage patterns
------------------

**Instance-based** (quick, inline):

    from definable.skill import Skill
    from definable.tool import tool

    @tool
    def search_docs(query: str) -> str:
        '''Search internal docs.'''
        return docs_api.search(query)

    support = Skill(
        name="customer_support",
        instructions="You are a customer support specialist. Always greet warmly...",
        tools=[search_docs],
    )

    agent = Agent(model=model, skills=[support])

**Class-based** (reusable, configurable):

    from definable.skill import Skill
    from definable.tool import tool

    class CustomerSupport(Skill):
        name = "customer_support"
        instructions = '''
        You are a customer support specialist.
        Always greet the customer warmly, identify their issue,
        provide a solution, and ask if they need anything else.
        '''

        def __init__(self, db_url: str):
            super().__init__(dependencies={"db_url": db_url})

        @tool
        def lookup_order(self, order_id: str) -> str:
            '''Look up an order by its ID.'''
            return database.query(order_id)

    agent = Agent(model=model, skills=[CustomerSupport(db_url="...")])
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
  from definable.tool.function import Function


class Skill:
  """
  Base class for agent skills — domain expertise bundled with tools.

  A Skill provides:
    - **instructions**: Domain expertise injected into the agent's system
      prompt. This is the "knowledge" the agent gains from the skill.
    - **tools**: Capabilities the agent can call. Discovered automatically
      from ``@tool``-decorated methods or passed explicitly.
    - **dependencies**: Shared configuration injected into all skill tools.

  When an agent has skills, it merges each skill's instructions into its
  system prompt and each skill's tools into its available tool set.

  Lifecycle hooks (optional override in subclasses):
    - ``setup()``: Called once when the skill is attached to an agent.
      Use for one-time initialization (DB connections, API clients, etc.).
    - ``teardown()``: Called when the agent shuts down. Use for cleanup.

  Args:
    name: Skill identifier. Defaults to the class name.
    instructions: Domain expertise for the agent. Merged into the system
        prompt when the skill is attached to an agent.
    tools: Explicit list of tools. If not provided, tools are discovered
        automatically from ``@tool``-decorated methods on the class.
    dependencies: Shared dependencies injected into all skill tools,
        merged with agent-level dependencies at runtime.

  Example:
    Using with an agent::

      from definable.agent import Agent
      from definable.skill import Skill, WebSearch, Calculator

      agent = Agent(
          model=model,
          skills=[WebSearch(api_key="..."), Calculator()],
          instructions="You are a helpful assistant.",
      )
      output = agent.run("What is 15% of $249.99?")
  """

  # Override in subclasses for class-based skills
  name: str = ""
  instructions: str = ""

  def __init__(
    self,
    *,
    name: Optional[str] = None,
    instructions: Optional[str] = None,
    tools: Optional[List["Function"]] = None,
    dependencies: Optional[Dict[str, Any]] = None,
  ):
    """
    Initialize the skill.

    Args:
      name: Override the skill name (defaults to class attribute or class name).
      instructions: Override the skill instructions (defaults to class attribute).
      tools: Explicit tool list. If None, tools are auto-discovered from methods.
      dependencies: Shared dependencies to inject into all skill tools.
    """
    if name is not None:
      self.name = name
    elif not self.name:
      self.name = self.__class__.__name__

    if instructions is not None:
      self.instructions = instructions

    self._explicit_tools = tools
    self._dependencies = dependencies or {}
    self._initialized = False

  @property
  def tools(self) -> List["Function"]:
    """
    Return the tools provided by this skill.

    If tools were passed explicitly to ``__init__``, those are returned.
    Otherwise, auto-discovers ``Function``-typed attributes on the instance
    (e.g. methods decorated with ``@tool``).

    Override this property in subclasses for dynamic tool generation.

    Returns:
      List of Function objects.
    """
    if self._explicit_tools is not None:
      return list(self._explicit_tools)

    from definable.tool.function import Function

    discovered: List[Function] = []
    for attr_name in dir(self):
      if attr_name.startswith("_"):
        continue
      try:
        attr = getattr(self, attr_name)
        if isinstance(attr, Function):
          discovered.append(attr)
      except Exception:
        continue
    return discovered

  @property
  def dependencies(self) -> Dict[str, Any]:
    """
    Get the shared dependencies for this skill.

    Returns:
      Dictionary of dependencies injected into all skill tools.
    """
    return self._dependencies

  def get_instructions(self) -> str:
    """
    Get the skill's instructions for system prompt injection.

    Override in subclasses for dynamic instruction generation
    (e.g. instructions that depend on runtime configuration).

    Returns:
      Instructions string. Empty string means no instructions to inject.
    """
    return self.instructions

  def setup(self) -> None:
    """Called once when the skill is attached to an agent.

    Override in subclasses for one-time initialization: opening database
    connections, creating API clients, loading configuration, etc.

    This is called during ``Agent.__init__`` (or the first ``agent.run()``
    if lazy initialization is needed). Errors here will propagate.

    Example::

      class DatabaseSkill(Skill):
          def setup(self):
              self._conn = create_connection(self.dependencies["db_url"])
    """
    pass

  def teardown(self) -> None:
    """Called when the agent shuts down (context manager exit).

    Override in subclasses for cleanup: closing connections, flushing
    buffers, releasing resources, etc.

    Example::

      class DatabaseSkill(Skill):
          def teardown(self):
              if hasattr(self, "_conn"):
                  self._conn.close()
    """
    pass

  def __repr__(self) -> str:
    tool_count = len(self.tools)
    has_instructions = bool(self.get_instructions())
    return f"{self.name}(tools={tool_count}, instructions={has_instructions})"
