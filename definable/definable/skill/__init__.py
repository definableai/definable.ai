"""
Definable Skills - domain expertise bundled with tools.

Skills are a higher-level abstraction than Toolkits. While Toolkits only
group related tools, Skills combine **instructions** (domain expertise
the agent should follow) with **tools** (capabilities the agent can use)
and optional **dependencies** (shared config injected into tools).

Quick Start:
    from definable.agent import Agent
    from definable.model import OpenAIChat
    from definable.skill import Calculator, WebSearch, DateTime

    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        skills=[Calculator(), WebSearch(), DateTime()],
        instructions="You are a helpful assistant.",
    )

    output = agent.run("What's the weather like today?")

Custom Skills:
    from definable.skill import Skill
    from definable.tool import tool

    class CustomerSupport(Skill):
        name = "customer_support"
        instructions = "You are a customer support specialist."

        def __init__(self, db_url: str):
            super().__init__(dependencies={"db_url": db_url})

    agent = Agent(model=model, skills=[CustomerSupport(db_url="...")])

Inline Skills:
    from definable.skill import Skill
    from definable.tool import tool

    @tool
    def search_docs(query: str) -> str:
        '''Search internal docs.'''
        return docs_api.search(query)

    support = Skill(
        name="internal_docs",
        instructions="Search internal docs before answering questions.",
        tools=[search_docs],
    )

    agent = Agent(model=model, skills=[support])
"""

from definable.skill.base import Skill
from definable.skill.builtin.calculator import Calculator
from definable.skill.builtin.datetime_skill import DateTime
from definable.skill.builtin.file_ops import FileOperations
from definable.skill.builtin.http_requests import HTTPRequests
from definable.skill.builtin.json_ops import JSONOperations
from definable.skill.builtin.macos import MacOS
from definable.skill.builtin.shell import Shell
from definable.skill.builtin.text_processing import TextProcessing
from definable.skill.builtin.web_search import WebSearch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.skill.executor import SkillScriptExecutor
  from definable.skill.markdown import MarkdownSkill, MarkdownSkillMeta, SkillLoader, validate_agent_skills_name
  from definable.skill.registry import SkillRegistry

__all__ = [
  # Base class
  "Skill",
  # Built-in skills
  "Calculator",
  "DateTime",
  "FileOperations",
  "HTTPRequests",
  "JSONOperations",
  "MacOS",
  "Shell",
  "TextProcessing",
  "WebSearch",
  # Markdown skills
  "MarkdownSkill",
  "MarkdownSkillMeta",
  "SkillLoader",
  "SkillRegistry",
  # Agent Skills spec
  "SkillScriptExecutor",
  "validate_agent_skills_name",
]


# Lazy imports for markdown skills layer
def __getattr__(name: str):
  if name in ("MarkdownSkill", "MarkdownSkillMeta", "SkillLoader", "validate_agent_skills_name"):
    from definable.skill.markdown import MarkdownSkill, MarkdownSkillMeta, SkillLoader, validate_agent_skills_name

    globals()["MarkdownSkill"] = MarkdownSkill
    globals()["MarkdownSkillMeta"] = MarkdownSkillMeta
    globals()["SkillLoader"] = SkillLoader
    globals()["validate_agent_skills_name"] = validate_agent_skills_name
    return globals()[name]
  if name == "SkillRegistry":
    from definable.skill.registry import SkillRegistry

    globals()["SkillRegistry"] = SkillRegistry
    return SkillRegistry
  if name == "SkillScriptExecutor":
    from definable.skill.executor import SkillScriptExecutor

    globals()["SkillScriptExecutor"] = SkillScriptExecutor
    return SkillScriptExecutor
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
