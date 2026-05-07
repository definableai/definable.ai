"""Built-in skills for common agent capabilities."""

from typing import TYPE_CHECKING

from definable.agent.skill.builtin.calculator import Calculator
from definable.agent.skill.builtin.datetime_skill import DateTime
from definable.agent.skill.builtin.file_ops import FileOperations
from definable.agent.skill.builtin.http_requests import HTTPRequests
from definable.agent.skill.builtin.json_ops import JSONOperations
from definable.agent.skill.builtin.shell import Shell
from definable.agent.skill.builtin.text_processing import TextProcessing
from definable.agent.skill.builtin.web_search import WebSearch

if TYPE_CHECKING:
  from definable.agent.skill.builtin.csv_tools import CSVTools
  from definable.agent.skill.builtin.duckdb_analytics import DuckDBAnalytics
  from definable.agent.skill.builtin.email_tools import EmailTools
  from definable.agent.skill.builtin.firecrawl import Firecrawl
  from definable.agent.skill.builtin.github import GitHub
  from definable.agent.skill.builtin.python_exec import PythonExec
  from definable.agent.skill.builtin.slack_tools import SlackTools
  from definable.agent.skill.builtin.sql_database import SQLDatabase

__all__ = [
  # Core (always available)
  "Calculator",
  "DateTime",
  "FileOperations",
  "HTTPRequests",
  "JSONOperations",
  "Shell",
  "TextProcessing",
  "WebSearch",
  # Extended (lazy-loaded, require optional deps)
  "CSVTools",  # noqa: F822
  "DuckDBAnalytics",  # noqa: F822
  "EmailTools",  # noqa: F822
  "Firecrawl",  # noqa: F822
  "GitHub",  # noqa: F822
  "PythonExec",  # noqa: F822
  "SlackTools",  # noqa: F822
  "SQLDatabase",  # noqa: F822
]


def __getattr__(name: str):
  if name == "CSVTools":
    from definable.agent.skill.builtin.csv_tools import CSVTools

    return CSVTools
  if name == "DuckDBAnalytics":
    from definable.agent.skill.builtin.duckdb_analytics import DuckDBAnalytics

    return DuckDBAnalytics
  if name == "EmailTools":
    from definable.agent.skill.builtin.email_tools import EmailTools

    return EmailTools
  if name == "Firecrawl":
    from definable.agent.skill.builtin.firecrawl import Firecrawl

    return Firecrawl
  if name == "GitHub":
    from definable.agent.skill.builtin.github import GitHub

    return GitHub
  if name == "PythonExec":
    from definable.agent.skill.builtin.python_exec import PythonExec

    return PythonExec
  if name == "SlackTools":
    from definable.agent.skill.builtin.slack_tools import SlackTools

    return SlackTools
  if name == "SQLDatabase":
    from definable.agent.skill.builtin.sql_database import SQLDatabase

    return SQLDatabase
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
