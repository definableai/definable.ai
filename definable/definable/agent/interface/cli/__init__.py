"""CLI interface — interactive terminal REPL for Definable agents.

Usage::

    from definable.agent import Agent
    from definable.agent.interface.cli import CLIInterface

    agent = Agent(model="openai/gpt-4o-mini", tools=[my_tool])
    agent.serve(CLIInterface())

    # With custom settings:
    agent.serve(CLIInterface(show_thinking=False, markdown_output=False))
"""

from definable.agent.interface.cli.config import CLIConfig
from definable.agent.interface.cli.interface import CLIInterface

__all__ = [
  "CLIConfig",
  "CLIInterface",
]
