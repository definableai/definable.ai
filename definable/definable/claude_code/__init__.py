"""Claude Code integration — wrap the Claude CLI with Definable features.

Usage::

    from definable.claude_code import ClaudeCodeAgent, run_cli

    agent = ClaudeCodeAgent(
        model="claude-sonnet-4-6",
        instructions="Senior backend developer.",
        cwd="/workspace/my-app",
    )

    # Non-interactive
    result = await agent.arun("Fix the auth bug")

    # Interactive CLI
    run_cli(agent)
"""

from definable.claude_code.agent import ClaudeCodeAgent
from definable.claude_code.cli import ClaudeCodeCLI, run_cli

__all__ = ["ClaudeCodeAgent", "ClaudeCodeCLI", "run_cli"]
