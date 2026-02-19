"""Claude Code integration — wrap the Claude CLI with Definable features.

Usage::

    from definable.claude_code import ClaudeCodeAgent

    agent = ClaudeCodeAgent(
        model="claude-sonnet-4-6",
        instructions="Senior backend developer.",
        cwd="/workspace/my-app",
    )
    result = await agent.arun("Fix the auth bug")
"""

from definable.claude_code.agent import ClaudeCodeAgent

__all__ = ["ClaudeCodeAgent"]
