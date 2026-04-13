"""Basic ClaudeCodeAgent usage.

Requires: Claude Code CLI installed (`npm install -g @anthropic-ai/claude-code`)
and authenticated (`claude auth`).
"""

import asyncio

from definable.claude_code import ClaudeCodeAgent


async def main():
  agent = ClaudeCodeAgent(
    model="claude-sonnet-4-6",
    instructions="You are a senior Python developer. Write clean, tested code.",
    allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
    cwd=".",  # current directory
    max_turns=10,
  )

  result = await agent.arun("List all Python files in this directory and summarize what each one does.")

  print(f"Content: {result.content}")
  print(f"Session: {result.session_id}")
  print(f"Tokens: {result.metrics.input_tokens} in / {result.metrics.output_tokens} out")  # type: ignore[union-attr]
  if result.metrics.cost:  # type: ignore[union-attr]
    print(f"Cost: ${result.metrics.cost:.4f}")  # type: ignore[union-attr]
  if result.tools:
    print(f"Tools used: {[t.tool_name for t in result.tools]}")


if __name__ == "__main__":
  asyncio.run(main())
