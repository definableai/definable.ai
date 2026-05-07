"""06 — skill loading.

Skills are filesystem directories: `SKILL.md` (frontmatter +
instructions) plus optional `tools.py` (decorated tools) plus optional
`scripts/` (executables). Pass loaded skills via `skills=[...]`.
"""

from __future__ import annotations

import asyncio

from definable import Agent
from definable.agent.skill import load_skills


async def main() -> None:
  # Point at any directory containing SKILL.md folders.
  # The framework's bundled built-in skills live at:
  #   definable/definable/agent/skill/builtin/
  skills = load_skills("definable/definable/agent/skill/builtin")

  agent = Agent(
    name="skilled",
    model="anthropic/claude-sonnet-4-6",
    instructions="Use the loaded skills when relevant.",
    skills=skills[:2],  # narrow for the demo
  )
  async with agent:
    print((await agent.arun("What can you do?")).content)


if __name__ == "__main__":
  asyncio.run(main())
