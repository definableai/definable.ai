# ruff: noqa: E501
"""Desktop Agent — control your Mac with natural language.

Give the agent a task and it uses screenshots, OCR, mouse, keyboard,
app management, and AppleScript to carry it out on your actual Mac.

Requirements:
    ~/.definable/bin/desktop-bridge serve   (bridge must be running)
    export OPENAI_API_KEY=sk-...

Usage:
    .venv/bin/python definable/examples/desktop/02_desktop_agent.py
"""

import asyncio

from definable.agent import Agent
from definable.model.openai import OpenAIChat
from definable.skill.builtin.macos import MacOS


async def main() -> None:
  # The MacOS skill gives the agent ~34 tools for controlling macOS.
  # Security gates: allowed_apps limits which apps can be targeted,
  # enable_input/enable_file_write/enable_shell control capabilities.
  skill = MacOS(
    allowed_apps={"Safari", "TextEdit", "Notes", "Terminal", "Finder"},
    enable_input=True,
    enable_file_write=False,  # read-only filesystem
    enable_shell=False,  # no shell commands
  )

  agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    skills=[skill],
    instructions=(
      "You are a macOS desktop assistant. You can see the screen, "
      "click, type, and manage apps. Always take a screenshot first "
      "to understand what's on screen before acting. After every "
      "action, take another screenshot to verify it worked."
    ),
  )

  # Example: open Safari and get top HN posts
  result = await agent.arun(
    "Open Safari, go to news.ycombinator.com, and tell me the top 5 posts. Use AppleScript to navigate (set URL of document 1)."
  )
  print(result.content)


if __name__ == "__main__":
  asyncio.run(main())
