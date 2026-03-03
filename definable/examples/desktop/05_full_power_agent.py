# ruff: noqa: E501
"""Full Power Desktop Agent — all capabilities enabled.

An unrestricted agent with shell, camera, screen recording, file writes,
and AppleScript. Use for trusted automation tasks like setting up a dev
environment, running builds, or orchestrating complex workflows.

Requirements:
    ~/.definable/bin/desktop-bridge serve   (bridge must be running)
    export OPENAI_API_KEY=sk-...

Usage:
    .venv/bin/python definable/examples/desktop/05_full_power_agent.py
"""

import asyncio

from definable.agent import Agent
from definable.agent.interface.cli import CLIInterface
from definable.model.openai import OpenAIChat
from definable.skill.builtin.macos import MacOS


async def main() -> None:
  # All gates open — 35 tools
  skill = MacOS(
    enable_input=True,
    enable_file_write=True,
    enable_applescript=True,
    enable_shell=True,
  )

  agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    skills=[skill],
    observability=True,
    instructions=(
      "You are a full-power macOS automation agent with shell access, "
      "camera, screen recording, file I/O, and complete input control. "
      "You can run terminal commands, take photos, record the screen, "
      "read/write files, and control any app.\n\n"
      "Workflow:\n"
      "1. Screenshot to see the current state\n"
      "2. Plan your actions\n"
      "3. Execute step by step\n"
      "4. Screenshot to verify each step\n\n"
      "Be careful with destructive operations. Prefer non-destructive "
      "approaches when possible."
      "You should not give up so easily. If something does not work out, try some different approach to fulfil the user's query."
    ),
  )
  # result = await agent.arun("Can you read all the files in my download folder and tell me what they are?")
  # print(result.content)
  await agent.aserve(CLIInterface(mode="repl"), enable_server=True, port=8003)


if __name__ == "__main__":
  asyncio.run(main())
