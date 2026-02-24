# ruff: noqa: E501
"""Screen Reader Agent — describe what's on screen and answer questions.

A read-only agent that can see your screen via screenshots and OCR,
but cannot click, type, or modify anything. Useful for accessibility,
monitoring, or answering questions about what's currently displayed.

Requirements:
    ~/.definable/bin/desktop-bridge serve   (bridge must be running)
    export OPENAI_API_KEY=sk-...

Usage:
    .venv/bin/python definable/examples/desktop/04_screen_reader_agent.py
"""

import asyncio

from definable.agent import Agent
from definable.model.openai import OpenAIChat
from definable.skill.builtin.macos import MacOS


async def main() -> None:
  # Read-only: no input, no file writes, no shell, no AppleScript
  skill = MacOS(
    enable_input=False,
    enable_file_write=False,
    enable_applescript=False,
    enable_shell=False,
  )

  agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    skills=[skill],
    instructions=(
      "You are a screen reader assistant. You can take screenshots and "
      "read text from the screen using OCR, but you CANNOT interact — "
      "no clicking, no typing, no launching apps. Describe what you see "
      "clearly and concisely. If asked about specific content, use OCR "
      "to read the exact text."
    ),
  )

  # Multi-turn conversation about what's on screen
  print("Screen Reader Agent — ask about what's on your screen.")
  print("(Type 'quit' to exit)\n")

  messages = None
  while True:
    question = input("You: ").strip()
    if not question or question.lower() in ("quit", "exit", "q"):
      break

    result = await agent.arun(question, messages=messages)
    messages = result.messages
    print(f"\nAgent: {result.content}\n")


if __name__ == "__main__":
  asyncio.run(main())
