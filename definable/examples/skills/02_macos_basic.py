"""MacOS skill — basic local agent example.

Demonstrates how to give a Definable agent full macOS control via the
Definable Desktop Bridge.

Prerequisites:
  1. Build and run the Desktop Bridge:
       cd definable/desktop-bridge
       swift build -c release
       .build/release/DesktopBridge
  2. Install dependencies:
       pip install definable[desktop]
  3. Set your API key:
       export OPENAI_API_KEY="sk-proj-..."

The bridge writes a token to ~/.definable/bridge-token automatically.
The MacOS skill reads it from there — no configuration needed.

Usage:
  python definable/examples/skills/02_macos_basic.py
"""

import asyncio

from definable.agent import Agent
from definable.model.moonshot import MoonshotChat
from definable.skill.builtin.macos import MacOS


async def main() -> None:
  model = MoonshotChat(id="kimi-k2-thinking")

  # --- 1. Basic MacOS agent ---
  print("=" * 60)
  print("Example 1: Basic MacOS control")
  print("=" * 60)

  agent = Agent(
    model=model,
    skills=[MacOS()],
    instructions=(
      "You are a macOS automation assistant. Take a screenshot first to understand the current state, then perform requested actions precisely."
    ),
  )

  output = await agent.arun("What's currently open on my screen?")
  print(f"Agent: {output.content}\n")

  # --- 2. Restricted agent (read-only) ---
  print("=" * 60)
  print("Example 2: Read-only agent (no input simulation)")
  print("=" * 60)

  read_only_agent = Agent(
    model=model,
    skills=[MacOS(enable_input=False, enable_file_write=False, enable_applescript=False)],
    instructions="You can only observe the screen — never interact with it.",
  )

  output = await read_only_agent.arun("Describe what's on the screen.")
  print(f"Agent: {output.content}\n")

  # --- 3. App-restricted agent ---
  print("=" * 60)
  print("Example 3: Allow only Safari and TextEdit")
  print("=" * 60)

  restricted_agent = Agent(
    model=model,
    skills=[
      MacOS(
        allowed_apps={"Safari", "TextEdit"},
        blocked_apps={"Terminal"},  # belt and suspenders
      )
    ],
    instructions="You may only interact with Safari and TextEdit.",
  )

  output = await restricted_agent.arun("Open Safari and navigate to apple.com")
  print(f"Agent: {output.content}\n")

  # --- 4. System information ---
  print("=" * 60)
  print("Example 4: Query system information")
  print("=" * 60)

  agent = Agent(
    model=model,
    skills=[MacOS(enable_input=False, enable_file_write=False)],
    instructions="You are a system monitor. Report facts concisely.",
  )

  output = await agent.arun("What is the current battery level and which apps are running?")
  print(f"Agent: {output.content}\n")


if __name__ == "__main__":
  asyncio.run(main())
