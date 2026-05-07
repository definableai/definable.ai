"""03 — file-based memory.

`memory=True` mounts a FileMemory at `.definable/memory/{agent_name}/`
and auto-injects four tools: read_memory, write_memory, list_memories,
search_memory. The agent calls them when it decides it needs to.
"""

from __future__ import annotations

import asyncio

from definable import Agent


async def main() -> None:
  agent = Agent(
    name="rememberer",
    model="openai/gpt-5.4-mini",
    instructions=(
      "You can save and recall facts via the read_memory / write_memory / "
      "list_memories / search_memory tools. When the user shares info worth "
      "keeping, write it to a descriptively-named file."
    ),
    memory=True,
  )
  async with agent:
    print("→", (await agent.arun("My name is Anandesh and I prefer concise output. Save that.")).content)
    print("→", (await agent.arun("What do you know about me?")).content)


if __name__ == "__main__":
  asyncio.run(main())
