"""
Memory v2 — Tool-based memory with an AI agent.

The LLM manages its own memory via tool calls. Just pass
memory=Memory(...) to Agent — everything is auto-wired.

Run:
    export OPENAI_API_KEY=sk-...
    python definable/examples/memory/05_memory_v2.py
"""

import asyncio
import os
from pathlib import Path

from definable.agent import Agent
from definable.memory.v2 import Memory, SQLiteStore
from definable.model.openai import OpenAIChat

DB_PATH = "./example_memory_v2.db"
MODEL = os.environ.get("MEMORY_MODEL", "gpt-4o-mini")
USER_ID = "demo_user"


async def chat(agent: Agent, messages: list[str], prev=None):
  """Run a multi-turn chat session."""
  for msg in messages:
    print(f"  You: {msg}")
    kwargs = {"messages": prev} if prev else {}
    output = await agent.arun(msg, user_id=USER_ID, **kwargs)
    prev = output.messages
    print(f"  Aria: {output.content}\n")
  return prev


async def main():
  store = SQLiteStore(DB_PATH)
  memory = Memory(store=store)

  agent = Agent(
    model=OpenAIChat(id=MODEL),
    memory=memory,
    instructions="You are a helpful personal assistant named Aria. Be concise — 1-3 sentences max.",
  )

  try:
    # --- Session 1: Introduction ---
    print("=" * 50)
    print("SESSION 1: Getting to know you")
    print("=" * 50)
    await chat(
      agent,
      [
        "Hi! I'm Alex, a backend engineer at Shopify.",
        "I mostly write Go and Python. I prefer tabs over spaces — don't judge me.",
      ],
    )

    # --- Session 2: New session, test recall ---
    print("=" * 50)
    print("SESSION 2: Do you remember me?")
    print("=" * 50)
    await chat(
      agent,
      [
        "Hey, what do you know about me?",
        "What languages do I use?",
      ],
    )

    # --- Session 3: Correction ---
    print("=" * 50)
    print("SESSION 3: Things change")
    print("=" * 50)
    await chat(
      agent,
      [
        "Actually I switched teams — I'm on the payments team now, not the storefront team.",
        "And please forget the tabs thing. I've converted to spaces.",
      ],
    )

    # --- Show final memory state ---
    print("=" * 50)
    print("FINAL MEMORY STATE")
    print("=" * 50)

    wm = await store.get_working_memory(USER_ID)
    if wm:
      print(f"\nWorking Memory ({len(wm.content)} chars, v{wm.version}):")
      print(wm.content)

    entries = await store.search_index(USER_ID, limit=50)
    print(f"\nArchived Entries ({len(entries)}):")
    for e in entries:
      print(f"  [{e.id}] ({e.category}) {e.summary}")

    stats = await memory.get_stats(USER_ID)
    print(f"\nStats: {stats.entry_count} entries, {stats.total_content_chars} chars total")

  finally:
    await memory.close()
    for ext in ["", "-shm", "-wal"]:
      p = Path(DB_PATH + ext)
      if p.exists():
        p.unlink()
    print("\nCleaned up.")


if __name__ == "__main__":
  asyncio.run(main())
