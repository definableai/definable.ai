"""
Agent with Cortex Memory — next-gen memory that learns about you over time.

Cortex goes beyond simple session history:
  - Multi-representation ingestion (fast path: regex entities, signatures, embeddings)
  - 5-layer retrieval cascade (scratchpad → query analysis → signature pre-filter → targeted search → fusion)
  - Scratchpad for fast key-value beliefs
  - Behavioral learning (trait observation + inference)
  - Graph + tag indexes for relational and hierarchical retrieval

This example demonstrates:
  1. Using CortexMemory as a drop-in replacement for Memory in Agent
  2. Multi-turn conversation with memory recall
  3. The native Cortex API (remember, recall, set_belief, as_tools)
  4. Cortex tools exposed to the agent for active memory management

Requirements:
    export OPENAI_API_KEY=sk-...
    pip install aiosqlite
"""

import asyncio
import os

from definable.agent import Agent
from definable.embedder import OpenAIEmbedder
from definable.memory.cortex import CortexConfig, CortexMemory
from definable.model.openai import OpenAIChat


async def demo_agent_integration():
  """Demo 1: CortexMemory as a drop-in replacement for Memory."""
  print("=" * 60)
  print("DEMO 1: CortexMemory as Agent Memory (drop-in)")
  print("=" * 60)

  db_path = "./cortex_demo.db"
  if os.path.exists(db_path):
    os.remove(db_path)

  cortex = CortexMemory(
    config=CortexConfig(
      db_path=db_path,
      slow_path_enabled=False,  # Fast path only — no LLM during ingestion
      enable_learning=True,  # Observe behavioral traits
      enable_consolidation=False,  # No background merge for demo
    ),
    embedder=OpenAIEmbedder(),  # Enables semantic search
  )

  model = OpenAIChat(id="gpt-4o-mini")
  agent = Agent(
    model=model,
    instructions=(
      "You are a helpful assistant with persistent memory. "
      "Use the memory context provided to personalize your responses. "
      "If the user tells you facts about themselves, acknowledge and remember them."
    ),
    memory=cortex,  # Drop-in — same interface as Memory(store=...)
  )

  # Turn 1: Teach it some facts
  print("\n--- Turn 1: Introducing facts ---")
  r1 = await agent.arun("My name is Alice. I'm a ML engineer at Acme Corp. I prefer Python over Java and I'm currently working on a RAG pipeline.")
  print(f"Agent: {r1.content}\n")

  # Turn 2: Ask it to recall
  print("--- Turn 2: Testing recall ---")
  r2 = await agent.arun(
    "What programming language do I prefer?",
    messages=r1.messages,
  )
  print(f"Agent: {r2.content}\n")

  # Turn 3: Add more context, test cross-turn retrieval
  print("--- Turn 3: Adding more facts ---")
  r3 = await agent.arun(
    "I also love hiking in the mountains and my favorite food is sushi.",
    messages=r2.messages,
  )
  print(f"Agent: {r3.content}\n")

  # Turn 4: Cross-domain recall
  print("--- Turn 4: Cross-domain recall ---")
  r4 = await agent.arun(
    "What do you know about me? Summarize everything.",
    messages=r3.messages,
  )
  print(f"Agent: {r4.content}\n")

  # Cleanup
  await cortex.close()
  if os.path.exists(db_path):
    os.remove(db_path)


async def demo_native_api():
  """Demo 2: Using the Cortex-native API directly."""
  print("\n" + "=" * 60)
  print("DEMO 2: Cortex Native API")
  print("=" * 60)

  db_path = "./cortex_native_demo.db"
  if os.path.exists(db_path):
    os.remove(db_path)

  async with CortexMemory(
    config=CortexConfig(
      db_path=db_path,
      slow_path_enabled=False,
      enable_learning=False,
      enable_consolidation=False,
    ),
    embedder=OpenAIEmbedder(),
  ) as cortex:
    # Remember some facts
    print("\n--- Storing memories ---")
    r1 = await cortex.remember("Alice works at Acme Corp as an ML engineer")
    r2 = await cortex.remember("Alice's favorite programming language is Python")
    r3 = await cortex.remember("Alice is building a RAG pipeline for document search")
    r4 = await cortex.remember("Bob works at Beta Inc as a frontend developer")
    r5 = await cortex.remember("Bob prefers TypeScript and React")
    print(f"Stored 5 memories: {r1[:8]}..., {r2[:8]}..., {r3[:8]}..., {r4[:8]}..., {r5[:8]}...")

    # Recall by semantic query
    print("\n--- Semantic recall: 'What does Alice do?' ---")
    result = await cortex.recall("What does Alice do?", top_k=3)
    for mem in result.memories:
      print(f"  [{mem.score:.3f}] [{mem.source_layer}] {mem.record.raw_content}")

    print(f"\n  Total candidates searched: {result.total_candidates}")
    if result.scratchpad_context:
      print(f"  Scratchpad context: {result.scratchpad_context}")

    # Recall for Bob
    print("\n--- Semantic recall: 'What language does Bob prefer?' ---")
    result2 = await cortex.recall("What language does Bob prefer?", top_k=3)
    for mem in result2.memories:
      print(f"  [{mem.score:.3f}] [{mem.source_layer}] {mem.record.raw_content}")

    # Scratchpad beliefs
    print("\n--- Scratchpad beliefs ---")
    await cortex.set_belief("alice_role", "ML Engineer")
    await cortex.set_belief("project_focus", "RAG pipeline")
    state = await cortex.get_state()
    print(f"  alice_role = {state.get_belief('alice_role')}")
    print(f"  project_focus = {state.get_belief('project_focus')}")

    # Update a memory
    print("\n--- Updating a memory ---")
    updated = await cortex.update(r2, "Alice's favorite programming language is now Rust")
    if updated:
      print(f"  Updated memory {r2[:8]}... → new record {updated.record_id[:8]}...")

    # Verify update
    print("\n--- Recall after update: 'favorite language' ---")
    result3 = await cortex.recall("favorite programming language", top_k=3)
    for mem in result3.memories:
      print(f"  [{mem.score:.3f}] {mem.record.raw_content}")

    # Forget a memory
    print("\n--- Forgetting Bob's work info ---")
    forgotten = await cortex.forget(r4, reason="No longer relevant")
    print(f"  Forgotten: {forgotten}")

  # Cleanup
  if os.path.exists(db_path):
    os.remove(db_path)


async def demo_cortex_tools():
  """Demo 3: Expose Cortex as tools so the agent can actively manage memory."""
  print("\n" + "=" * 60)
  print("DEMO 3: Cortex Tools (agent-managed memory)")
  print("=" * 60)

  db_path = "./cortex_tools_demo.db"
  if os.path.exists(db_path):
    os.remove(db_path)

  cortex = CortexMemory(
    config=CortexConfig(
      db_path=db_path,
      slow_path_enabled=False,
      enable_learning=False,
      enable_consolidation=False,
    ),
    embedder=OpenAIEmbedder(),
  )

  # Pre-load some memories
  await cortex._ensure_initialized()
  await cortex.remember("The quarterly report deadline is March 15th")
  await cortex.remember("Budget for Q1 marketing campaign is $50,000")
  await cortex.remember("Team meeting every Tuesday at 2pm")

  model = OpenAIChat(id="gpt-4o-mini")
  agent = Agent(
    model=model,
    instructions=(
      "You are a smart assistant with active memory management. "
      "You can store, recall, and forget information using your cortex tools. "
      "When the user asks you to remember something, use cortex_remember. "
      "When they ask what you know, use cortex_recall to search your memory."
    ),
    tools=cortex.as_tools(),  # Exposes cortex_remember, cortex_recall, cortex_set_belief, cortex_forget
  )

  # The agent can now actively use memory
  print("\n--- Agent using cortex_recall tool ---")
  r1 = await agent.arun("What do you know about upcoming deadlines?")
  print(f"Agent: {r1.content}\n")

  print("--- Agent using cortex_remember tool ---")
  r2 = await agent.arun(
    "Please remember that the design review is on March 20th",
    messages=r1.messages,
  )
  print(f"Agent: {r2.content}\n")

  print("--- Verifying the stored memory ---")
  r3 = await agent.arun(
    "What events do I have in March?",
    messages=r2.messages,
  )
  print(f"Agent: {r3.content}\n")

  # Cleanup
  await cortex.close()
  if os.path.exists(db_path):
    os.remove(db_path)


async def main():
  await demo_agent_integration()
  await demo_native_api()
  await demo_cortex_tools()
  print("\nAll demos complete!")


if __name__ == "__main__":
  asyncio.run(main())
