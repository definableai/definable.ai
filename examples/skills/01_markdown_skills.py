"""Markdown Skills — teaching agents methodology via .md files.

This example demonstrates:
1. Loading the built-in skill library
2. Listing and searching skills
3. Using skills in eager mode (all injected into system prompt)
4. Using skills in lazy mode (catalog + read_skill tool)
5. Using skill_registry= on Agent for automatic mode selection

No API keys required — uses MockModel for demonstration.
"""

from definable.agent import Agent
from definable.agent.tracing import Tracing
from definable.agent.testing import MockModel
from definable.skill import SkillRegistry


def main():
  # Create a mock model (no API keys needed)
  model = MockModel(responses=["I'll follow the methodology to help you."])
  tracing = Tracing(enabled=False)

  # --- 1. Load the built-in skill library ---
  print("=" * 60)
  print("1. Loading built-in skill library")
  print("=" * 60)

  registry = SkillRegistry()
  print(f"Loaded {len(registry)} skills:\n")

  for meta in registry.list_skills():
    tags = ", ".join(meta.tags)
    print(f"  {meta.name:25s}  {meta.description}")
    print(f"  {'':25s}  tags: {tags}\n")

  # --- 2. Search skills ---
  print("=" * 60)
  print("2. Searching for 'code' skills")
  print("=" * 60)

  results = registry.search_skills("code")
  for skill in results:
    print(f"  {skill.meta.name}: {skill.meta.description}")
  print()

  # --- 3. Eager mode — all skills in system prompt ---
  print("=" * 60)
  print("3. Eager mode (all skills injected)")
  print("=" * 60)

  agent = Agent(
    model=model,  # type: ignore[arg-type]
    skills=registry.as_eager(),
    instructions="You are a helpful assistant.",
    tracing=tracing,
  )
  print(f"Agent has {len(agent.skills)} skills")
  print(f"Skill instructions length: {len(agent._build_skill_instructions())} chars")
  output = agent.run("Review this code for bugs.")
  print(f"Response: {output.content}\n")

  # --- 4. Lazy mode — catalog + read_skill tool ---
  print("=" * 60)
  print("4. Lazy mode (catalog + read_skill tool)")
  print("=" * 60)

  lazy_skill = registry.as_lazy()
  agent = Agent(
    model=model,  # type: ignore[arg-type]
    skills=[lazy_skill],
    instructions="You are a helpful assistant.",
    tracing=tracing,
  )
  print(f"Agent has {len(agent.skills)} skill wrapper(s)")
  print(f"Tools: {agent.tool_names}")

  # Show the catalog preview
  catalog = lazy_skill.get_instructions()
  preview = catalog[:300] + "..." if len(catalog) > 300 else catalog
  print(f"Catalog preview:\n{preview}\n")

  # --- 5. skill_registry= auto mode ---
  print("=" * 60)
  print("5. skill_registry= parameter (auto mode)")
  print("=" * 60)

  agent = Agent(
    model=model,  # type: ignore[arg-type]
    skill_registry=registry,
    instructions="You are a helpful assistant.",
    tracing=tracing,
  )
  print(f"Agent has {len(agent.skills)} skills (auto-selected eager mode)")
  output = agent.run("Help me debug this error.")
  print(f"Response: {output.content}\n")


if __name__ == "__main__":
  main()
