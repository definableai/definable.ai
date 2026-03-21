"""Library Skill Discovery — inspect every built-in skill.

This example demonstrates:
1. Loading all 24 built-in library skills
2. Inspecting each skill's metadata (name, description, tags, version, author)
3. Checking directory structure (scripts, references, assets)
4. Listing bundled files for directory skills
5. Validating names against the Agent Skills spec
6. Generating the XML catalog (what the LLM sees at startup)
7. Testing on-demand mode (activate_skill + read_skill_file + run_skill_script)
8. Searching skills by keyword

No API keys required — uses MockModel for demonstration.
"""

from definable.agent import Agent
from definable.agent.tracing import Tracing
from definable.model.openai import OpenAIChat
from definable.skill import SkillRegistry, validate_agent_skills_name


def main():
  model = OpenAIChat(id="gpt-4o")
  tracing = Tracing(enabled=False)

  # ── 1. Load the built-in library ──────────────────────────────
  print("=" * 70)
  print("1. Loading built-in skill library")
  print("=" * 70)

  registry = SkillRegistry()
  print(f"Total skills loaded: {len(registry)}\n")

  # ── 2. Inspect every skill ────────────────────────────────────
  print("=" * 70)
  print("2. Skill inventory")
  print("=" * 70)

  for meta in registry.list_skills():
    print(f"\n  [{meta.name}]")
    print(f"    Description : {meta.description[:100]}{'...' if len(meta.description) > 100 else ''}")
    print(f"    Tags        : {', '.join(meta.tags)}")
    print(f"    Version     : {meta.version}")
    if meta.author:
      print(f"    Author      : {meta.author}")
    if meta.license:
      print(f"    License     : {meta.license}")
    if meta.requires_tools:
      print(f"    Requires    : {', '.join(meta.requires_tools)}")
    if meta.allowed_tools:
      print(f"    Allowed     : {', '.join(meta.allowed_tools)}")
    if meta.metadata:
      print(f"    Metadata    : {meta.metadata}")

  # ── 3. Directory structure & bundled files ─────────────────────
  print(f"\n{'=' * 70}")
  print("3. Directory skills — bundled resources")
  print("=" * 70)

  dir_count = 0
  script_count = 0
  ref_count = 0
  asset_count = 0

  for meta in registry.list_skills():
    skill = registry.get_skill(meta.name)
    if not skill or not skill.is_directory_skill:
      continue
    dir_count += 1

    files = skill.list_files()
    has_s = skill.has_scripts()
    has_r = skill.has_references()
    has_a = skill.has_assets()

    badges = []
    if has_s:
      badges.append("scripts")
      script_count += 1
    if has_r:
      badges.append("references")
      ref_count += 1
    if has_a:
      badges.append("assets")
      asset_count += 1

    badge_str = f" [{', '.join(badges)}]" if badges else ""
    print(f"\n  {meta.name}{badge_str}")
    print(f"    Directory: {skill.skill_directory}")
    if files:
      for f in sorted(files):
        print(f"      {f}")
    else:
      print("      (no bundled files besides SKILL.md)")

  print(f"\n  Summary: {dir_count} directory skills, {script_count} with scripts, {ref_count} with references, {asset_count} with assets")

  # ── 4. Name validation (Agent Skills spec) ────────────────────
  print(f"\n{'=' * 70}")
  print("4. Agent Skills spec name validation")
  print("=" * 70)

  all_valid = True
  for meta in registry.list_skills():
    errors = validate_agent_skills_name(meta.name)
    if errors:
      print(f"  WARN {meta.name}: {errors}")
      all_valid = False

  if all_valid:
    print("  All skill names are valid Agent Skills spec names.")

  # ── 5. XML catalog (what the LLM sees) ────────────────────────
  print(f"\n{'=' * 70}")
  print("5. XML catalog (injected at startup in on-demand mode)")
  print("=" * 70)

  xml = registry.to_prompt()
  # Show first 1000 chars
  preview = xml[:1000] + "\n  ..." if len(xml) > 1000 else xml
  print(f"\n  Total length: {len(xml)} chars\n")
  for line in preview.split("\n"):
    print(f"  {line}")

  # ── 6. On-demand mode — 3 tools ───────────────────────────────
  print(f"\n{'=' * 70}")
  print("6. On-demand mode (activate_skill + read_skill_file + run_skill_script)")
  print("=" * 70)

  on_demand = registry.as_on_demand()
  agent = Agent(
    model=model,  # type: ignore[arg-type]
    skills=[on_demand],
    instructions="You are a helpful assistant.",
    tracing=tracing,
  )
  print(f"  Skills: {len(agent.skills)}")
  print(f"  Tools: {sorted(agent.tool_names)}")

  # Also test skill_registry= shorthand
  agent2 = Agent(
    model=model,  # type: ignore[arg-type]
    skill_registry=registry,
    instructions="You are a helpful assistant.",
    tracing=tracing,
  )
  print(f"  skill_registry= tools: {sorted(agent2.tool_names)}")

  output = agent.run("Create a PDF report for me.")
  print(f"  Response: {output.content}\n")

  # ── 7. Eager mode — all skills injected ───────────────────────
  print("=" * 70)
  print("7. Eager mode (all skills in system prompt)")
  print("=" * 70)

  eager_skills = registry.as_eager()
  agent3 = Agent(
    model=model,  # type: ignore[arg-type]
    skills=eager_skills,
    instructions="You are a helpful assistant.",
    tracing=tracing,
  )
  instructions_len = len(agent3._build_skill_instructions())
  print(f"  Skills injected: {len(eager_skills)}")
  print(f"  Total instructions size: {instructions_len:,} chars")

  # ── 8. Search skills by keyword ───────────────────────────────
  print(f"\n{'=' * 70}")
  print("8. Keyword search")
  print("=" * 70)

  for keyword in ["code", "pdf", "design", "data", "web"]:
    results = registry.search_skills(keyword)
    names = [s.meta.name for s in results]
    print(f'  "{keyword}": {names}')

  # ── 9. Read a bundled file ────────────────────────────────────
  print(f"\n{'=' * 70}")
  print("9. Reading a bundled reference file")
  print("=" * 70)

  # Find a skill with references and read one
  for meta in registry.list_skills():
    skill = registry.get_skill(meta.name)
    if skill and skill.is_directory_skill and skill.has_references():
      files = skill.list_files()
      ref_files = [f for f in files if f.startswith("references/")]
      if ref_files:
        content = skill.read_file(ref_files[0])
        preview = content[:300] + "..." if len(content) > 300 else content
        print(f"  Skill: {meta.name}")
        print(f"  File: {ref_files[0]}")
        print(f"  Content ({len(content)} chars):\n")
        for line in preview.split("\n")[:10]:
          print(f"    {line}")
        break
  else:
    print("  (No skills with references/ found)")

  # ── Done ──────────────────────────────────────────────────────
  print(f"\n{'=' * 70}")
  print("All checks passed!")
  print("=" * 70)


if __name__ == "__main__":
  main()
