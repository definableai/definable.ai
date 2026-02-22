"""Coding Agent with Auto-Skill Selection — lazy mode in action.

This example demonstrates:
1. Creating domain-specific MarkdownSkills programmatically (no .md files)
2. Registering them in a SkillRegistry (without built-in library)
3. Using lazy mode so the agent auto-selects the right skill per task
4. Running two tasks: one back-end, one front-end — watch the agent pick

Requires: OPENAI_API_KEY in environment.
"""

import asyncio

from definable.agent import Agent
from definable.skill import MarkdownSkill, MarkdownSkillMeta, SkillRegistry

# --- 1. Define two coding skills as MarkdownSkills ---

backend_skill = MarkdownSkill(
  meta=MarkdownSkillMeta(
    name="backend-engineering",
    description="Back-end engineering: APIs, databases, system design, security",
    tags=["backend", "api", "database", "python", "security"],
  ),
  content="""\
## Role
You are a senior back-end engineer. You design robust, scalable server-side systems.

## Principles
- Design APIs that are RESTful, versioned, and well-documented
- Use connection pooling and query optimization for database access
- Apply input validation at every boundary (never trust the caller)
- Prefer idempotent operations; design for retry safety
- Handle errors explicitly — no silent swallowing, no bare excepts
- Log structured data (JSON) with correlation IDs for traceability

## Response Format
When asked to write code or design a system:
1. State assumptions and constraints up front
2. Provide the implementation with inline comments on non-obvious decisions
3. Call out security considerations (auth, injection, rate limiting)
4. Suggest one concrete improvement the user could make next
""",
)

frontend_skill = MarkdownSkill(
  meta=MarkdownSkillMeta(
    name="frontend-engineering",
    description="Front-end engineering: UI components, state management, accessibility",
    tags=["frontend", "react", "css", "accessibility", "ux"],
  ),
  content="""\
## Role
You are a senior front-end engineer. You build accessible, performant user interfaces.

## Principles
- Components should be small, composable, and testable in isolation
- State lives as close to where it is used as possible (avoid prop drilling)
- Semantic HTML first — use ARIA attributes only when native semantics fall short
- CSS: prefer design tokens and utility classes over deeply nested selectors
- Performance: lazy-load routes, defer non-critical JS, optimize images
- Always consider keyboard navigation and screen-reader experience

## Response Format
When asked to build a UI or component:
1. Clarify the user interaction (what triggers what)
2. Provide the component code with accessibility attributes
3. Note any state management decisions (local vs shared)
4. Suggest one UX improvement the user could make next
""",
)


def print_tool_calls(result):
  """Extract and print tool calls from the conversation to prove skill selection."""
  if not result.messages:
    return
  for msg in result.messages:
    if msg.tool_calls:
      for tc in msg.tool_calls:
        fn = tc.get("function", {})
        print(f"  -> Tool call: {fn.get('name')}({fn.get('arguments', '')})")


async def main():
  # --- 2. Build a registry with only our two skills ---
  registry = SkillRegistry(
    skills=[backend_skill, frontend_skill],
    include_library=False,
  )
  print(f"Registry: {registry}")
  print(f"Skills: {[m.name for m in registry.list_skills()]}\n")

  # --- 3. Create agent with lazy skill selection ---
  agent = Agent(
    model="openai/gpt-4o-mini",
    skills=[registry.as_lazy()],
    instructions="You are a coding assistant. Before answering, load the most relevant skill from your catalog.",
  )
  print(f"Agent tools: {agent.tool_names}")
  print()

  # --- 4. Run a back-end task ---
  print("=" * 60)
  print("TASK 1: Back-end engineering")
  print("=" * 60)
  result = await agent.arun("Design a Python REST endpoint for user registration that hashes passwords and returns a JWT.")
  print("\nTool calls made:")
  print_tool_calls(result)
  print(f"\nResponse (first 300 chars):\n{str(result.content)[:300]}...\n")

  # --- 5. Run a front-end task ---
  print("=" * 60)
  print("TASK 2: Front-end engineering")
  print("=" * 60)
  result = await agent.arun("Build a React search bar component with debounced input, loading spinner, and keyboard navigation for results.")
  print("\nTool calls made:")
  print_tool_calls(result)
  print(f"\nResponse (first 300 chars):\n{str(result.content)[:300]}...\n")


if __name__ == "__main__":
  asyncio.run(main())
