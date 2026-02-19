"""ClaudeCodeAgent with full Definable features.

Demonstrates: memory, knowledge/RAG, custom tools, guardrails, and tracing.

Requires: Claude Code CLI installed and authenticated.
"""

import asyncio

from definable.agent.tracing import Tracing
from definable.memory import Memory
from definable.claude_code import ClaudeCodeAgent
from definable.agent.guardrail import Guardrails, tool_blocklist
from definable.knowledge import Document, Knowledge
from definable.memory.store.in_memory import InMemoryStore
from definable.tool.decorator import tool
from definable.vectordb import InMemoryVectorDB


# --- Custom tools (exposed to Claude Code via MCP protocol) ---


@tool
def query_database(sql: str) -> str:
  """Execute a read-only SQL query against the application database."""
  # In a real app, this would connect to your DB
  return f"Query result for: {sql} → 42 rows returned"


@tool
def deploy_to_staging(branch: str = "main") -> str:
  """Deploy a branch to the staging environment."""
  return f"Deployed branch '{branch}' to staging. URL: https://staging.myapp.com"


async def main():
  # --- Knowledge base (RAG) ---
  kb = Knowledge(vector_db=InMemoryVectorDB(), top_k=3)
  kb.add(Document(content="The auth system uses JWT tokens with 24-hour expiry.", meta_data={"source": "auth-docs"}))
  kb.add(Document(content="Database migrations use Alembic. Run `alembic upgrade head`.", meta_data={"source": "db-docs"}))
  kb.add(Document(content="The API rate limit is 100 requests per minute per user.", meta_data={"source": "api-docs"}))

  # --- Agent with all features ---
  agent = ClaudeCodeAgent(
    model="claude-sonnet-4-6",
    instructions=(
      "You are a senior backend developer for the Lovable app. Use your tools to read, edit, and test code. Always run tests after making changes."
    ),
    allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
    cwd=".",
    max_turns=20,
    max_budget_usd=5.0,
    # Definable features:
    memory=Memory(store=InMemoryStore()),
    knowledge=kb,
    tools=[query_database, deploy_to_staging],
    guardrails=Guardrails(
      tool=[tool_blocklist({"Bash"})],  # Block direct bash access from Claude Code
    ),
    tracing=Tracing(),
  )

  # --- First turn ---
  r1 = await agent.arun(
    "What's the auth token expiry? And check if there are any Python files in the current directory.",
    user_id="dev-alice",
  )
  print("=== Turn 1 ===")
  print(f"Content: {r1.content[:200]}..." if r1.content and len(str(r1.content)) > 200 else f"Content: {r1.content}")
  print(f"Session: {r1.session_id}")
  print(f"Tools: {[t.tool_name for t in (r1.tools or [])]}")
  print()

  # --- Second turn (multi-turn via session) ---
  r2 = await agent.arun(
    "Now deploy to staging.",
    user_id="dev-alice",
    session_id=r1.session_id,
  )
  print("=== Turn 2 ===")
  print(f"Content: {r2.content[:200]}..." if r2.content and len(str(r2.content)) > 200 else f"Content: {r2.content}")
  print(f"Total cost: ${(r1.metrics.cost or 0) + (r2.metrics.cost or 0):.4f}")  # type: ignore[union-attr]


if __name__ == "__main__":
  asyncio.run(main())
