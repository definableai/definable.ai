"""Full-featured coding agent built with ClaudeCodeAgent.

A realistic coding assistant that can:
  - Read, write, edit, and search code via Claude Code CLI
  - Run tests, lint, and type-check via custom tools
  - Look up project docs via knowledge/RAG
  - Remember context across multi-turn sessions via memory
  - Block destructive operations via guardrails
  - Stream progress events in real-time
  - Return structured analysis via output schemas

Prerequisites:
  - Claude Code CLI: npm install -g @anthropic-ai/claude-code
  - Authenticated: claude auth
  - pip install -e "." (Definable)

Usage:
  .venv/bin/python definable/examples/claude_code/03_coding_agent.py
"""

import asyncio
from typing import Optional

from definable.agent.guardrail import Guardrails, tool_blocklist
from definable.agent.tracing import JSONLExporter, Tracing
from definable.claude_code import ClaudeCodeAgent
from definable.knowledge import Document, Knowledge
from definable.memory import Memory
from definable.memory.store.in_memory import InMemoryStore
from definable.tool.decorator import tool
from definable.vectordb import InMemoryVectorDB
from pydantic import BaseModel, Field

# =============================================================================
# Custom tools — exposed to Claude Code via MCP protocol
# =============================================================================


@tool
def run_tests(path: str = ".", verbose: bool = False) -> str:
  """Run pytest on the given path and return the summary."""
  import subprocess

  cmd = ["python", "-m", "pytest", path, "-q", "--tb=short"]
  if verbose:
    cmd.append("-v")
  try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    output = result.stdout + result.stderr
    return output[-2000:] if len(output) > 2000 else output
  except subprocess.TimeoutExpired:
    return "ERROR: Tests timed out after 120 seconds."
  except Exception as e:
    return f"ERROR: Failed to run tests: {e}"


@tool
def run_linter(path: str = ".") -> str:
  """Run ruff linter and return any issues found."""
  import subprocess

  try:
    result = subprocess.run(
      ["python", "-m", "ruff", "check", path, "--output-format=concise"],
      capture_output=True,
      text=True,
      timeout=30,
    )
    if result.returncode == 0:
      return "No lint issues found."
    return result.stdout[-2000:]
  except Exception as e:
    return f"ERROR: Linter failed: {e}"


@tool
def run_type_check(path: str = ".") -> str:
  """Run mypy type checker and return results."""
  import subprocess

  try:
    result = subprocess.run(
      ["python", "-m", "mypy", path, "--no-error-summary"],
      capture_output=True,
      text=True,
      timeout=60,
    )
    output = result.stdout + result.stderr
    if result.returncode == 0:
      return "No type errors found."
    return output[-2000:]
  except Exception as e:
    return f"ERROR: Type check failed: {e}"


@tool
def search_github_issues(query: str, repo: str = "myorg/myapp") -> str:
  """Search GitHub issues for context on known bugs or feature requests."""
  # In a real app, this would call the GitHub API
  issues = [
    {"number": 142, "title": "Auth tokens not refreshing on mobile", "state": "open"},
    {"number": 138, "title": "Rate limiter not respecting per-user quotas", "state": "open"},
    {"number": 127, "title": "Alembic migration fails on Postgres 16", "state": "closed"},
  ]
  matches = [i for i in issues if query.lower() in i["title"].lower()]
  if not matches:
    return f"No issues matching '{query}' in {repo}."
  lines = [f"#{i['number']} [{i['state']}] {i['title']}" for i in matches]
  return "\n".join(lines)


# =============================================================================
# Structured output schemas
# =============================================================================


class CodeReview(BaseModel):
  """Structured output for code review results."""

  summary: str = Field(description="One-paragraph summary of the review")
  issues: list[str] = Field(description="List of issues found", default_factory=list)
  suggestions: list[str] = Field(description="List of improvement suggestions", default_factory=list)
  risk_level: str = Field(description="Risk level: low, medium, high")
  tests_needed: bool = Field(description="Whether new tests are needed")


# =============================================================================
# Knowledge base — project documentation
# =============================================================================


def build_knowledge_base() -> Knowledge:
  """Build a knowledge base with project documentation."""
  kb = Knowledge(vector_db=InMemoryVectorDB(), top_k=3)
  kb.add([
    # Architecture docs
    Document(
      content=(
        "The application uses a layered architecture: API routes (FastAPI)"
        " → Service layer → Repository layer → PostgreSQL."
        " All business logic lives in services, never in routes."
      ),
      meta_data={"source": "architecture.md", "section": "overview"},
    ),
    Document(
      content=(
        "Authentication uses JWT tokens with 24-hour expiry."
        " Refresh tokens last 30 days. Tokens are stored in HTTP-only cookies."
        " The auth middleware is in app/middleware/auth.py."
      ),
      meta_data={"source": "auth.md", "section": "jwt"},
    ),
    Document(
      content=(
        "Database migrations use Alembic."
        " Create: `alembic revision --autogenerate -m 'description'`."
        " Apply: `alembic upgrade head`. Rollback: `alembic downgrade -1`."
      ),
      meta_data={"source": "database.md", "section": "migrations"},
    ),
    Document(
      content=(
        "The API rate limit is 100 requests per minute per user,"
        " enforced by Redis-based sliding window."
        " Config in app/config/rate_limit.py."
        " Override per-route with @rate_limit(max=N) decorator."
      ),
      meta_data={"source": "api.md", "section": "rate-limiting"},
    ),
    Document(
      content=(
        "Testing conventions: Use pytest with fixtures in conftest.py."
        " Integration tests in tests/integration/ use a real test database."
        " Unit tests in tests/unit/ must mock all external calls."
        " Run with: pytest tests/ -q"
      ),
      meta_data={"source": "testing.md", "section": "conventions"},
    ),
    Document(
      content=(
        "Code style: Python 3.12, ruff for linting (line length 120),"
        " mypy for type checking (strict mode)."
        " All functions must have type annotations. Use Google-style docstrings."
      ),
      meta_data={"source": "style.md", "section": "python"},
    ),
    Document(
      content=(
        "The background task system uses Celery with Redis broker."
        " Task definitions in app/tasks/."
        " Schedule config in app/config/celery.py."
        " Monitor with: celery -A app flower."
      ),
      meta_data={"source": "architecture.md", "section": "async-tasks"},
    ),
  ])
  return kb


# =============================================================================
# Event handler — track what the agent does
# =============================================================================


def _truncate(text: str, max_len: int = 120) -> str:
  """Truncate text for display."""
  if not text:
    return ""
  text = text.replace("\n", " ").strip()
  return text[:max_len] + "..." if len(text) > max_len else text


def on_agent_event(event):
  """Rich event handler — shows tool calls, thinking, content in real-time."""
  event_type = type(event).__name__

  if event_type == "RunStartedEvent":
    print(f"  [start] Run {event.run_id[:8]}... model={getattr(event, 'model', '?')}")

  elif event_type == "ToolCallStartedEvent":
    tool = getattr(event, "tool", None)
    if tool:
      args_preview = _truncate(str(tool.tool_args or {}), 80)
      print(f"  [tool] {tool.tool_name}({args_preview})")

  elif event_type == "ToolCallCompletedEvent":
    tool = getattr(event, "tool", None)
    content = getattr(event, "content", "")
    result_len = len(content) if content else 0
    name = tool.tool_name if tool and tool.tool_name else "?"
    print(f"  [tool done] {name} ({result_len} chars)")

  elif event_type == "ReasoningContentDeltaEvent":
    thinking = getattr(event, "reasoning_content", "")
    print(f"  [thinking] {_truncate(thinking, 80)}")

  elif event_type == "RunContentEvent":
    content = getattr(event, "content", "")
    print(f"  [content] {_truncate(content)}")

  elif event_type == "RunCompletedEvent":
    metrics = getattr(event, "metrics", None)
    if metrics:
      cost_str = f"${metrics.cost:.4f}" if getattr(metrics, "cost", None) else "n/a"
      print(f"  [done] {getattr(metrics, 'input_tokens', 0):,} in / {getattr(metrics, 'output_tokens', 0):,} out | cost: {cost_str}")
    else:
      print(f"  [done] Run {event.run_id[:8]}...")

  elif event_type == "RunErrorEvent":
    print(f"  [error] {getattr(event, 'content', 'unknown')}")


# =============================================================================
# Build the coding agent
# =============================================================================


def create_coding_agent(
  cwd: str = ".",
  user_id: Optional[str] = None,
) -> ClaudeCodeAgent:
  """Create a fully-featured coding agent."""
  agent = ClaudeCodeAgent(
    model="claude-sonnet-4-6",
    instructions="""\
You are a senior software engineer working on a Python web application.

Your workflow:
1. Understand the task by reading relevant code and docs
2. Plan your changes before writing code
3. Make small, focused changes
4. Run tests after every change
5. Run the linter to catch style issues
6. If tests fail, fix them before moving on

Rules:
- Never commit directly — only make changes locally
- Always run tests after modifying code
- Follow the project's code style (check knowledge base)
- Search GitHub issues for context on bugs before fixing
- Explain your reasoning before making changes
""",
    # Claude Code built-in tools
    allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
    cwd=cwd,
    max_turns=25,
    max_budget_usd=5.0,
    thinking_budget_tokens=10000,
    # Custom tools — run tests, lint, type-check, search issues
    tools=[run_tests, run_linter, run_type_check, search_github_issues],
    # Knowledge — project docs injected into system prompt via RAG
    knowledge=build_knowledge_base(),
    # Memory — remembers across sessions (what was done, decisions made)
    memory=Memory(store=InMemoryStore()),
    # Guardrails — block dangerous tool usage
    guardrails=Guardrails(
      tool=[
        tool_blocklist({
          "rm -rf",
          "DROP TABLE",
          "DELETE FROM",
          "git push --force",
          "git reset --hard",
        }),
      ],
    ),
    # Tracing — write JSONL traces for debugging
    tracing=Tracing(exporters=[JSONLExporter(trace_dir="./traces")]),
  )

  agent.on_event(on_agent_event)
  return agent


# =============================================================================
# Example sessions
# =============================================================================


async def demo_basic_task():
  """Demo: Ask the agent to explore and analyze code."""
  print("=" * 60)
  print("DEMO 1: Basic code exploration")
  print("=" * 60)

  agent = create_coding_agent()

  result = await agent.arun(
    "List the Python files in this directory and give me a brief summary of the project structure.",
    user_id="dev-alice",
  )

  print(f"\nResponse:\n{result.content}")
  print(f"\nSession: {result.session_id}")
  print(f"Tools used: {[t.tool_name for t in (result.tools or [])]}")
  if result.metrics:
    print(f"Tokens: {result.metrics.input_tokens} in / {result.metrics.output_tokens} out")
    if result.metrics.cost:
      print(f"Cost: ${result.metrics.cost:.4f}")

  return result


async def demo_multi_turn():
  """Demo: Multi-turn coding session with context."""
  print("\n" + "=" * 60)
  print("DEMO 2: Multi-turn coding session")
  print("=" * 60)

  agent = create_coding_agent()

  # Turn 1: Explore
  print("\n--- Turn 1: Explore ---")
  r1 = await agent.arun(
    "Look at the Python files in this project. What testing framework do they use?",
    user_id="dev-alice",
  )
  print(f"Response: {str(r1.content)[:300]}")

  # Turn 2: Follow-up using session context
  print("\n--- Turn 2: Follow-up ---")
  r2 = await agent.arun(
    "Based on what you found, run the linter on the project and report any issues.",
    user_id="dev-alice",
    session_id=r1.session_id,
  )
  print(f"Response: {str(r2.content)[:300]}")

  total_cost = (getattr(r1.metrics, "cost", 0) or 0) + (getattr(r2.metrics, "cost", 0) or 0)
  print(f"\nTotal cost for 2 turns: ${total_cost:.4f}")

  return r2


async def demo_streaming():
  """Demo: Stream events as the agent works."""
  print("\n" + "=" * 60)
  print("DEMO 3: Streaming events")
  print("=" * 60)

  agent = create_coding_agent()

  async for event in agent.arun_stream(
    "What files are in the current directory? List them briefly.",
    user_id="dev-alice",
  ):
    event_type = type(event).__name__
    if event_type == "RunStartedEvent":
      print("  [stream] Agent started working...")
    elif event_type == "ToolCallStartedEvent":
      tool = getattr(event, "tool", None)
      if tool:
        print(f"  [stream:tool] {tool.tool_name}({_truncate(str(tool.tool_args or {}), 60)})")
    elif event_type == "ToolCallCompletedEvent":
      content = getattr(event, "content", "")
      print(f"  [stream:tool done] ({len(content) if content else 0} chars)")
    elif event_type == "ReasoningContentDeltaEvent":
      print(f"  [stream:thinking] {_truncate(getattr(event, 'reasoning_content', ''), 60)}")
    elif event_type == "RunContentEvent":
      content = getattr(event, "content", "")
      if content:
        print(f"  [stream:content] {_truncate(content, 100)}")
    elif event_type == "RunCompletedEvent":
      metrics = getattr(event, "metrics", None)
      if metrics:
        cost_str = f"${metrics.cost:.4f}" if getattr(metrics, "cost", None) else "n/a"
        print(f"  [stream:done] {getattr(metrics, 'input_tokens', 0):,} in / {getattr(metrics, 'output_tokens', 0):,} out | cost: {cost_str}")
      else:
        print("  [stream:done]")
    elif event_type == "RunErrorEvent":
      print(f"  [stream:error] {getattr(event, 'content', 'unknown')}")


async def main():
  """Run all demos sequentially."""
  print("ClaudeCodeAgent — Coding Agent Examples\n")

  # Run basic demo
  await demo_basic_task()

  # Multi-turn demo
  await demo_multi_turn()

  # Streaming demo
  await demo_streaming()

  print("\n" + "=" * 60)
  print("All demos complete. Traces written to ./traces/")


if __name__ == "__main__":
  asyncio.run(main())
