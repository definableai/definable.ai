"""
Pipeline Customization — Building a Production-Grade Agent

This example walks through the full power of Definable's Agent Pipeline:
  1. Inspect the default 8-phase pipeline
  2. Write and insert a custom phase
  3. Register before/after hooks for observability
  4. Replace a phase entirely with an "instead" hook
  5. Read phase metrics and stream phase events
  6. Use ToolRetry for model-feedback retries
  7. Attach a DebugConfig inspector callback
  8. See conditional phase skipping (should_run)
  9. Remove a phase at runtime

Each section is self-contained — copy the pattern you need.

Requirements:
    export OPENAI_API_KEY=sk-...
"""

import asyncio

from definable.agent import Agent
from definable.agent.events import PhaseCompletedEvent, PhaseStartedEvent, RunCompletedEvent, RunContentEvent
from definable.agent.pipeline import BasePhase, DebugConfig, LoopState, ToolRetry
from definable.model.message import Message
from definable.tool import tool

# ── Tools ────────────────────────────────────────────────────────────


@tool(name="lookup_user", description="Look up a user by username (min 3 chars)")
def lookup_user(username: str) -> str:
  """Look up user details. Raises ToolRetry if username is too short."""
  if len(username) < 3:
    raise ToolRetry("Username must be at least 3 characters. Try again with a valid username.")
  return f"User '{username}' — joined 2024, role: engineer, team: platform"


@tool(name="get_weather", description="Get current weather for a city")
def get_weather(city: str) -> str:
  """Get weather information."""
  return f"Sunny, 72°F in {city}"


# ── Custom Phase ─────────────────────────────────────────────────────


class RequestLoggerPhase(BasePhase):
  """Logs every request as it enters the pipeline.

  This is a minimal custom phase — override _name and execute().
  """

  _name = "request_logger"

  async def execute(self, state):
    preview = ""
    if state.new_messages:
      preview = (state.new_messages[0].content or "")[:60]
    print(f"  [request_logger] run={state.run_id[:8]}... input={preview!r}")
    yield state, None


# ── Helpers ──────────────────────────────────────────────────────────


def section(num, title):
  print(f"\n{'=' * 60}")
  print(f"  {num}. {title}")
  print(f"{'=' * 60}\n")


def print_phase_metrics(result):
  """Pretty-print phase timing table from RunOutput."""
  if not result.phase_metrics:
    print("  (no phase metrics)")
    return
  print(f"  {'Phase':<20} {'Duration':>10} {'Skipped':>8}")
  print(f"  {'-' * 20} {'-' * 10} {'-' * 8}")
  for m in result.phase_metrics:
    dur = f"{m.duration_ms:.1f}ms" if not m.skipped else "-"
    skip = "YES" if m.skipped else ""
    print(f"  {m.phase_name:<20} {dur:>10} {skip:>8}")


# ── Section 1: Inspect the Default Pipeline ──────────────────────────


async def inspect_pipeline():
  section(1, "Inspect the Default Pipeline")

  agent = Agent(
    model="openai/gpt-4o-mini",
    instructions="Be concise.",
  )

  print("  Default 8-phase pipeline:")
  for i, name in enumerate(agent.pipeline.phase_names, 1):
    print(f"    {i}. {name}")

  print("""
  Phase summary:
    prepare      — Normalize input, build RunContext, prepare tools
    recall       — Retrieve memory + knowledge + readers (parallel)
    think        — Reasoning / thinking layer
    guard_input  — Input guardrail checks
    compose      — Build system prompt + invoke_messages
    invoke_loop  — Main LLM call loop (handles tool calls)
    guard_output — Output guardrail checks
    store        — Persist to memory + emit tracing
  """)
  return agent


# ── Section 2: Custom Phase ──────────────────────────────────────────


async def custom_phase(agent):
  section(2, "Custom Phase — Request Logging")

  print(f"  Before: {agent.pipeline.phase_names}")

  # Insert our logger right after recall
  agent.pipeline.add_phase(RequestLoggerPhase(), after="recall")

  print(f"  After:  {agent.pipeline.phase_names}")
  print()

  # Run — the logger will print during execution
  result = await agent.arun("What's the weather in Tokyo?")
  print(f"\n  Agent: {(result.content or '')[:100]}")
  return agent


# ── Section 3: Hooks — Observability ─────────────────────────────────


async def hooks_observability(agent):
  section(3, "Hooks — Before/After Observability")

  # Register hooks with the @agent.hook() decorator
  @agent.hook("before:invoke_loop")
  async def on_before_invoke(state: LoopState):
    print(f"  [before:invoke_loop] {len(state.all_messages)} messages, {len(state.tools)} tools")
    return state

  @agent.hook("after:invoke_loop")
  async def on_after_invoke(state: LoopState):
    has_content = bool(state.content)
    print(f"  [after:invoke_loop] content_generated={has_content}")
    return state

  result = await agent.arun("Look up user 'alice' for me.")
  print(f"\n  Agent: {(result.content or '')[:100]}")
  return agent


# ── Section 4: Instead Hook — Replace a Phase ───────────────────────


async def instead_hook():
  section(4, "Instead Hook — Replace Compose Phase")

  agent = Agent(
    model="openai/gpt-4o-mini",
    tools=[get_weather],
    instructions="Be concise.",
  )

  @agent.hook("instead:compose")
  async def custom_compose(state: LoopState):
    # Build invoke_messages manually with a custom system prompt
    state.invoke_messages = [
      Message(role="system", content="You are a weather expert. Always be enthusiastic about weather!"),
      *state.all_messages,
    ]
    print("  [instead:compose] Custom compose replaced the default!")
    yield state, None

  result = await agent.arun("Weather in Paris?")
  print(f"\n  Agent: {(result.content or '')[:100]}")


# ── Section 5: Phase Metrics & Streaming Events ─────────────────────


async def phase_metrics_and_streaming():
  section(5, "Phase Metrics & Streaming Events")

  agent = Agent(
    model="openai/gpt-4o-mini",
    tools=[get_weather],
    instructions="Be concise.",
  )

  # First: non-streaming — inspect phase_metrics on RunOutput
  result = await agent.arun("Weather in London?")
  print("  Phase timing (from RunOutput.phase_metrics):")
  print_phase_metrics(result)

  # Second: streaming — collect PhaseStarted/PhaseCompleted events
  print("\n  Streaming phase events:")
  content_chunks = []

  async for event in agent.arun_stream("Weather in Berlin?"):
    if isinstance(event, PhaseStartedEvent):
      print(f"    >> PhaseStarted: {event.phase_name}")
    elif isinstance(event, PhaseCompletedEvent):
      dur = f" ({event.duration_ms:.1f}ms)" if not event.skipped else " (skipped)"
      print(f"    << PhaseCompleted: {event.phase_name}{dur}")
    elif isinstance(event, RunContentEvent):
      content_chunks.append(event.content)
    elif isinstance(event, RunCompletedEvent):
      print(f"\n  Agent: {(event.content or '')[:100]}")


# ── Section 6: ToolRetry ─────────────────────────────────────────────


async def tool_retry_demo():
  section(6, "ToolRetry — Model-Feedback Retries")

  print("  The lookup_user tool raises ToolRetry if username < 3 chars.")
  print("  The model receives the error message and self-corrects.\n")

  agent = Agent(
    model="openai/gpt-4o-mini",
    tools=[lookup_user],
    instructions="You are a helpful assistant. When asked to look up a user, call the lookup_user tool.",
  )

  result = await agent.arun("Look up user 'alice'")
  print(f"  Agent: {(result.content or '')[:120]}")


# ── Section 7: DebugConfig — Inspector Callback ─────────────────────


async def debug_inspector():
  section(7, "DebugConfig — Inspector Callback")

  inspected = []

  async def my_inspector(state, phase_name):
    inspected.append(phase_name)

  agent = Agent(
    model="openai/gpt-4o-mini",
    instructions="Be concise.",
    debug=DebugConfig(
      step_mode=True,
      inspector=my_inspector,
      enable_trace=False,  # Suppress DebugExporter stderr noise
    ),
  )

  result = await agent.arun("Say hello in one word.")
  print(f"  Agent: {(result.content or '')[:60]}")
  print(f"  Inspector called for: {inspected}")


# ── Section 8: Conditional Execution — should_run ────────────────────


async def conditional_execution():
  section(8, "Conditional Execution — should_run")

  # A bare agent with no memory, knowledge, thinking, or guardrails
  agent = Agent(
    model="openai/gpt-4o-mini",
    instructions="Be concise.",
  )

  result = await agent.arun("Hi")
  print(f"  Agent: {(result.content or '')[:60]}\n")

  executed = [m.phase_name for m in (result.phase_metrics or []) if not m.skipped]
  skipped = [m.phase_name for m in (result.phase_metrics or []) if m.skipped]

  print(f"  Executed phases: {executed}")
  print(f"  Skipped phases:  {skipped}")
  print()
  print("  Phases like recall, think, guard_input, guard_output, store")
  print("  are automatically skipped when their feature isn't configured.")


# ── Section 9: Phase Removal ─────────────────────────────────────────


async def phase_removal():
  section(9, "Phase Removal")

  agent = Agent(
    model="openai/gpt-4o-mini",
    instructions="Be concise.",
  )

  # Add a custom phase, then remove it
  agent.pipeline.add_phase(RequestLoggerPhase(), after="prepare")
  print(f"  After add:    {agent.pipeline.phase_names}")

  agent.pipeline.remove_phase("request_logger")
  print(f"  After remove: {agent.pipeline.phase_names}")
  print()
  print("  Use remove_phase() for runtime pipeline modification —")
  print("  e.g., disabling a logging phase in production.")


# ── Main ─────────────────────────────────────────────────────────────


async def main():
  print("Pipeline Customization — Full Walkthrough")
  print("=" * 60)

  # Sections 1-3 share an agent (progressive build-up)
  agent = await inspect_pipeline()
  agent.tools = [lookup_user, get_weather]
  agent = await custom_phase(agent)
  await hooks_observability(agent)

  # Sections 4-9 are self-contained
  await instead_hook()
  await phase_metrics_and_streaming()
  await tool_retry_demo()
  await debug_inspector()
  await conditional_execution()
  await phase_removal()

  print(f"\n{'=' * 60}")
  print("  Done — all 9 sections complete.")
  print(f"{'=' * 60}")


if __name__ == "__main__":
  asyncio.run(main())
