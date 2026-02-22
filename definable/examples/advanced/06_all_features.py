"""
All-Features Showcase — Definable AI Framework
================================================
A narrative-driven walkthrough of every major feature. Run it to see
everything in action and copy patterns for your own agents.

16 sections, progressive complexity:
  1. Agent + Tools          9. Guardrails
  2. Knowledge (RAG)       10. Tracing
  3. Memory                11. Pipeline Customization
  4. Multi-Turn            12. ToolRetry
  5. Thinking              13. CancellationToken
  6. Structured Output     14. Debug Mode
  7. Streaming             15. Sub-Agents
  8. Event Callbacks       16. Full Stack (everything)

Required:
  OPENAI_API_KEY

Usage:
  source .env.test && .venv/bin/python test_all_features.py
"""

import asyncio
import json
import os
import shutil
import tempfile
import time
from typing import AsyncGenerator, Optional, Tuple

from definable.agent import Agent
from definable.agent.cancellation import AgentCancelled, CancellationToken
from definable.agent.events import (
  RunCompletedEvent,
  RunContentEvent,
  RunStartedEvent,
  SubAgentCompletedEvent,
  ToolCallCompletedEvent,
)
from definable.agent.guardrail.base import GuardrailResult, Guardrails
from definable.agent.pipeline import BasePhase, DebugConfig, LoopState, SubAgentPolicy, ToolRetry
from definable.agent.reasoning import Thinking
from definable.agent.run.base import BaseRunOutputEvent, RunContext
from definable.agent.tracing import JSONLExporter, Tracing
from definable.knowledge import Document, Knowledge
from definable.knowledge.embedder.openai import OpenAIEmbedder
from definable.memory import Memory, SQLiteStore
from definable.tool import tool
from definable.vectordb import InMemoryVectorDB
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEMP_DIR = tempfile.mkdtemp(prefix="definable_showcase_")
MODEL = "openai/gpt-5.2"


def section(num: int, title: str) -> None:
  print(f"\n{'=' * 60}")
  print(f"  [{num}/16] {title}")
  print(f"{'=' * 60}\n")


def print_phase_metrics(result) -> None:
  if not result.phase_metrics:
    print("  (no phase metrics)")
    return
  print(f"  {'Phase':<20} {'Duration':>10} {'Skipped':>8}")
  print(f"  {'-' * 40}")
  for m in result.phase_metrics:
    skip = "yes" if m.skipped else ""
    print(f"  {m.phase_name:<20} {m.duration_ms:>8.1f}ms {skip:>8}")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def calculate(expression: str) -> str:
  """Evaluate a math expression and return the result."""
  try:
    result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
    return str(result)
  except Exception as e:
    return f"Error: {e}"


@tool
def lookup_company_info(topic: str) -> str:
  """Look up internal company information by topic."""
  facts = {
    "founding": "Acme Corp was founded in 2020 in San Francisco.",
    "pto": "Employees receive 25 days of paid time off per year.",
    "stack": "The engineering team uses Python and TypeScript.",
    "mission": "Our mission is to make AI accessible to every developer.",
    "size": "We have 150 employees across 12 countries.",
  }
  for key, value in facts.items():
    if key in topic.lower():
      return value
  return f"No information found for topic: {topic}"


@tool
def lookup_user(username: str) -> str:
  """Look up a user profile by username. Username must be at least 3 characters."""
  if len(username) < 3:
    raise ToolRetry(f"Username '{username}' is too short. Provide at least 3 characters.")
  return f"User '{username}' — role: engineer, team: platform, joined: 2023-06-15"


@tool
def slow_task(seconds: float) -> str:
  """Simulate a long-running task that takes the specified number of seconds."""
  import time as _time

  _time.sleep(seconds)
  return f"Task completed after {seconds}s"


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


class ContentFilter:
  """Input guardrail that blocks messages containing 'BLOCKED'."""

  name = "content_filter"

  async def check(self, text: str, context: RunContext) -> GuardrailResult:
    if "BLOCKED" in text:
      return GuardrailResult.block("Input contains forbidden keyword.")
    return GuardrailResult.allow()


class LengthChecker:
  """Output guardrail that warns when the response is very long."""

  name = "length_checker"

  async def check(self, text: str, context: RunContext) -> GuardrailResult:
    if len(text) > 2000:
      return GuardrailResult.warn(f"Response is {len(text)} chars — consider summarizing.")
    return GuardrailResult.allow()


# ---------------------------------------------------------------------------
# Custom pipeline phase
# ---------------------------------------------------------------------------


class RequestLoggerPhase(BasePhase):
  """Custom pipeline phase that logs the incoming request."""

  _name = "request_logger"

  async def execute(self, state: LoopState) -> AsyncGenerator[Tuple[LoopState, Optional[BaseRunOutputEvent]], None]:
    preview = (state.raw_instruction or "")[:80]
    print(f"  [RequestLogger] run_id={state.run_id[:8]}... input='{preview}'")
    state.extra["request_logged"] = True
    yield state, None


# ---------------------------------------------------------------------------
# Structured output model
# ---------------------------------------------------------------------------


class CompanyReport(BaseModel):
  title: str
  key_points: list[str]
  sentiment: str


# ============================= SECTIONS =====================================


async def section_01_foundations():
  """Agent + Tools + String Shorthand"""
  section(1, "Foundations — Agent + Tools + String Shorthand")

  agent = Agent(
    model=MODEL,
    tools=[calculate],
    instructions="You are a helpful math assistant. Use the calculate tool for arithmetic.",
  )
  output = await agent.arun("What is 47 * 89?")

  print(f"  Answer: {output.content}")
  print(f"  Model: {output.model}")
  if output.metrics:
    print(f"  Tokens: {output.metrics.input_tokens} in / {output.metrics.output_tokens} out")
  if output.tools:
    print(f"  Tools used: {[t.tool_name for t in output.tools]}")


async def section_02_knowledge():
  """Knowledge — RAG"""
  section(2, "Knowledge — RAG")

  kb = Knowledge(
    vector_db=InMemoryVectorDB(),
    embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    trigger="always",
    top_k=3,
  )
  kb.add([
    Document(content="Acme Corp was founded in 2020 in San Francisco.", meta_data={"source": "wiki"}),
    Document(content="Employees receive 25 days of paid time off per year.", meta_data={"source": "hr"}),
    Document(content="The engineering team uses Python and TypeScript.", meta_data={"source": "eng"}),
    Document(content="Our mission is to make AI accessible to every developer.", meta_data={"source": "about"}),
    Document(content="We have 150 employees across 12 countries.", meta_data={"source": "hr"}),
  ])

  agent = Agent(
    model=MODEL,
    knowledge=kb,
    instructions="Answer questions using the retrieved context. Be concise.",
  )
  output = await agent.arun("When was the company founded and what tech stack do they use?")

  print(f"  Answer: {output.content}")
  print("  (Agent was never told these facts — retrieved via RAG)")


async def section_03_memory():
  """Memory — Session History"""
  section(3, "Memory — Session History")

  db_path = os.path.join(TEMP_DIR, "memory.db")
  agent = Agent(
    model=MODEL,
    memory=Memory(store=SQLiteStore(db_path=db_path), max_messages=20),
    instructions="You are a friendly assistant. Remember what users tell you.",
  )

  async with agent:
    r1 = await agent.arun("Hi! My name is Alice and I'm a software engineer.", session_id="demo-session")
    print(f"  Turn 1: {r1.content}")

    r2 = await agent.arun("What's my name and what do I do?", session_id="demo-session")
    print(f"  Turn 2: {r2.content}")
    print("  (Memory auto-recalled session history — no messages= needed)")


async def section_04_multi_turn():
  """Multi-Turn — Explicit Message Threading"""
  section(4, "Multi-Turn — Explicit Message Threading")

  agent = Agent(
    model=MODEL,
    instructions="You are a helpful assistant. Be concise.",
  )

  r1 = await agent.arun("My favorite color is blue.")
  print(f"  Turn 1: {r1.content}")

  r2 = await agent.arun("What's my favorite color?", messages=r1.messages)
  print(f"  Turn 2: {r2.content}")
  print("  (Stateless threading via messages= — no Memory needed)")


async def section_05_thinking():
  """Thinking / Reasoning"""
  section(5, "Thinking / Reasoning")

  agent = Agent(
    model=MODEL,
    thinking=Thinking(trigger="always"),
    instructions="Think step by step. Show your reasoning.",
  )
  output = await agent.arun("Which is larger: 9.11 or 9.9? Think carefully.")

  print(f"  Answer: {output.content}")
  if output.reasoning_content:
    preview = output.reasoning_content[:200]
    print(f"  Reasoning: {preview}...")
  else:
    print("  (Model did not return separate reasoning content)")


async def section_06_structured_output():
  """Structured Output"""
  section(6, "Structured Output")

  agent = Agent(
    model=MODEL,
    instructions="You are an analyst. Produce structured reports.",
  )
  output = await agent.arun(
    "Summarize the key benefits of remote work for companies.",
    output_schema=CompanyReport,
  )

  report = output.content
  # Structured output may arrive as a parsed model or a JSON string
  if isinstance(report, str):
    report = CompanyReport.model_validate_json(report)
  if isinstance(report, CompanyReport):
    print(f"  Title: {report.title}")
    print(f"  Sentiment: {report.sentiment}")
    print(f"  Key points ({len(report.key_points)}):")
    for pt in report.key_points[:3]:
      print(f"    - {pt}")
  else:
    print(f"  Raw content: {str(output.content)[:200]}")


async def section_07_streaming():
  """Streaming"""
  section(7, "Streaming")

  agent = Agent(
    model=MODEL,
    instructions="You are a science teacher. Explain concepts clearly.",
  )

  chunks = 0
  print("  ", end="", flush=True)
  final_metrics = None

  async for event in agent.arun_stream("Explain the water cycle in 3 sentences."):
    if isinstance(event, RunContentEvent) and event.content:
      print(event.content, end="", flush=True)
      chunks += 1
    elif isinstance(event, RunCompletedEvent) and event.metrics:
      final_metrics = event.metrics

  print()
  print(f"  ({chunks} chunks streamed)")
  if final_metrics:
    print(f"  Tokens: {final_metrics.input_tokens} in / {final_metrics.output_tokens} out")


async def section_08_events():
  """Event Callbacks"""
  section(8, "Event Callbacks")

  agent = Agent(
    model=MODEL,
    tools=[calculate],
    instructions="You are a math assistant. Use the calculate tool.",
  )

  collected: list[str] = []

  @agent.events.on(RunStartedEvent)
  def on_start(event):
    collected.append(f"RunStarted(model={event.model})")

  @agent.events.on(ToolCallCompletedEvent)
  def on_tool(event):
    collected.append(f"ToolCompleted({event.tool.tool_name})")

  @agent.events.on(RunCompletedEvent)
  def on_complete(event):
    collected.append("RunCompleted")

  output = await agent.arun("Calculate 123 + 456 using the tool.")
  print(f"  Answer: {output.content}")
  print(f"  Events captured ({len(collected)}):")
  for e in collected:
    print(f"    - {e}")


async def section_09_guardrails():
  """Guardrails"""
  section(9, "Guardrails")

  agent = Agent(
    model=MODEL,
    guardrails=Guardrails(
      input=[ContentFilter()],
      output=[LengthChecker()],
      on_block="return_message",
    ),
    instructions="You are a helpful assistant.",
  )

  # Clean input — passes
  r1 = await agent.arun("What is the capital of France?")
  print(f"  Clean input → status={r1.status.value}, answer={str(r1.content)[:80]}")

  # Blocked input
  r2 = await agent.arun("This message contains BLOCKED content.")
  print(f"  Blocked input → status={r2.status.value}, message={str(r2.content)[:80]}")


async def section_10_tracing():
  """Tracing"""
  section(10, "Tracing")

  trace_dir = os.path.join(TEMP_DIR, "traces")
  os.makedirs(trace_dir, exist_ok=True)

  agent = Agent(
    model=MODEL,
    tracing=Tracing(exporters=[JSONLExporter(trace_dir=trace_dir)]),
    instructions="You are a helpful assistant.",
  )
  await agent.arun("Name three primary colors.")

  # Read the trace file
  trace_files = [f for f in os.listdir(trace_dir) if f.endswith(".jsonl")]
  if trace_files:
    trace_path = os.path.join(trace_dir, trace_files[0])
    with open(trace_path) as f:
      events = [json.loads(line) for line in f if line.strip()]
    event_types = {e.get("event_type", e.get("type", "unknown")) for e in events}
    print(f"  Trace file: {trace_files[0]}")
    print(f"  Events recorded: {len(events)}")
    print(f"  Event types: {sorted(event_types)}")
  else:
    print("  (no trace files generated)")


async def section_11_pipeline():
  """Pipeline Customization"""
  section(11, "Pipeline Customization")

  agent = Agent(
    model=MODEL,
    tools=[calculate],
    instructions="You are a math assistant. Use the calculate tool.",
  )

  # Show default phases
  print(f"  Default phases: {agent.pipeline.phase_names}")

  # Add custom phase
  agent.pipeline.add_phase(RequestLoggerPhase(), after="recall")

  # Add a hook
  @agent.hook("before:invoke_loop")
  async def log_message_count(state: LoopState) -> LoopState:
    msg_count = len(state.invoke_messages) if state.invoke_messages else 0
    print(f"  [Hook before:invoke_loop] {msg_count} messages entering LLM")
    return state

  print(f"  After customization: {agent.pipeline.phase_names}")

  output = await agent.arun("What is 7 * 13?")
  print(f"  Answer: {output.content}")
  print_phase_metrics(output)


async def section_12_tool_retry():
  """ToolRetry"""
  section(12, "ToolRetry")

  agent = Agent(
    model=MODEL,
    tools=[lookup_user],
    instructions=("You are a user directory assistant. Use the lookup_user tool to find users. Always pass the full username the user gives you."),
  )

  output = await agent.arun("Look up the user 'alice' for me.")
  print(f"  Answer: {output.content}")
  if output.tools:
    print(f"  Tool calls: {len(output.tools)}")
    for t in output.tools:
      print(f"    - {t.tool_name}({t.tool_args}) → {str(t.result)[:60]}")
  print("  (ToolRetry would fire if model tried a username < 3 chars)")


async def section_13_cancellation():
  """CancellationToken"""
  section(13, "CancellationToken")

  agent = Agent(
    model=MODEL,
    tools=[slow_task],
    instructions="When asked to run a slow task, use the slow_task tool with seconds=3.",
  )
  token = CancellationToken()

  async def run_and_cancel():
    await asyncio.sleep(0.5)
    token.cancel()
    print("  Token cancelled after 0.5s")

  try:
    cancel_task = asyncio.create_task(run_and_cancel())
    output = await agent.arun(
      "Run a slow task that takes 3 seconds.",
      cancellation_token=token,
    )
    await cancel_task
    # If we get here, the model responded before cancellation kicked in
    print(f"  Agent completed before cancellation: {str(output.content)[:80]}")
  except AgentCancelled:
    print("  AgentCancelled raised — cooperative cancellation worked")
  except Exception as e:
    print(f"  Caught: {type(e).__name__}: {e}")


async def section_14_debug_mode():
  """Debug Mode"""
  section(14, "Debug Mode")

  inspected_phases: list[str] = []

  def my_inspector(state: LoopState, phase_name: str) -> None:
    inspected_phases.append(phase_name)

  agent = Agent(
    model=MODEL,
    instructions="You are a helpful assistant. Be concise.",
  )
  # Set debug config directly on the pipeline to avoid DebugExporter stderr noise
  agent.pipeline._debug = DebugConfig(step_mode=True, inspector=my_inspector)

  output = await agent.arun("What is the tallest mountain on Earth?")
  print(f"  Answer: {str(output.content)[:80]}")
  print(f"  Phases inspected ({len(inspected_phases)}): {inspected_phases}")


async def section_15_sub_agents():
  """Sub-Agent Spawning"""
  section(15, "Sub-Agent Spawning")

  completed_sub_agents: list[str] = []

  @tool
  def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather = {"paris": "15C sunny", "tokyo": "22C cloudy", "nyc": "8C rainy"}
    return weather.get(city.lower(), f"No data for {city}")

  agent = Agent(
    model=MODEL,
    tools=[calculate, get_weather],
    sub_agents=SubAgentPolicy(max_concurrent=3, inherit_tools=True),
    instructions=(
      "You are a coordinator agent with sub-agents. When asked to do multiple "
      "independent tasks, use the spawn_agent tool to delegate each task to a "
      "separate sub-agent. Each sub-agent inherits your tools."
    ),
  )

  @agent.events.on(SubAgentCompletedEvent)
  def on_sub_complete(event):
    completed_sub_agents.append(str(event))

  output = await agent.arun("Do these tasks in parallel using spawn_agent:\n1. Calculate 99 * 77\n2. Get the weather in Paris")
  print(f"  Answer: {str(output.content)[:200]}")
  print(f"  Sub-agent events: {len(completed_sub_agents)}")
  if output.tools:
    print(f"  Tool calls: {[t.tool_name for t in output.tools]}")


async def section_16_full_stack():
  """Full Stack — Everything Together"""
  section(16, "Full Stack — Everything Together")

  trace_dir = os.path.join(TEMP_DIR, "traces_full")
  os.makedirs(trace_dir, exist_ok=True)
  db_path = os.path.join(TEMP_DIR, "full_stack.db")

  kb = Knowledge(
    vector_db=InMemoryVectorDB(),
    embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    trigger="always",
    top_k=3,
  )
  kb.add([
    Document(content="Our company Acme Corp was founded in 2020.", meta_data={"src": "wiki"}),
    Document(content="We use Python, TypeScript, and Go.", meta_data={"src": "eng"}),
    Document(content="Employees get 25 PTO days per year.", meta_data={"src": "hr"}),
  ])

  events_log: list[str] = []
  inspected: list[str] = []

  def full_inspector(state: LoopState, phase_name: str) -> None:
    inspected.append(phase_name)

  agent = Agent(
    model=MODEL,
    tools=[calculate, lookup_company_info],
    knowledge=kb,
    memory=Memory(store=SQLiteStore(db_path=db_path), max_messages=20),
    thinking=Thinking(trigger="auto"),
    tracing=Tracing(exporters=[JSONLExporter(trace_dir=trace_dir)]),
    guardrails=Guardrails(input=[ContentFilter()], output=[LengthChecker()], on_block="return_message"),
    instructions="You are Acme Corp's AI assistant. Use tools and knowledge to help employees.",
  )
  # Set debug directly on pipeline to get inspector without DebugExporter stderr noise
  agent.pipeline._debug = DebugConfig(step_mode=True, inspector=full_inspector)

  @agent.events.on(RunCompletedEvent)
  def track(event):
    events_log.append("RunCompleted")

  @agent.hook("before:invoke_loop")
  async def count_msgs(state: LoopState) -> LoopState:
    state.extra["msg_count"] = len(state.invoke_messages) if state.invoke_messages else 0
    return state

  async with agent:
    # Turn 1
    r1 = await agent.arun(
      "Hi, I'm Bob. When was our company founded, and what languages do we use?",
      session_id="full-stack-demo",
    )
    print(f"  Turn 1: {str(r1.content)[:150]}")

    # Turn 2 — memory recalls session history
    r2 = await agent.arun(
      "What's my name? And calculate 25 * 8 for me.",
      session_id="full-stack-demo",
    )
    print(f"  Turn 2: {str(r2.content)[:150]}")

  print(f"\n  Events: {len(events_log)} RunCompleted")
  print(f"  Phases inspected: {inspected}")
  print_phase_metrics(r2)
  print("\n  All features composed without conflict.")


# ===========================================================================
# Main
# ===========================================================================


async def main():
  start = time.time()

  sections = [
    section_01_foundations,
    section_02_knowledge,
    section_03_memory,
    section_04_multi_turn,
    section_05_thinking,
    section_06_structured_output,
    section_07_streaming,
    section_08_events,
    section_09_guardrails,
    section_10_tracing,
    section_11_pipeline,
    section_12_tool_retry,
    section_13_cancellation,
    section_14_debug_mode,
    section_15_sub_agents,
    section_16_full_stack,
  ]

  for fn in sections:
    try:
      await fn()
    except Exception as e:
      print(f"  ERROR: {type(e).__name__}: {e}")

  # Cleanup
  shutil.rmtree(TEMP_DIR, ignore_errors=True)

  elapsed = time.time() - start
  print(f"\n{'=' * 60}")
  print(f"  All 16 sections completed in {elapsed:.1f}s")
  print(f"{'=' * 60}")


if __name__ == "__main__":
  asyncio.run(main())
