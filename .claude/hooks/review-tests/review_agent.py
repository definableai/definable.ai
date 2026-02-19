#!/usr/bin/env python3
"""Review test: Agent core — construction, run, config, events.

Tests the Agent class directly with all constructor permutations,
RunOutput shape, multi-turn, event bus, and config.
"""

import sys
import os
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

passed, failed, skipped = 0, 0, 0


def check(name, condition, error=""):
  global passed, failed
  if condition:
    print(f"✅ PASS: {name}")
    passed += 1
  else:
    print(f"❌ FAIL: {name} — {error}")
    failed += 1


def skip(name, reason):
  global skipped
  print(f"⚠️  SKIP: {name} — {reason}")
  skipped += 1


# ── Imports ─────────────────────────────────────────────────────
try:
  from definable.agent import (
    Agent,
    AgentConfig,
    MockModel,
    Tracing,
  )
  from definable.agent.events import RunStatus

  check("Import agent core", True)
except Exception as e:
  check("Import agent core", False, str(e))
  print(f"\n{'=' * 60}\nRESULT: {passed} passed | {failed} failed | {skipped} skipped")
  sys.exit(1)


# ── Agent construction variants ─────────────────────────────────
try:
  Agent(model=MockModel())
  check("Agent(model=MockModel()) — bare minimum", True)
except Exception as e:
  check("Agent bare minimum", False, str(e))

try:
  Agent(model=MockModel(), instructions="Be helpful")
  check("Agent with instructions", True)
except Exception as e:
  check("Agent with instructions", False, str(e))

try:
  Agent(model=MockModel(), instructions="", name="test-agent", session_id="sess-001")
  check("Agent with empty instructions + name + session_id", True)
except Exception as e:
  check("Agent with name/session_id", False, str(e))

try:
  Agent(model=MockModel(), config=AgentConfig())
  check("Agent with explicit AgentConfig()", True)
except Exception as e:
  check("Agent with AgentConfig", False, str(e))

try:
  Agent(model=MockModel(), tools=[])
  check("Agent with empty tools list", True)
except Exception as e:
  check("Agent with empty tools", False, str(e))

try:
  Agent(model=MockModel(), skills=[])
  check("Agent with empty skills list", True)
except Exception as e:
  check("Agent with empty skills", False, str(e))

try:
  Agent(model=MockModel(), toolkits=[])
  check("Agent with empty toolkits list", True)
except Exception as e:
  check("Agent with empty toolkits", False, str(e))


# ── Agent.run() output shape ────────────────────────────────────
try:
  agent = Agent(model=MockModel(responses=["test output"]))
  output = agent.run("hello")

  check("output.content is str", isinstance(output.content, str), type(output.content).__name__)
  check("output.status is RunStatus", isinstance(output.status, RunStatus), type(output.status).__name__)
  check("output.messages is list", isinstance(output.messages, list), type(output.messages).__name__)
  check("output has metrics", hasattr(output, "metrics"), "no metrics attr")
  check("output has session_id", hasattr(output, "session_id"), "no session_id attr")
  check("output has tools attr", hasattr(output, "tools"), "no tools attr")
except Exception as e:
  check("Agent.run() output shape", False, str(e))


# ── Agent multi-turn ────────────────────────────────────────────
try:
  mock = MockModel(responses=["r1", "r2", "r3"])
  agent = Agent(model=mock)

  o1 = agent.run("turn 1")
  o2 = agent.run("turn 2", messages=o1.messages)
  o3 = agent.run("turn 3", messages=o2.messages)

  check("Multi-turn: 3 turns complete", o3.content == "r3", f"got: {o3.content!r}")
  check(
    "Multi-turn: messages grow monotonically",
    len(o3.messages) > len(o2.messages) > len(o1.messages),
    f"turn counts: {len(o1.messages)}, {len(o2.messages)}, {len(o3.messages)}",
  )
except Exception as e:
  check("Agent multi-turn", False, str(e))


# ── Agent async run ─────────────────────────────────────────────
try:
  agent = Agent(model=MockModel(responses=["async test"]))
  output = asyncio.run(agent.arun("async"))
  check("Agent.arun() async works", output.content == "async test", f"got: {output.content!r}")
except Exception as e:
  check("Agent.arun()", False, str(e))


# ── Agent with tracing disabled ─────────────────────────────────
try:
  agent = Agent(model=MockModel(), tracing=False)
  output = agent.run("traced")
  check("Agent with tracing=False runs", output.content is not None, "no content")
except Exception as e:
  check("Agent tracing=False", False, str(e))

try:
  agent = Agent(model=MockModel(), tracing=Tracing(enabled=False))
  output = agent.run("traced")
  check("Agent with Tracing(enabled=False) runs", output.content is not None, "no content")
except Exception as e:
  check("Agent Tracing object", False, str(e))


# ── Agent memory=True shorthand ─────────────────────────────────
try:
  agent = Agent(model=MockModel(), memory=True)
  check("Agent(memory=True) constructs", True)
except Exception as e:
  check("Agent memory=True", False, str(e))


# ── Agent knowledge=True should raise ──────────────────────────
try:
  Agent(model=MockModel(), knowledge=True)
  check("Agent(knowledge=True) raises ValueError", False, "should have raised but didn't")
except ValueError:
  check("Agent(knowledge=True) raises ValueError", True)
except Exception as e:
  check("Agent(knowledge=True) raises ValueError", False, f"raised {type(e).__name__}: {e}")


# ── Agent model=None error ──────────────────────────────────────
try:
  Agent(model=None)
  check("Agent(model=None) raises error", False, "should have raised but didn't")
except (TypeError, AttributeError, ValueError) as e:
  err_msg = str(e).lower()
  check("Agent(model=None) raises error", True)
except Exception as e:
  check("Agent(model=None) error", False, f"unexpected: {type(e).__name__}: {e}")


# ── MockModel reset ────────────────────────────────────────────
try:
  mock = MockModel(responses=["a"])
  agent = Agent(model=mock)
  agent.run("1")
  agent.run("2")
  check("MockModel call_count tracks", mock.call_count >= 2, f"count={mock.call_count}")
  mock.reset()
  check("MockModel.reset() clears count", mock.call_count == 0, f"count={mock.call_count}")
  check("MockModel.reset() clears history", len(mock.call_history) == 0, f"history={len(mock.call_history)}")
except Exception as e:
  check("MockModel reset", False, str(e))


# ── Result ──────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"RESULT: {passed} passed | {failed} failed | {skipped} skipped")
sys.exit(1 if failed else 0)
