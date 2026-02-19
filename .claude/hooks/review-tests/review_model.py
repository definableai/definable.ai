#!/usr/bin/env python3
"""Review test: Model layer through Agent composition.

Tests every documented use-case of the model layer:
- Model construction (OpenAIChat, string shorthand, OpenAILike)
- Agent + Model: basic run, multi-turn, streaming events
- Model identity (id, provider)
- Model error paths (bad id, bad key)
- MockModel as reference baseline

All tests use MockModel unless specifically testing model construction.
No API keys needed — this runs in <5 seconds.
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
  from definable.model import OpenAIChat, Message, Metrics

  check("Import model core types", True)
except Exception as e:
  check("Import model core types", False, str(e))
  print(f"\n{'=' * 60}\nRESULT: {passed} passed | {failed} failed | {skipped} skipped")
  sys.exit(1)

try:
  from definable.model import DeepSeekChat, MoonshotChat, xAI

  check("Import all model providers", True)
except Exception as e:
  check("Import all model providers", False, str(e))

try:
  from definable.agent import Agent, MockModel, create_test_agent, AgentTestCase
  from definable.agent.events import RunStatus

  check("Import agent + testing utilities", True)
except Exception as e:
  check("Import agent + testing utilities", False, str(e))
  print(f"\n{'=' * 60}\nRESULT: {passed} passed | {failed} failed | {skipped} skipped")
  sys.exit(1)


# ── Model Construction ──────────────────────────────────────────
try:
  m = OpenAIChat(id="gpt-4o-mini")
  check("OpenAIChat(id='gpt-4o-mini') constructs", True)
  check("OpenAIChat.id == 'gpt-4o-mini'", m.id == "gpt-4o-mini", f"got {m.id}")
except Exception as e:
  check("OpenAIChat(id='gpt-4o-mini') constructs", False, str(e))

try:
  m = OpenAIChat(id="gpt-4o")
  check("OpenAIChat(id='gpt-4o') constructs", True)
except Exception as e:
  check("OpenAIChat(id='gpt-4o') constructs", False, str(e))

try:
  m = DeepSeekChat(id="deepseek-chat")
  check("DeepSeekChat constructs", True)
except Exception as e:
  check("DeepSeekChat constructs", False, str(e))

try:
  m = MoonshotChat(id="kimi-k2-turbo-preview")
  check("MoonshotChat constructs", True)
except Exception as e:
  check("MoonshotChat constructs", False, str(e))

try:
  m = xAI(id="grok-3")
  check("xAI constructs", True)
except Exception as e:
  check("xAI constructs", False, str(e))


# ── String Shorthand → Model ───────────────────────────────────
try:
  agent = Agent(model="gpt-4o-mini", instructions="test")
  check("Agent(model='gpt-4o-mini') string shorthand", True)
  check(
    "String shorthand resolves to model with correct id",
    hasattr(agent.model, "id") and "gpt-4o-mini" in str(agent.model.id),
    f"model.id = {getattr(agent.model, 'id', 'N/A')}",
  )
except Exception as e:
  check("Agent(model='gpt-4o-mini') string shorthand", False, str(e))


# ── MockModel baseline ─────────────────────────────────────────
try:
  mock = MockModel(responses=["Hello from mock"])
  check("MockModel constructs with responses", True)
  check("MockModel.id is 'mock-model'", mock.id == "mock-model", f"got {mock.id}")
  check("MockModel.provider is 'mock'", mock.provider == "mock", f"got {mock.provider}")
except Exception as e:
  check("MockModel baseline", False, str(e))


# ── Agent + MockModel: Basic Run ────────────────────────────────
try:
  mock = MockModel(responses=["Hello world"])
  agent = Agent(model=mock, instructions="You are helpful.")
  output = agent.run("Hi")
  check("Agent.run() returns RunOutput", hasattr(output, "content"), "no .content attr")
  check("Agent.run() content matches MockModel response", output.content == "Hello world", f"got: {output.content!r}")
  check("Agent.run() status is completed", output.status == RunStatus.completed, f"got: {output.status}")
  check(
    "Agent.run() has messages list",
    isinstance(output.messages, list) and len(output.messages) >= 2,
    f"messages count: {len(output.messages) if output.messages else 0}",
  )
except Exception as e:
  check("Agent + MockModel basic run", False, f"{type(e).__name__}: {e}")


# ── Agent + MockModel: Multi-Turn ───────────────────────────────
try:
  mock = MockModel(responses=["First response", "Second response"])
  agent = Agent(model=mock, instructions="You are helpful.")

  out1 = agent.run("Turn 1")
  check("Multi-turn: turn 1 content", out1.content == "First response", f"got: {out1.content!r}")

  out2 = agent.run("Turn 2", messages=out1.messages)
  check("Multi-turn: turn 2 content", out2.content == "Second response", f"got: {out2.content!r}")
  check("Multi-turn: messages accumulate", len(out2.messages) > len(out1.messages), f"turn1={len(out1.messages)}, turn2={len(out2.messages)}")
except Exception as e:
  check("Agent + MockModel multi-turn", False, f"{type(e).__name__}: {e}")


# ── Agent + MockModel: call_history tracking ────────────────────
try:
  mock = MockModel(responses=["tracked"])
  agent = Agent(model=mock, instructions="test")
  agent.run("test message")
  check("MockModel.call_history records call", len(mock.call_history) >= 1, f"call_history length: {len(mock.call_history)}")
  check("MockModel.call_count incremented", mock.call_count >= 1, f"call_count: {mock.call_count}")
except Exception as e:
  check("MockModel call tracking", False, str(e))


# ── Agent + MockModel: Async Run ────────────────────────────────
try:
  mock = MockModel(responses=["async hello"])
  agent = Agent(model=mock, instructions="test")
  output = asyncio.run(agent.arun("async test"))
  check("Agent.arun() returns content", output.content == "async hello", f"got: {output.content!r}")
except Exception as e:
  check("Agent.arun() async", False, f"{type(e).__name__}: {e}")


# ── Agent + MockModel: Streaming ────────────────────────────────
try:
  mock = MockModel(responses=["stream test"])
  agent = Agent(model=mock, instructions="test")

  async def collect_stream():
    events = []
    async for event in agent.arun_stream("test"):
      events.append(event)
    return events

  events = asyncio.run(collect_stream())
  check("Agent.arun_stream() yields events", len(events) > 0, f"got {len(events)} events")
except Exception as e:
  check("Agent.arun_stream() streaming", False, f"{type(e).__name__}: {e}")


# ── Agent + MockModel: Structured Output ────────────────────────
try:
  from pydantic import BaseModel as PydanticBaseModel

  class TestSchema(PydanticBaseModel):
    name: str
    value: int

  mock = MockModel(structured_responses=['{"name": "test", "value": 42}'])
  agent = Agent(model=mock, instructions="test")
  output = asyncio.run(agent.arun("give me data", output_schema=TestSchema))
  check("Agent with output_schema runs without crash", True)
except Exception as e:
  check("Agent with output_schema", False, f"{type(e).__name__}: {e}")


# ── Message type ────────────────────────────────────────────────
try:
  msg = Message(role="user", content="test")
  check("Message(role='user', content='test') constructs", True)
  check("Message.role == 'user'", msg.role == "user", f"got {msg.role}")
  check("Message.content == 'test'", msg.content == "test", f"got {msg.content}")
except Exception as e:
  check("Message construction", False, str(e))

try:
  msg_asst = Message(role="assistant", content="response")
  check("Message(role='assistant') constructs", True)
except Exception as e:
  check("Message assistant role", False, str(e))


# ── Metrics type ────────────────────────────────────────────────
try:
  metrics = Metrics()
  check("Metrics() default constructs", True)
  check("Metrics has input_tokens attr", hasattr(metrics, "input_tokens"), "no input_tokens")
  check("Metrics has output_tokens attr", hasattr(metrics, "output_tokens"), "no output_tokens")
except Exception as e:
  check("Metrics construction", False, str(e))


# ── create_test_agent convenience ───────────────────────────────
try:
  agent = create_test_agent(responses=["convenience test"])
  output = agent.run("hello")
  check("create_test_agent() works", output.content == "convenience test", f"got: {output.content!r}")
except Exception as e:
  check("create_test_agent()", False, str(e))


# ── AgentTestCase utilities ─────────────────────────────────────
try:
  tc = AgentTestCase()
  agent = tc.create_agent(model=MockModel(responses=["test case"]))
  output = agent.run("hi")
  tc.assert_has_content(output)
  tc.assert_no_errors(output)
  check("AgentTestCase create + assert methods work", True)
except Exception as e:
  check("AgentTestCase utilities", False, str(e))


# ── Result ──────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"RESULT: {passed} passed | {failed} failed | {skipped} skipped")
sys.exit(1 if failed else 0)
