#!/usr/bin/env python3
"""Review test: Guardrails through Agent composition.

Tests built-in guardrails, composability, and Agent + Guardrails integration.
"""

import sys, os

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


try:
  from definable.agent.guardrail import (
    Guardrails,
    GuardrailResult,
    max_tokens,
    block_topics,
    pii_filter,
    regex_filter,
    tool_allowlist,
    tool_blocklist,
    max_output_tokens,
    ALL,
    ANY,
    NOT,
    input_guardrail,
  )
  from definable.agent import Agent, MockModel

  check("Import guardrails + agent", True)
except Exception as e:
  check("Import guardrails + agent", False, str(e))
  print(f"\n{'=' * 60}\nRESULT: {passed} passed | {failed} failed | 0 skipped")
  sys.exit(1)


# ── Built-in guardrail construction ────────────────────────────
try:
  g = max_tokens(100)
  check("max_tokens(100) constructs", g is not None, "returned None")
except Exception as e:
  check("max_tokens(100)", False, str(e))

try:
  g = block_topics(["violence", "weapons"])
  check("block_topics([...]) constructs", g is not None, "returned None")
except Exception as e:
  check("block_topics()", False, str(e))

try:
  g = pii_filter()
  check("pii_filter() constructs", g is not None, "returned None")
except Exception as e:
  check("pii_filter()", False, str(e))

try:
  g = regex_filter(r"password|secret")
  check("regex_filter(pattern) constructs", g is not None, "returned None")
except Exception as e:
  check("regex_filter()", False, str(e))

try:
  g = tool_allowlist({"calculator"})
  check("tool_allowlist(set) constructs", g is not None, "returned None")
except Exception as e:
  check("tool_allowlist()", False, str(e))

try:
  g = tool_blocklist({"shell"})
  check("tool_blocklist(set) constructs", g is not None, "returned None")
except Exception as e:
  check("tool_blocklist()", False, str(e))

try:
  g = max_output_tokens(500)
  check("max_output_tokens(500) constructs", g is not None, "returned None")
except Exception as e:
  check("max_output_tokens()", False, str(e))


# ── Guardrails container ───────────────────────────────────────
try:
  gs = Guardrails(input=[max_tokens(100)])
  check("Guardrails(input=[...]) constructs", True)
except Exception as e:
  check("Guardrails(input=)", False, str(e))

try:
  gs = Guardrails(input=[max_tokens(100)], output=[pii_filter()], tool=[tool_blocklist({"shell"})])
  check("Guardrails(input, output, tool) all three", True)
except Exception as e:
  check("Guardrails all three", False, str(e))


# ── Composable operators ───────────────────────────────────────
try:
  g = ALL(max_tokens(100), block_topics(["test"]))
  check("ALL() composable AND", g is not None, "returned None")
except Exception as e:
  check("ALL()", False, str(e))

try:
  g = ANY(max_tokens(100), block_topics(["test"]))
  check("ANY() composable OR", g is not None, "returned None")
except Exception as e:
  check("ANY()", False, str(e))

try:
  g = NOT(max_tokens(10))
  check("NOT() composable invert", g is not None, "returned None")
except Exception as e:
  check("NOT()", False, str(e))


# ── Decorators ──────────────────────────────────────────────────
try:

  @input_guardrail
  async def custom_input_guard(text, context=None):
    if "blocked" in text:
      return GuardrailResult(allowed=False, message="Blocked word found")
    return GuardrailResult(allowed=True)

  check("@input_guardrail decorator", custom_input_guard is not None, "returned None")
except Exception as e:
  check("@input_guardrail", False, str(e))


# ── Agent + Guardrails ──────────────────────────────────────────
try:
  agent = Agent(model=MockModel(), guardrails=Guardrails(input=[max_tokens(500)]))
  check("Agent(guardrails=...) constructs", True)
except Exception as e:
  check("Agent + guardrails construction", False, str(e))

try:
  agent = Agent(model=MockModel(responses=["ok"]), guardrails=Guardrails(input=[max_tokens(5000)]))
  output = agent.run("short message")
  check("Agent + guardrails: short msg passes", output.content is not None, "no content")
except Exception as e:
  check("Agent + guardrails run", False, str(e))


print(f"\n{'=' * 60}")
print(f"RESULT: {passed} passed | {failed} failed | 0 skipped")
sys.exit(1 if failed else 0)
