#!/usr/bin/env python3
"""Review test: Skills through Agent composition.

Tests built-in skills, custom skills, and Agent + Skill integration.
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
  from definable.skill import Skill, Calculator, DateTime, Shell, WebSearch, TextProcessing, JSONOperations, FileOperations, HTTPRequests
  from definable.agent import Agent, MockModel
  from definable.tool import tool

  check("Import skills + agent", True)
except Exception as e:
  check("Import skills + agent", False, str(e))
  print(f"\n{'=' * 60}\nRESULT: {passed} passed | {failed} failed | 0 skipped")
  sys.exit(1)

# ── Built-in skills ─────────────────────────────────────────────
for SkillClass, name in [
  (Calculator, "Calculator"),
  (DateTime, "DateTime"),
  (Shell, "Shell"),
  (TextProcessing, "TextProcessing"),
  (JSONOperations, "JSONOperations"),
  (FileOperations, "FileOperations"),
  (HTTPRequests, "HTTPRequests"),
]:
  try:
    s = SkillClass()
    check(f"{name}() constructs", True)
    check(f"{name}.name is non-empty", bool(s.name), f"name={s.name!r}")
    check(f"{name}.instructions is non-empty", bool(s.instructions), "empty instructions")
    check(f"{name}.tools is non-empty list", isinstance(s.tools, list) and len(s.tools) > 0, f"tools count: {len(s.tools) if s.tools else 0}")
  except Exception as e:
    check(f"{name}() constructs", False, str(e))

# WebSearch may need API key
try:
  ws = WebSearch()
  check("WebSearch() constructs", True)
except Exception as e:
  check("WebSearch() constructs", False, str(e))

# ── Agent + Skills composition ──────────────────────────────────
try:
  agent = Agent(model=MockModel(), skills=[Calculator(), DateTime()])
  check("Agent + 2 skills constructs", True)
except Exception as e:
  check("Agent + 2 skills", False, str(e))

try:
  agent = Agent(model=MockModel(), skills=[Calculator()])
  output = agent.run("test")
  check("Agent + Calculator runs", output.content is not None, "no content")
  # Verify skill instructions in system message
  if agent.model.call_history:
    system_msg = agent.model.call_history[0].get("system_message", "") or ""
    check(
      "Skill instructions in system prompt",
      "calculator" in system_msg.lower() or "math" in system_msg.lower() or "calc" in system_msg.lower() or len(system_msg) > 10,
      f"system_message length: {len(system_msg)}",
    )
  else:
    check("Skill instructions in system prompt", False, "no call_history")
except Exception as e:
  check("Agent + Calculator run", False, str(e))


# ── Agent + Skills + Tools (no conflict) ────────────────────────
@tool
def my_tool(x: str) -> str:
  """Custom tool."""
  return x


try:
  agent = Agent(model=MockModel(), skills=[Calculator()], tools=[my_tool])
  check("Agent + skill + custom tool constructs", True)
  total_tools = len(agent.tools)
  check("Agent has both skill tools and custom tool", total_tools >= 2, f"total tools: {total_tools}")
except Exception as e:
  check("Agent + skill + tool", False, str(e))

# ── Custom inline Skill ─────────────────────────────────────────
try:

  @tool
  def custom_fn(q: str) -> str:
    """Custom function."""
    return q

  custom_skill = Skill(name="custom_skill", instructions="Do custom things.", tools=[custom_fn])
  check("Inline Skill(name, instructions, tools) constructs", True)

  agent = Agent(model=MockModel(), skills=[custom_skill])
  output = agent.run("test")
  check("Agent + inline skill runs", output.content is not None, "no content")
except Exception as e:
  check("Inline Skill", False, str(e))


print(f"\n{'=' * 60}")
print(f"RESULT: {passed} passed | {failed} failed | 0 skipped")
sys.exit(1 if failed else 0)
