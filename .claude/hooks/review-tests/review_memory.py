#!/usr/bin/env python3
"""Review test: Memory through Agent composition.

Tests Memory, stores (InMemory, SQLite), and Agent + Memory integration.
"""

import sys, os, tempfile

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


_cleanup = []

try:
  from definable.memory import Memory, InMemoryStore, SQLiteStore
  from definable.agent import Agent, MockModel

  check("Import memory + agent", True)
except Exception as e:
  check("Import memory + agent", False, str(e))
  print(f"\n{'=' * 60}\nRESULT: {passed} passed | {failed} failed | 0 skipped")
  sys.exit(1)


# ── InMemoryStore ───────────────────────────────────────────────
try:
  store = InMemoryStore()
  check("InMemoryStore() constructs", True)
except Exception as e:
  check("InMemoryStore()", False, str(e))


# ── SQLiteStore ─────────────────────────────────────────────────
try:
  db_path = os.path.join(tempfile.gettempdir(), "definable_review_memory.db")
  _cleanup.append(db_path)
  store = SQLiteStore(db_path)
  check("SQLiteStore(path) constructs", True)
except Exception as e:
  check("SQLiteStore(path)", False, str(e))


# ── Memory with InMemoryStore ──────────────────────────────────
try:
  memory = Memory(store=InMemoryStore())
  check("Memory(store=InMemoryStore()) constructs", True)
except Exception as e:
  check("Memory construction", False, str(e))


# ── Memory with SQLiteStore ────────────────────────────────────
try:
  db_path2 = os.path.join(tempfile.gettempdir(), "definable_review_memory2.db")
  _cleanup.append(db_path2)
  memory = Memory(store=SQLiteStore(db_path2))
  check("Memory(store=SQLiteStore()) constructs", True)
except Exception as e:
  check("Memory + SQLiteStore", False, str(e))


# ── Agent + Memory ──────────────────────────────────────────────
try:
  agent = Agent(model=MockModel(), memory=Memory(store=InMemoryStore()))
  check("Agent(memory=Memory(...)) constructs", True)
except Exception as e:
  check("Agent + Memory construction", False, str(e))

try:
  agent = Agent(model=MockModel(), memory=Memory(store=InMemoryStore()))
  output = agent.run("My name is Alice")
  check("Agent + Memory runs", output.content is not None, "no content")
  check("Agent + Memory completes", hasattr(output, "status"), "no status")
except Exception as e:
  check("Agent + Memory run", False, str(e))


# ── Agent memory=True shorthand ─────────────────────────────────
try:
  agent = Agent(model=MockModel(), memory=True)
  check("Agent(memory=True) shorthand", True)
  output = agent.run("test memory shorthand")
  check("Agent(memory=True) runs", output.content is not None, "no content")
except Exception as e:
  check("Agent memory=True", False, str(e))


# ── Agent memory=False (default) ────────────────────────────────
try:
  agent = Agent(model=MockModel(), memory=False)
  check("Agent(memory=False) constructs", True)
except Exception as e:
  check("Agent memory=False", False, str(e))


# ── Cleanup ─────────────────────────────────────────────────────
for path in _cleanup:
  try:
    if os.path.exists(path):
      os.remove(path)
  except OSError:
    pass

print(f"\n{'=' * 60}")
print(f"RESULT: {passed} passed | {failed} failed | 0 skipped")
sys.exit(1 if failed else 0)
