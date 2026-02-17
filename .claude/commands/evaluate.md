---
description: >
  Full autonomous evaluation of the Definable library.
  Stress-tests every module with adversarial inputs and real-world patterns.
  Files GitHub issues for bugs. Saves report to .claude/reports/.
  Test scripts go in .workspace/ (gitignored, never in the library).
  Zero user interaction. Never asks questions.
---

# /evaluate — Autonomous Evaluation Pipeline

**ZERO USER INTERACTION. Never ask. Never pause. All decisions autonomous.**

## Pre-flight (Silent — no output unless fatal)

```bash
cd "$(git rev-parse --show-toplevel)"

# 1. Check credentials exist
if [ ! -f .env.test ] && [ ! -f .claude/memory/credentials.md ]; then
  echo "❌ Run /setup first — no credentials found."
  exit 1
fi

# 2. Clean workspace, create dirs
rm -rf .workspace/* 2>/dev/null
mkdir -p .workspace .workspace/test_files .claude/reports .claude/memory

# 3. Source environment
source .env.test 2>/dev/null || true
source .venv/bin/activate 2>/dev/null || true

# 4. Check gh auth (silent — fallback if not available)
gh auth status 2>/dev/null
# If fails, issues go to .claude/reports/unfiled-issues.md — don't stop

# 5. Install library with all extras
pip install -e ".[readers,serve,jwt,cron,runtime]" 2>&1 | tail -5
# If fails → P0 blocker. File issue if possible, STOP.
```

## Pipeline — Execute ALL steps sequentially without pausing

### Step 1: Load Memory + Context
- Read `.claude/CLAUDE.md` — exact API surface (constructor signatures, exports)
- Read `.claude/memory/` — known issues, project profile, eval history, preferences
- Determine which API keys are available from `.env.test`

### Step 2: Static Analysis (always runs, no API keys needed)
1. **Import integrity** — every public symbol from every `__init__.py`
2. **Circular import detection** — subprocess isolation per module
3. **mypy** — `python -m mypy definable/definable/ --ignore-missing-imports 2>&1 | head -100`
4. **ruff** — `python -m ruff check definable/definable/ 2>&1 | head -100`

### Step 3: Write ALL Test Scripts to .workspace/
Follow the **defineable-evaluator** agent's 12-tier test matrix. Write scripts for:

**Always (no API key needed) — 14 scripts:**
1. `eval_imports.py` — all modules, all public symbols from `__all__`
2. `eval_circular.py` — subprocess isolation per module
3. `eval_agent_construction.py` — Agent() with every edge case input
4. `eval_agent_run_mock.py` — Agent.run() with MockModel, multi-turn, bad inputs
5. `eval_agent_middleware.py` — middleware chain, ordering, error handling
6. `eval_tools.py` — @tool decorator, schemas, edge cases, agent integration
7. `eval_skills.py` — all 8 built-in skills + custom skill creation + security
8. `eval_knowledge.py` — InMemoryVectorDB, Document, chunkers, pipeline
9. `eval_memory.py` — InMemoryStore, SQLiteMemoryStore, CognitiveMemory
10. `eval_guardrails.py` — all built-ins + composability + decorators + agent integration
11. `eval_readers.py` — file parsers with test files
12. `eval_auth.py` — JWT, API key, allowlist, composite
13. `eval_testing.py` — MockModel, create_test_agent, AgentTestCase
14. `eval_dx.py` — error message quality audit (score 1-5 per scenario)

**If OPENAI_API_KEY available — 3 scripts:**
15. `eval_agent_run_real.py` — real model calls, metrics, error handling
16. `eval_agent_streaming.py` — streaming events, ordering, completeness
17. `eval_realworld.py` — calculator agent, multi-turn chat, tool+skill combo, concurrent runs

### Step 4: Execute All Scripts
```bash
cd /Users/hash/work/definable.ai
source .env.test 2>/dev/null || true
source .venv/bin/activate 2>/dev/null || true

# Run each with timeout, capture output
for script in .workspace/eval_*.py; do
  echo "=== Running $(basename $script) ==="
  timeout 120 python "$script" 2>&1
  echo "Exit code: $?"
  echo "==========================="
done
```

- Run no-API-key scripts first
- Run API-key scripts if keys available
- Failed scripts → re-run 2 more times for flaky detection
- Run existing e2e tests (same marker filter as CI):
  ```bash
  python -m pytest definable/tests_e2e/ \
    -m "not openai and not deepseek and not moonshot and not xai and not telegram and not discord and not signal and not postgres and not redis and not qdrant and not chroma and not mongodb and not pinecone and not mistral and not mem0" \
    -v --tb=short --timeout=120 2>&1 || true
  ```

### Step 5: File Issues
For each confirmed bug:
1. Check `.claude/memory/known-issues.md` — skip if title matches existing
2. Check GitHub: `gh issue list --search "<keywords>" --limit 10`
3. New bug → `gh issue create --title "..." --body "..." --label bug,evaluator-found,<component>`
4. `gh` unavailable → append to `.claude/reports/unfiled-issues.md`
5. Update `.claude/memory/known-issues.md` immediately
6. Max 20 issues per run (configurable in user-preferences.md)

### Step 6: Generate Report
Save to `.claude/reports/eval-<YYYY-MM-DD>-<HHMMSS>.md`
Print compact summary to stdout with scores, pass/fail counts, issues filed.

### Step 7: Save Memory — MANDATORY
Write all 5 memory files. For append-only files (evaluation-history.md, known-issues.md): READ existing → MERGE → WRITE.

Verify: `ls -la .claude/memory/` — all 5 files must exist.

## Quick Reference

| What | Where |
|------|-------|
| Test scripts | `.workspace/eval_*.py` |
| Test fixture files | `.workspace/test_files/` |
| Report | `.claude/reports/eval-<timestamp>.md` |
| Unfiled issues | `.claude/reports/unfiled-issues.md` |
| Memory | `.claude/memory/*.md` |
| Library source | `definable/definable/` (READ ONLY!) |
| Existing examples | `definable/examples/` (READ ONLY!) |
| Existing e2e tests | `definable/tests_e2e/` (READ ONLY!) |
