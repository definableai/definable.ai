---
description: >
  Stability evaluation: every test is a developer use-case with Agent.
  Writes eval scripts to .workspace/evals/, runs each, reads stdout.
  Files GitHub issues for confirmed bugs. Reports to .claude/reports/.
  Zero user interaction. Never asks questions.
---

# /evaluate — Stability Evaluation Pipeline

**ZERO USER INTERACTION. All decisions autonomous. Run everything, report everything.**

## Pre-flight

```bash
cd "$(git rev-parse --show-toplevel)"

# Check credentials
if [ ! -f .env.test ]; then
  echo "⚠️  No .env.test found. Running MockModel-only evals."
fi

# Clean workspace
rm -rf .workspace/evals 2>/dev/null
mkdir -p .workspace/evals .workspace/evals/fixtures .claude/reports .claude/memory

# Environment
source .env.test 2>/dev/null || true
source .venv/bin/activate 2>/dev/null || true

# Install
pip install -e ".[readers,serve,jwt,cron,runtime]" 2>&1 | tail -3
```

## Pipeline — Sequential, No Pausing

### Step 1: Load Context
- Read `CLAUDE.md` — API surface, constructor signatures, import paths
- Read `.claude/memory/` — known issues, eval history, user preferences
- Determine available API keys from environment

### Step 2: Read Library Source (MANDATORY)
Before writing any eval, read the actual source for every module you'll test:
- `definable/definable/agent/__init__.py` and `agent/agent.py` (constructor, run, arun)
- `definable/definable/tool/__init__.py` and `tool/decorator.py`
- `definable/definable/skill/__init__.py` and at least one builtin
- `definable/definable/knowledge/__init__.py` and `knowledge/base.py`
- `definable/definable/memory/__init__.py`
- `definable/definable/vectordb/__init__.py`
- `definable/definable/mcp/__init__.py`
- `definable/definable/agent/guardrail/__init__.py`
- `definable/definable/agent/tracing/__init__.py`
- `definable/definable/agent/testing.py`
- `definable/definable/agent/events.py` (RunOutput, RunStatus)
- At least 3 examples from `definable/examples/`

**Do NOT write eval scripts from memory. Use the actual exports and signatures.**

### Step 3: Write ALL Eval Scripts to `.workspace/evals/`

Follow the **defineable-evaluator** agent's use-case matrix. Write these scripts in order:

| # | Script | Use Case | Needs API Key? |
|---|--------|----------|----------------|
| 00 | `eval_00_foundation.py` | Imports + circular dependency check | No |
| 01 | `eval_01_bare_agent.py` | Agent + MockModel basics | No |
| 02 | `eval_02_agent_tools.py` | Agent + @tool (customer support) | Partial |
| 03 | `eval_03_agent_skills.py` | Agent + Skills (data analyst) | Partial |
| 04 | `eval_04_agent_knowledge.py` | Agent + Knowledge RAG (HR assistant) | Partial |
| 05 | `eval_05_agent_memory.py` | Agent + Memory (personal assistant) | Partial |
| 06 | `eval_06_agent_guardrails.py` | Agent + Guardrails (safety) | Partial |
| 07 | `eval_07_agent_observability.py` | Agent + Middleware + Tracing | No |
| 08 | `eval_08_tools_and_knowledge.py` | Agent + Tools + Knowledge (tech support) | Yes |
| 09 | `eval_09_tools_and_memory.py` | Agent + Tools + Memory (PA) | Yes |
| 10 | `eval_10_knowledge_and_memory.py` | Agent + Knowledge + Memory (HR onboarding) | Yes |
| 11 | `eval_11_guardrails_and_tools.py` | Agent + Guardrails + Tools (security) | Yes |
| 12 | `eval_12_agent_mcp.py` | Agent + MCP (filesystem server) | Partial (npx) |
| 13 | `eval_13_full_stack.py` | All systems wired together | Yes |
| 14 | `eval_14_multi_turn_stress.py` | 10-turn conversation stability | Yes |
| 15 | `eval_15_error_handling.py` | Bad inputs, error messages | Partial |

**"Partial"** = some tests use MockModel (always run), some need OPENAI_API_KEY (skip if missing).

**Every script MUST:**
- Print `✅ PASS: <description>` for each passing check
- Print `❌ FAIL: <description> — <error detail>` for each failure
- Print `⚠️  SKIP: <description> — <reason>` for skipped checks
- Print final summary: `RESULT: X passed | Y failed | Z skipped`
- Exit 0 if no failures, exit 1 if any failures
- Clean up temp files (db files, trace dirs) in a finally block
- Be runnable standalone: `python .workspace/evals/eval_02_agent_tools.py`

### Step 4: Execute ALL Scripts — Read Stdout

**This is the critical step. You MUST run each script and read the output.**

```bash
cd "$(git rev-parse --show-toplevel)"
source .env.test 2>/dev/null || true
source .venv/bin/activate 2>/dev/null || true

for script in $(ls .workspace/evals/eval_*.py | sort); do
    echo ""
    echo "╔══════════════════════════════════════════════════╗"
    echo "║ $(basename $script)"
    echo "╚══════════════════════════════════════════════════╝"
    timeout 180 python "$script" 2>&1
    echo "Exit: $?"
done
```

After running each script:
1. **Read** the stdout — count ✅ / ❌ / ⚠️
2. For each ❌: is this a library bug or a test bug?
3. If a script crashes (exit code != 0 and no ✅/❌ output): the crash itself is the finding
4. **Re-run failed scripts once** to confirm. If still fails → confirmed bug.

### Step 5: File Issues
For each confirmed bug:
1. Check `.claude/memory/known-issues.md` — skip duplicates
2. Check GitHub: `gh issue list --search "<keywords>" --limit 5`
3. New → `gh issue create --title "..." --body "..." --label bug,evaluator-found`
4. `gh` unavailable → `.claude/reports/unfiled-issues.md`
5. Update `.claude/memory/known-issues.md`
6. Max 15 issues per run

### Step 6: Generate Report
Save to `.claude/reports/eval-<YYYY-MM-DD>-<HHMMSS>.md`
Print summary table to stdout with pass/fail counts and stability score.

### Step 7: Save Memory — MANDATORY
Update all 5 memory files in `.claude/memory/`:
- `evaluation-history.md` (APPEND)
- `known-issues.md` (MERGE)
- `project-profile.md` (OVERWRITE)
- `credentials.md` (OVERWRITE)
- `user-preferences.md` (PRESERVE)

Verify all 5 exist: `ls -la .claude/memory/`

## Quick Reference

| What | Where |
|------|-------|
| Eval scripts | `.workspace/evals/eval_*.py` |
| Reports | `.claude/reports/eval-<timestamp>.md` |
| Unfiled issues | `.claude/reports/unfiled-issues.md` |
| Memory | `.claude/memory/*.md` |
| Library source | `definable/definable/` (READ ONLY) |
| Examples | `definable/examples/` (READ ONLY) |
