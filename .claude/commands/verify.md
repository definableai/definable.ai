---
description: "Run verification on changed files: ruff, mypy, imports, examples, scenarios."
allowed-tools: Bash(.venv/bin/*), Bash(git diff*), Bash(git status*), Bash(timeout *), Bash(source .env.test*), Read(**), Grep(**), Glob(**)
---

# Verify Changes

Run full verification on modified files. Fixes issues inline if possible.

## Current State

Changed files:
!`git diff --name-only HEAD | grep '\.py$'`

Untracked .py files:
!`git ls-files --others --exclude-standard | grep '\.py$' || echo "(none)"`

## Instructions

For the argument "$ARGUMENTS":
- If empty: verify all changed .py files
- If a module name (e.g. "agents", "models"): verify only that module
- If "--full": run ALL examples across ALL modules

### Step 1: Ruff Format + Lint
Run `.venv/bin/ruff format` and `.venv/bin/ruff check --fix` on all changed library files under `definable/definable/`. Fix any remaining issues.

### Step 2: Mypy
Run `.venv/bin/python -m mypy --ignore-missing-imports` on changed library files. Report issues but don't block.

### Step 3: Import Smoke Test
Run this to verify all public symbols import cleanly:
```
.venv/bin/python -c "
from definable.agents import Agent
from definable.models.openai import OpenAIChat
from definable.tools.decorator import tool
from definable.knowledge import Document, Knowledge
from definable.vectordbs import InMemoryVectorDB
print('All core imports OK')
"
```

### Step 4: Run Examples
Source `.env.test` first, then run examples for each changed module:

| Module | Examples |
|--------|----------|
| agents/ | examples/agents/01_simple_agent.py, 02_agent_with_tools.py |
| models/ | examples/models/01_basic_invoke.py, 04_structured_output.py |
| tools/ | examples/tools/01_basic_tool.py, 02_tool_parameters.py |
| knowledge/ | examples/knowledge/01_basic_rag.py, 06_agent_with_knowledge.py |
| memory/ | examples/memory/01_basic_memory.py |
| guardrails/ | examples/guardrails/01_basic_guardrails.py |
| skills/ | examples/skills/01_skills_and_registry.py |
| vectordbs/ | examples/knowledge/05_vector_databases.py |

Use `timeout 45` for each example. Skip if the required API key is missing.

### Step 5: Run Scenarios
Run all scenario scripts in `.claude/hooks/scenarios/`:
```
.venv/bin/python .claude/hooks/scenarios/scenario_knowledge_pipeline.py
.venv/bin/python .claude/hooks/scenarios/scenario_guardrails_and_middleware.py
```
For scenarios requiring OPENAI_API_KEY, only run if the key is set.

### Step 6: Summary
Print a clear summary:
```
=== Verification Summary ===
Ruff Format:  PASS/FAIL
Ruff Lint:    PASS/FAIL
Mypy:         PASS/WARN (advisory)
Imports:      PASS/FAIL
Examples:     X/Y passed
Scenarios:    X/Y passed
```
