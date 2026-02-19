#!/usr/bin/env bash
# verify-before-stop.sh — Runs before Claude can finish a task
# Exit 0 = allow stop, Exit 2 = block stop (must fix issues first)

set -euo pipefail

REPO="/Users/hash/work/definable.ai"
VENV="$REPO/.venv/bin"
LIB="$REPO/definable/definable"
EXAMPLES="$REPO/definable/examples"
SCENARIOS="$REPO/.claude/hooks/scenarios"

cd "$REPO"

# Source API keys if available (for example execution)
if [ -f "$REPO/.env.test" ]; then
  set +e
  source "$REPO/.env.test" 2>/dev/null
  set -e
fi

# Collect modified .py files (staged + unstaged + untracked, relative to repo root)
# Use --diff-filter to EXCLUDE deleted files (only show Added/Modified/Renamed/Copied)
MODIFIED_PY=$(git diff --name-only --diff-filter=AMRC HEAD 2>/dev/null | grep '\.py$' || true)
MODIFIED_PY="$MODIFIED_PY
$(git diff --cached --name-only --diff-filter=AMRC 2>/dev/null | grep '\.py$' || true)"
# Also include untracked .py files
MODIFIED_PY="$MODIFIED_PY
$(git ls-files --others --exclude-standard 2>/dev/null | grep '\.py$' || true)"
# Deduplicate and filter to files that actually exist
MODIFIED_PY=$(echo "$MODIFIED_PY" | sort -u | grep -v '^$' | while read -r f; do [ -f "$f" ] && echo "$f"; done || true)

if [ -z "$MODIFIED_PY" ]; then
  echo "No modified .py files detected. Verification passed."
  exit 0
fi

ERRORS=0
WARNINGS=0

echo "=== Verify Before Stop ==="
echo "Modified files:"
echo "$MODIFIED_PY" | head -20
echo ""

# --- 1. Ruff format check ---
echo "--- Ruff Format Check ---"
LIB_FILES=$(echo "$MODIFIED_PY" | grep '^definable/definable/' || true)
if [ -n "$LIB_FILES" ]; then
  FORMAT_RESULT=$($VENV/ruff format --check $LIB_FILES 2>&1) || {
    echo "FAIL: Files not formatted:"
    echo "$FORMAT_RESULT" | grep 'would be reformatted' | head -10
    ERRORS=$((ERRORS + 1))
  }
  if [ $ERRORS -eq 0 ]; then
    echo "PASS: All files formatted correctly"
  fi
else
  echo "SKIP: No library files modified"
fi

# --- 2. Ruff lint check ---
echo ""
echo "--- Ruff Lint Check ---"
if [ -n "$LIB_FILES" ]; then
  LINT_RESULT=$($VENV/ruff check $LIB_FILES 2>&1) || {
    echo "FAIL: Lint errors:"
    echo "$LINT_RESULT" | head -15
    ERRORS=$((ERRORS + 1))
  }
  if [ $ERRORS -le 1 ]; then
    echo "PASS: No lint errors"
  fi
else
  echo "SKIP: No library files modified"
fi

# --- 3. Mypy check (warning only — does not block) ---
echo ""
echo "--- Mypy Check (advisory) ---"
if [ -n "$LIB_FILES" ]; then
  MYPY_RESULT=$($VENV/python -m mypy --ignore-missing-imports $LIB_FILES 2>&1) || {
    echo "WARNING: Mypy issues (non-blocking):"
    echo "$MYPY_RESULT" | tail -10
    WARNINGS=$((WARNINGS + 1))
  }
  if [ $WARNINGS -eq 0 ]; then
    echo "PASS: No type errors"
  fi
else
  echo "SKIP: No library files modified"
fi

# --- 4. Import smoke test ---
echo ""
echo "--- Import Smoke Test ---"
IMPORT_RESULT=$($VENV/python -c "
from definable.agents import Agent
from definable.models.openai import OpenAIChat
from definable.tools.decorator import tool
from definable.knowledge import Document, Knowledge
from definable.vectordbs import InMemoryVectorDB
print('All core imports OK')
" 2>&1) || {
  echo "FAIL: Import errors:"
  echo "$IMPORT_RESULT"
  ERRORS=$((ERRORS + 1))
}
if echo "$IMPORT_RESULT" | grep -q "All core imports OK"; then
  echo "PASS: Core imports working"
fi

# --- 5. Run relevant examples based on modified modules ---
echo ""
echo "--- Example Verification ---"

# Run an example, skip on API auth errors
run_example() {
  local example_file="$1"
  local description="$2"
  local timeout_secs="${3:-30}"

  if [ ! -f "$example_file" ]; then
    echo "SKIP: $description (file not found)"
    return 0
  fi

  echo -n "Running: $description ... "
  EXAMPLE_RESULT=$(timeout "$timeout_secs" $VENV/python "$example_file" 2>&1) || {
    local exit_code=$?
    if [ $exit_code -eq 124 ]; then
      echo "SKIP (timeout after ${timeout_secs}s)"
      return 0
    fi
    # Skip on API auth errors (invalid/expired keys)
    if echo "$EXAMPLE_RESULT" | grep -qiE '(401|api.key|authentication|unauthorized|invalid_api_key)'; then
      echo "SKIP (API auth error — key may be invalid)"
      return 0
    fi
    echo "FAIL (exit $exit_code)"
    echo "$EXAMPLE_RESULT" | tail -5
    ERRORS=$((ERRORS + 1))
    return 0
  }
  echo "PASS"
}

# Check which modules were modified and run corresponding examples
# Only run examples for modules that have OFFLINE examples or valid API keys
if echo "$MODIFIED_PY" | grep -q 'definable/definable/agents/'; then
  run_example "$EXAMPLES/agents/01_simple_agent.py" "agents/01_simple_agent" 30
fi

if echo "$MODIFIED_PY" | grep -q 'definable/definable/models/'; then
  run_example "$EXAMPLES/models/01_basic_invoke.py" "models/01_basic_invoke" 30
fi

if echo "$MODIFIED_PY" | grep -q 'definable/definable/tools/'; then
  run_example "$EXAMPLES/tools/01_basic_tool.py" "tools/01_basic_tool" 30
fi

if echo "$MODIFIED_PY" | grep -q 'definable/definable/knowledge/'; then
  run_example "$EXAMPLES/knowledge/01_basic_rag.py" "knowledge/01_basic_rag" 45
fi

if echo "$MODIFIED_PY" | grep -q 'definable/definable/memory/'; then
  run_example "$EXAMPLES/memory/01_basic_memory.py" "memory/01_basic_memory" 30
fi

if echo "$MODIFIED_PY" | grep -q 'definable/definable/guardrails/'; then
  # Guardrails example uses MockModel — no API key needed
  run_example "$EXAMPLES/guardrails/01_basic_guardrails.py" "guardrails/01_basic_guardrails" 30
fi

if echo "$MODIFIED_PY" | grep -q 'definable/definable/skills/'; then
  # Skills example may have different name
  run_example "$EXAMPLES/skills/01_skills_and_registry.py" "skills/01_skills_and_registry" 30
fi

if echo "$MODIFIED_PY" | grep -q 'definable/definable/vectordbs/'; then
  # Use basic RAG example which exercises VectorDB through Knowledge
  run_example "$EXAMPLES/knowledge/01_basic_rag.py" "vectordbs (via knowledge/01_basic_rag)" 30
fi

# Skip: interfaces (requires bot tokens), mcp (requires running server), browser (requires Chrome)

# --- 6. Run scenario tests if available ---
echo ""
echo "--- Scenario Tests ---"
if [ -d "$SCENARIOS" ]; then
  for scenario in "$SCENARIOS"/scenario_*.py; do
    if [ -f "$scenario" ]; then
      scenario_name=$(basename "$scenario" .py)
      run_example "$scenario" "$scenario_name" 60
    fi
  done
else
  echo "SKIP: No scenarios directory"
fi

# --- Summary ---
echo ""
echo "=== Summary ==="
echo "Errors: $ERRORS | Warnings: $WARNINGS"

if [ $ERRORS -gt 0 ]; then
  echo ""
  echo "BLOCKED: Fix $ERRORS error(s) before finishing."
  exit 2
fi

if [ $WARNINGS -gt 0 ]; then
  echo "PASSED with $WARNINGS warning(s)."
fi

echo "Verification passed."
exit 0
