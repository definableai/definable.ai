#!/usr/bin/env bash
# post-task-review.sh — Code review hook for Claude Code Stop event.
#
# Runs automatically when Claude Code finishes a task.
# Detects which library modules changed, runs targeted review tests,
# and feeds results back to Claude Code.
#
# Returns JSON:
#   { "decision": "approve" }           — all good, Claude stops
#   { "decision": "block", "reason": "..." } — issues found, Claude continues
#
# All review tests use MockModel — no API keys, runs in seconds.

set -euo pipefail

# Read stdin (hook input JSON) — we don't need it but must consume it
cat > /dev/null

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
HOOK_DIR="$PROJECT_DIR/.claude/hooks"
REVIEW_DIR="$HOOK_DIR/review-tests"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"

# Fallback to system python if venv not found
if [ ! -x "$VENV_PYTHON" ]; then
  VENV_PYTHON="python3"
fi

# ── 1. Detect changed files ────────────────────────────────────────
# Check both staged and unstaged changes against HEAD
CHANGED_FILES=$(cd "$PROJECT_DIR" && git diff --name-only HEAD 2>/dev/null || true)
STAGED_FILES=$(cd "$PROJECT_DIR" && git diff --name-only --cached 2>/dev/null || true)
ALL_CHANGED=$(echo -e "${CHANGED_FILES}\n${STAGED_FILES}" | sort -u | grep -v '^$' || true)

# No changes? Nothing to review.
if [ -z "$ALL_CHANGED" ]; then
  echo '{"decision": "approve"}'
  exit 0
fi

# Only non-Python library files changed? Skip review.
LIB_PYTHON_CHANGES=$(echo "$ALL_CHANGED" | grep '^definable/definable/.*\.py$' || true)
if [ -z "$LIB_PYTHON_CHANGES" ]; then
  echo '{"decision": "approve"}'
  exit 0
fi

# ── 2. Map changed files → modules ─────────────────────────────────
MODULES_TO_TEST=()

map_file_to_module() {
  local file="$1"
  # Strip the definable/definable/ prefix
  local rel="${file#definable/definable/}"

  case "$rel" in
    model/*)        echo "model" ;;
    agent/guardrail/*) echo "guardrail" ;;
    agent/tracing/*) echo "agent" ;;
    agent/middleware*) echo "agent" ;;
    agent/reasoning/*) echo "agent" ;;
    agent/research/*) echo "agent" ;;
    agent/replay/*) echo "agent" ;;
    agent/auth/*) echo "agent" ;;
    agent/testing*) echo "agent" ;;
    agent/*)        echo "agent" ;;
    tool/*)         echo "tool" ;;
    skill/*)        echo "skill" ;;
    knowledge/*)    echo "knowledge" ;;
    vectordb/*)     echo "knowledge" ;;
    embedder/*)     echo "knowledge" ;;
    memory/*)       echo "memory" ;;
    mcp/*)          echo "mcp" ;;
    guardrail/*)    echo "guardrail" ;;
    *)              echo "" ;;
  esac
}

for file in $LIB_PYTHON_CHANGES; do
  mod=$(map_file_to_module "$file")
  if [ -n "$mod" ]; then
    MODULES_TO_TEST+=("$mod")
  fi
done

# Deduplicate
MODULES_TO_TEST=($(echo "${MODULES_TO_TEST[@]}" | tr ' ' '\n' | sort -u))

if [ ${#MODULES_TO_TEST[@]} -eq 0 ]; then
  echo '{"decision": "approve"}'
  exit 0
fi

# ── 3. Run review tests ────────────────────────────────────────────
FAILURES=""
TOTAL_PASS=0
TOTAL_FAIL=0
TOTAL_SKIP=0
TESTS_RUN=""

cd "$PROJECT_DIR"

for mod in "${MODULES_TO_TEST[@]}"; do
  SCRIPT="$REVIEW_DIR/review_${mod}.py"
  if [ ! -f "$SCRIPT" ]; then
    continue
  fi

  TESTS_RUN="${TESTS_RUN}${mod} "

  # Run with timeout, capture output
  OUTPUT=$(timeout 60 "$VENV_PYTHON" "$SCRIPT" 2>&1) || true
  EXIT_CODE=${PIPESTATUS[0]:-$?}

  # Count pass/fail/skip from output
  PASS_COUNT=$(echo "$OUTPUT" | grep -c '✅' || true)
  FAIL_COUNT=$(echo "$OUTPUT" | grep -c '❌' || true)
  SKIP_COUNT=$(echo "$OUTPUT" | grep -c '⚠️' || true)

  TOTAL_PASS=$((TOTAL_PASS + PASS_COUNT))
  TOTAL_FAIL=$((TOTAL_FAIL + FAIL_COUNT))
  TOTAL_SKIP=$((TOTAL_SKIP + SKIP_COUNT))

  if [ "$FAIL_COUNT" -gt 0 ] || [ "$EXIT_CODE" -ne 0 ]; then
    # Extract only the failure lines
    FAIL_LINES=$(echo "$OUTPUT" | grep -E '❌|Error|Traceback|assert' | head -20 || true)
    FAILURES="${FAILURES}
── review_${mod}.py ──
${FAIL_LINES}
"
  fi
done

# ── 4. Return decision ─────────────────────────────────────────────
if [ -z "$TESTS_RUN" ]; then
  echo '{"decision": "approve"}'
  exit 0
fi

if [ "$TOTAL_FAIL" -gt 0 ]; then
  # Escape for JSON: replace newlines, quotes, backslashes
  REASON=$(cat <<EOF
CODE REVIEW: ${TOTAL_FAIL} issue(s) found in modules [${TESTS_RUN}].

Present these findings to the user. Do NOT silently fix them — show each failure and ask the user which ones to address.

Failures:
${FAILURES}

Summary: ${TOTAL_PASS} passed | ${TOTAL_FAIL} failed | ${TOTAL_SKIP} skipped
EOF
)
  # JSON-safe escaping
  REASON_JSON=$(echo "$REASON" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')

  echo "{\"decision\": \"block\", \"reason\": ${REASON_JSON}}"
  exit 0
else
  echo '{"decision": "approve"}'
  exit 0
fi
