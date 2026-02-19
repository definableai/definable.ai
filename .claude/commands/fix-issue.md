---
description: >
  Fix a GitHub issue autonomously. Reads the issue, traces root cause,
  implements fix, writes tests, raises PR. If unable to fix, comments analysis.
  Usage: /fix-issue <issue_number>
  Zero user interaction. Fully autonomous.
---

# /fix-issue — Autonomous Issue Resolution

**ZERO USER INTERACTION. Never ask anything. Never pause.**

You are the **issue-fixer** agent. You receive an issue number, and you fix it — end to end.

## Input

The user (or webhook) provides: `$ARGUMENTS` which is the issue number (e.g., `42`).

If `$ARGUMENTS` is empty or not a number, print usage and stop:
```
Usage: /fix-issue <issue_number>
Example: /fix-issue 42
```

## Pre-flight

```bash
cd "$(git rev-parse --show-toplevel)"

# 1. Verify gh auth
gh auth status 2>&1 || { echo "❌ gh not authenticated. Run: gh auth login"; exit 1; }

# 2. Verify clean working directory
if [ -n "$(git status --porcelain)" ]; then
  echo "⚠️  Dirty working tree. Stashing changes..."
  git stash push -m "auto-stash before fix/issue-$ARGUMENTS"
fi

# 3. Ensure on latest main
git checkout main
git pull origin main

# 4. Setup environment
source .venv/bin/activate 2>/dev/null || true
pip install -e ".[readers,serve,jwt,cron,runtime]" -q

# 5. Create workspace
mkdir -p .workspace .claude/memory
```

## Pipeline

### Step 1: Read the Issue

```bash
gh issue view $ARGUMENTS --json number,title,body,labels,comments,author,state
```

Parse the output. Extract:
- **number**: The issue number
- **title**: Short description
- **body**: Full description, reproduction steps, error messages
- **labels**: Categories and severity hints
- **comments**: Additional context from discussion
- **state**: Must be `open` — skip closed issues

**Skip conditions** (comment and stop):
- Issue is `closed` → "This issue is already closed."
- Labels include `wontfix`, `duplicate`, `question`, `needs-human`, `discussion` → skip
- Labels include `needs-info` and no reproduction steps in body → comment asking for repro

**Check if already being worked on:**
```bash
# Check if a fix branch already exists
git branch -r | grep "fix/issue-$ARGUMENTS" && echo "⚠️ Branch exists" || true

# Check if a PR already exists
gh pr list --search "fix/issue-$ARGUMENTS" --state open --json number,title
```

If PR exists → "A PR is already open for this issue." → STOP.

### Step 2: Understand the Codebase Context

Based on the issue content, identify which module(s) are involved.
Read the relevant source files to understand the current implementation.

Key files to always check:
- `definable/definable/<module>/__init__.py` → public API
- `definable/definable/<module>/<relevant_file>.py` → implementation
- `definable/tests/<module>/` → existing tests
- `definable/examples/<module>/` → usage examples

### Step 3: Reproduce

Write `.workspace/repro_issue_$ARGUMENTS.py`:
- Extract reproduction code from the issue body
- If no repro provided, write one based on the described behavior
- Run it: `timeout 60 python .workspace/repro_issue_$ARGUMENTS.py`
- If it passes (bug not reproduced) → comment on issue, stop

### Step 4: Analyze Root Cause

Read the source code along the execution path. Write analysis to
`.workspace/analysis_issue_$ARGUMENTS.md` with:
- Exact file:line where bug manifests
- Why it happens
- Blast radius
- Proposed fix approach

### Step 5: Create Branch and Fix

```bash
git checkout -b fix/issue-$ARGUMENTS
```

Implement the fix following the conventions in the issue-fixer agent definition.

### Step 6: Write Tests

Add test(s) to `definable/tests/`. The test must:
1. Fail on `main` (verify by checking out main, running test, checking out fix branch)
2. Pass on the fix branch
3. Cover the exact issue scenario
4. Not require API keys (use MockModel where possible)

### Step 7: Run Full Validation

```bash
# New test
pytest definable/tests/<new_test_path> -v

# Full offline suite
pytest definable/tests/ \
  -m "not openai and not deepseek and not moonshot and not xai and not telegram and not discord and not signal and not postgres and not redis and not qdrant and not chroma and not mongodb and not pinecone and not mistral and not mem0" \
  -v --tb=short

# Linter + formatter
ruff check definable/definable/ --fix
ruff format definable/definable/

# Type checker
mypy definable/definable/ --ignore-missing-imports

# Original repro
python .workspace/repro_issue_$ARGUMENTS.py
```

If ANY step fails → fix it. If you can't fix it → revert, comment on issue, clean up.

### Step 8: Real-World Validation

Write `.workspace/realworld_issue_$ARGUMENTS.py` — a realistic developer scenario.
Run it. Must pass.

### Step 9: Commit, Push, PR

```bash
# Remove debug artifacts
grep -rn "breakpoint()\|import pdb\|print(" definable/definable/ --include="*.py" | grep -v "# noqa" || true

# Commit
git add -A
git commit -m "fix(<module>): <description>

Fixes #$ARGUMENTS.

<explanation>"

# Push
git push origin fix/issue-$ARGUMENTS

# Create PR
gh pr create \
  --base main \
  --head fix/issue-$ARGUMENTS \
  --title "fix(<module>): <title from issue>" \
  --body "<structured PR body with summary, changes, root cause, testing checklist>"
```

### Step 9b: If Unable to Fix

```bash
# Comment analysis on the issue
gh issue comment $ARGUMENTS --body "<detailed analysis>"

# Label it
gh issue edit $ARGUMENTS --add-label "needs-human"

# Clean up
git checkout main
git branch -D fix/issue-$ARGUMENTS 2>/dev/null || true
```

### Step 10: Update Memory

Append to `.claude/memory/fix-history.md`:
```markdown
## Issue #$ARGUMENTS — <title>
- **Date**: <ISO timestamp>
- **Status**: fixed / unable-to-fix
- **Branch**: fix/issue-$ARGUMENTS
- **PR**: #<PR number> (if created)
- **Files changed**: <list>
- **Root cause**: <1 sentence>
```

### Step 11: Cleanup

```bash
rm -f .workspace/repro_issue_$ARGUMENTS.py
rm -f .workspace/analysis_issue_$ARGUMENTS.md
rm -f .workspace/realworld_issue_$ARGUMENTS.py
git checkout main

# Restore stash if we stashed earlier
git stash pop 2>/dev/null || true
```

## Output

Print a summary:
```
══════════════════════════════════════════════════
  ISSUE #$ARGUMENTS — FIX RESULT
══════════════════════════════════════════════════
  Status:     ✅ Fixed / ❌ Unable to fix
  Branch:     fix/issue-$ARGUMENTS
  PR:         #<N> (or "N/A")
  Files:      <N> files changed
  Tests:      <N> new, all passing
  Comment:    <link if commented>
══════════════════════════════════════════════════
```
