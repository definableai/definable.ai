#!/usr/bin/env bash
#
# fix-issue.sh — Triggered by the webhook server to fix a GitHub issue.
#
# Usage: ./fix-issue.sh <issue_number>
#
# This script:
# 1. Pulls latest main
# 2. Sets up the Python environment
# 3. Runs Claude Code with the /fix-issue command
# 4. Logs everything
#
# Environment:
#   REPO_PATH          — Path to repo (default: current directory)
#   ANTHROPIC_API_KEY  — Required for Claude Code
#   GITHUB_TOKEN       — For gh CLI (set via gh auth login or env)
#

set -euo pipefail

ISSUE_NUMBER="${1:?Usage: fix-issue.sh <issue_number>}"
REPO_PATH="${REPO_PATH:-$(pwd)}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

echo "═══════════════════════════════════════════════════"
echo "  Issue Fixer — Starting fix for #$ISSUE_NUMBER"
echo "  Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  Repo: $REPO_PATH"
echo "═══════════════════════════════════════════════════"

cd "$REPO_PATH"

# ─── Pre-flight ────────────────────────────────

# Ensure clean state
echo "[1/7] Ensuring clean working state..."
git stash push -m "auto-stash-fix-$ISSUE_NUMBER-$TIMESTAMP" 2>/dev/null || true
git checkout main 2>/dev/null || true
git pull origin main

# Activate virtual environment
echo "[2/7] Setting up Python environment..."
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    python3.12 -m venv .venv
    source .venv/bin/activate
fi

# Install deps
pip install -e ".[readers,serve,jwt,cron,runtime]" -q
pip install pytest pytest-asyncio -q

# Create workspace
mkdir -p .workspace .claude/memory

# ─── Verify tools ──────────────────────────────

echo "[3/7] Verifying tools..."
command -v claude >/dev/null 2>&1 || {
    echo "❌ Claude Code CLI not found. Install: npm install -g @anthropic-ai/claude-code"
    exit 1
}
command -v gh >/dev/null 2>&1 || {
    echo "❌ GitHub CLI not found. Install: https://cli.github.com"
    exit 1
}
gh auth status >/dev/null 2>&1 || {
    echo "❌ gh not authenticated. Run: gh auth login"
    exit 1
}

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "❌ ANTHROPIC_API_KEY not set"
    exit 1
fi

# ─── Verify issue exists and is open ──────────

echo "[4/7] Checking issue #$ISSUE_NUMBER..."
ISSUE_STATE=$(gh issue view "$ISSUE_NUMBER" --json state -q '.state' 2>/dev/null || echo "NOT_FOUND")

if [ "$ISSUE_STATE" = "NOT_FOUND" ]; then
    echo "❌ Issue #$ISSUE_NUMBER not found"
    exit 1
fi

if [ "$ISSUE_STATE" != "OPEN" ]; then
    echo "⏭️ Issue #$ISSUE_NUMBER is $ISSUE_STATE, skipping"
    exit 0
fi

# Check if PR already exists
EXISTING_PR=$(gh pr list --head "fix/issue-$ISSUE_NUMBER" --state open --json number -q '.[0].number' 2>/dev/null || echo "")
if [ -n "$EXISTING_PR" ]; then
    echo "⏭️ PR #$EXISTING_PR already exists for issue #$ISSUE_NUMBER"
    exit 0
fi

# ─── Run Claude Code ──────────────────────────

echo "[5/7] Running Claude Code..."
echo "  Command: claude -p \"/fix-issue $ISSUE_NUMBER\" --dangerously-skip-permissions"

EXIT_CODE=0
claude -p "/fix-issue $ISSUE_NUMBER" \
    --dangerously-skip-permissions \
    --max-tokens 100000 \
    2>&1 || EXIT_CODE=$?

echo "[6/7] Claude Code finished with exit code: $EXIT_CODE"

# ─── Check result ─────────────────────────────

echo "[7/7] Checking results..."

# Check if a PR was created
PR_URL=$(gh pr list --head "fix/issue-$ISSUE_NUMBER" --state open --json url -q '.[0].url' 2>/dev/null || echo "")

if [ -n "$PR_URL" ]; then
    echo "══════════════════════════════════════════════"
    echo "  ✅ PR created: $PR_URL"
    echo "══════════════════════════════════════════════"
else
    echo "══════════════════════════════════════════════"
    echo "  ⚠️  No PR created for issue #$ISSUE_NUMBER"
    echo "  Check issue comments for analysis."
    echo "══════════════════════════════════════════════"
fi

# Cleanup — back to main
git checkout main 2>/dev/null || true
git stash pop 2>/dev/null || true
rm -rf .workspace/* 2>/dev/null || true

echo "Done. $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $EXIT_CODE
