#!/usr/bin/env python3
"""
Lightweight webhook server that receives GitHub issue events
and triggers Claude Code to fix them.

Usage:
    python webhook.py

Environment variables:
    GITHUB_WEBHOOK_SECRET  — Webhook secret for signature verification
    REPO_PATH              — Path to the cloned repo (default: /opt/definable/repo)
    LOG_DIR                — Path for logs (default: /opt/definable/logs)
    PORT                   — Server port (default: 9876)
    ALLOWED_LABELS         — Comma-separated labels that trigger fixes (default: claude-fix,bug)
    SKIP_LABELS            — Comma-separated labels to skip (default: wontfix,duplicate,question,needs-human)

Requires: pip install fastapi uvicorn httpx
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

WEBHOOK_SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET", "")
REPO_PATH = Path(os.environ.get("REPO_PATH", "/opt/definable/repo"))
LOG_DIR = Path(os.environ.get("LOG_DIR", "/opt/definable/logs"))
PORT = int(os.environ.get("PORT", "9876"))
ALLOWED_LABELS = set(os.environ.get("ALLOWED_LABELS", "claude-fix,bug").split(","))
SKIP_LABELS = set(os.environ.get("SKIP_LABELS", "wontfix,duplicate,question,needs-human,discussion,enhancement").split(","))
MAX_CONCURRENT_FIXES = int(os.environ.get("MAX_CONCURRENT_FIXES", "1"))

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(message)s",
  handlers=[
    logging.StreamHandler(sys.stdout),
    logging.FileHandler(LOG_DIR / "webhook.log"),
  ],
)
logger = logging.getLogger("issue-fixer-webhook")

# ──────────────────────────────────────────────
# State
# ──────────────────────────────────────────────

active_fixes: dict[int, asyncio.Task] = {}
fix_semaphore = asyncio.Semaphore(MAX_CONCURRENT_FIXES)

# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────

app = FastAPI(title="Definable Issue Fixer Webhook", version="1.0.0")


def verify_signature(payload: bytes, signature: str) -> bool:
  """Verify GitHub webhook HMAC-SHA256 signature."""
  if not WEBHOOK_SECRET:
    logger.warning("No GITHUB_WEBHOOK_SECRET set — skipping signature check")
    return True

  if not signature.startswith("sha256="):
    return False

  expected = hmac.new(
    WEBHOOK_SECRET.encode("utf-8"),
    payload,
    hashlib.sha256,
  ).hexdigest()

  return hmac.compare_digest(f"sha256={expected}", signature)


def should_fix_issue(payload: dict) -> tuple[bool, str]:
  """Determine if an issue should be auto-fixed. Returns (should_fix, reason)."""
  action = payload.get("action", "")
  issue = payload.get("issue", {})
  issue_number = issue.get("number", 0)
  state = issue.get("state", "")
  labels = {label["name"] for label in issue.get("labels", [])}

  # Must be open
  if state != "open":
    return False, f"Issue #{issue_number} is {state}"

  # Check skip labels
  skip_matches = labels & SKIP_LABELS
  if skip_matches:
    return False, f"Issue #{issue_number} has skip label(s): {skip_matches}"

  # Already being fixed
  if issue_number in active_fixes:
    return False, f"Issue #{issue_number} is already being fixed"

  # On 'labeled' action, check if the added label triggers a fix
  if action == "labeled":
    added_label = payload.get("label", {}).get("name", "")
    if added_label == "claude-fix":
      return True, f"Label 'claude-fix' added to #{issue_number}"
    return False, f"Label '{added_label}' doesn't trigger auto-fix"

  # On 'opened' action, check if any label matches
  if action == "opened":
    trigger_matches = labels & ALLOWED_LABELS
    if trigger_matches:
      return True, f"Issue #{issue_number} opened with trigger label(s): {trigger_matches}"
    # If no trigger labels, check if AUTOFIX_ON_OPEN is enabled
    if os.environ.get("AUTOFIX_ON_OPEN", "false").lower() == "true":
      return True, f"Issue #{issue_number} opened (AUTOFIX_ON_OPEN=true)"
    return False, f"Issue #{issue_number} opened but no trigger labels"

  return False, f"Action '{action}' doesn't trigger auto-fix"


async def run_fixer(issue_number: int, issue_title: str):
  """Run Claude Code to fix an issue. Runs in background."""
  async with fix_semaphore:
    log_file = LOG_DIR / f"fix-issue-{issue_number}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.log"
    logger.info(f"Starting fix for issue #{issue_number}: {issue_title}")
    logger.info(f"Log file: {log_file}")

    try:
      process = await asyncio.create_subprocess_exec(
        "bash",
        str(REPO_PATH / ".claude" / "server" / "fix-issue.sh"),
        str(issue_number),
        cwd=str(REPO_PATH),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={
          **os.environ,
          "ISSUE_NUMBER": str(issue_number),
          "REPO_PATH": str(REPO_PATH),
        },
      )

      stdout, _ = await asyncio.wait_for(
        process.communicate(),
        timeout=1800,  # 30 minute timeout
      )

      # Write log
      log_file.write_bytes(stdout)

      if process.returncode == 0:
        logger.info(f"✅ Fix for issue #{issue_number} completed successfully")
      else:
        logger.warning(f"⚠️ Fix for issue #{issue_number} exited with code {process.returncode}")

    except asyncio.TimeoutError:
      logger.error(f"❌ Fix for issue #{issue_number} timed out after 30 minutes")
      if process:
        process.kill()
    except Exception as e:
      logger.error(f"❌ Fix for issue #{issue_number} failed: {e}")
    finally:
      active_fixes.pop(issue_number, None)


@app.post("/webhook")
async def handle_webhook(request: Request):
  """Handle incoming GitHub webhook events."""
  # Verify signature
  payload_bytes = await request.body()
  signature = request.headers.get("X-Hub-Signature-256", "")

  if not verify_signature(payload_bytes, signature):
    raise HTTPException(status_code=401, detail="Invalid signature")

  # Check event type
  event_type = request.headers.get("X-GitHub-Event", "")
  if event_type != "issues":
    return {"status": "ignored", "reason": f"Event type '{event_type}' not handled"}

  # Parse payload
  payload = json.loads(payload_bytes)
  issue = payload.get("issue", {})
  issue_number = issue.get("number", 0)
  issue_title = issue.get("title", "")

  # Check eligibility
  should_fix, reason = should_fix_issue(payload)
  logger.info(f"Issue #{issue_number} ({issue_title}): {reason}")

  if not should_fix:
    return {"status": "skipped", "reason": reason, "issue": issue_number}

  # Launch fixer in background
  task = asyncio.create_task(run_fixer(issue_number, issue_title))
  active_fixes[issue_number] = task

  return {
    "status": "accepted",
    "issue": issue_number,
    "title": issue_title,
    "message": f"Fix started for issue #{issue_number}",
  }


@app.get("/health")
async def health():
  """Health check endpoint."""
  return {
    "status": "ok",
    "active_fixes": list(active_fixes.keys()),
    "repo_path": str(REPO_PATH),
    "timestamp": datetime.now(timezone.utc).isoformat(),
  }


@app.get("/status")
async def status():
  """Detailed status of active and recent fixes."""
  recent_logs = sorted(LOG_DIR.glob("fix-issue-*.log"), reverse=True)[:10]
  return {
    "active_fixes": list(active_fixes.keys()),
    "recent_logs": [
      {"name": log.name, "size": log.stat().st_size, "modified": datetime.fromtimestamp(log.stat().st_mtime, tz=timezone.utc).isoformat()}
      for log in recent_logs
    ],
  }


@app.post("/fix/{issue_number}")
async def manual_fix(issue_number: int):
  """Manually trigger a fix for an issue (no webhook needed)."""
  if issue_number in active_fixes:
    return {"status": "already_running", "issue": issue_number}

  task = asyncio.create_task(run_fixer(issue_number, f"Manual trigger #{issue_number}"))
  active_fixes[issue_number] = task

  return {"status": "accepted", "issue": issue_number}


if __name__ == "__main__":
  logger.info(f"Starting webhook server on port {PORT}")
  logger.info(f"Repo path: {REPO_PATH}")
  logger.info(f"Log directory: {LOG_DIR}")
  logger.info(f"Allowed labels: {ALLOWED_LABELS}")
  logger.info(f"Skip labels: {SKIP_LABELS}")
  logger.info(f"Max concurrent fixes: {MAX_CONCURRENT_FIXES}")

  uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
