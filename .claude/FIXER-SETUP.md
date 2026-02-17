# Issue Fixer Agent — Setup Guide

## Overview

The Issue Fixer is a Claude Code agent that **automatically fixes GitHub issues**.
When a new issue is created (or labeled `claude-fix`), it:

1. Reads the issue and understands the problem
2. Traces the root cause in the Definable source code
3. Creates a `fix/issue-<N>` branch
4. Implements the fix following library conventions
5. Writes tests that verify the fix
6. Runs the full test suite + linter + type checker
7. Raises a PR with detailed explanation
8. If unable to fix → comments analysis + labels `needs-human`

**Two deployment options:**
- **Option A: GitHub Actions** (recommended — zero infrastructure)
- **Option B: Self-hosted webhook server** (for faster response, more control)

---

## Option A: GitHub Actions (Recommended)

### 1. Add Secrets

Go to your repo → Settings → Secrets and variables → Actions → New repository secret:

| Secret | Value | Required |
|--------|-------|----------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key (`sk-ant-...`) | ✅ Yes |
| `OPENAI_API_KEY` | OpenAI key for tests that need it | Optional |

`GITHUB_TOKEN` is automatically available in Actions — no setup needed.

### 2. The Workflow Is Already Created

The file `.github/workflows/fix-issue.yml` is already in your repo. It triggers on:
- **Issue labeled `claude-fix`** — most controlled option
- **Issue opened** (only if you set `AUTOFIX_ON_OPEN=true` in repo variables)
- **Manual dispatch** — go to Actions → "Auto Fix Issue" → Run workflow → enter issue number

### 3. Configure Trigger Behavior

Go to repo → Settings → Variables → Actions → New repository variable:

| Variable | Value | Effect |
|----------|-------|--------|
| `AUTOFIX_ON_OPEN` | `true` or `false` | Auto-fix every new issue (default: `false`) |

**Recommended setup:** Leave `AUTOFIX_ON_OPEN=false` and manually add the `claude-fix` label
to issues you want auto-fixed. This gives you control over which issues trigger the agent.

### 4. Create the `claude-fix` Label

```bash
gh label create "claude-fix" --description "Trigger auto-fix agent" --color "7057ff"
gh label create "needs-human" --description "Auto-fixer couldn't resolve" --color "d73a4a"
```

### 5. Test It

```bash
# Create a test issue
gh issue create --title "Test: Agent should handle None instructions gracefully" \
  --body "When creating Agent(model=model, instructions=None), it should work or raise a clear error." \
  --label "claude-fix"
```

Watch the Actions tab — you should see "Auto Fix Issue" running.

### 6. Cost Estimation

Each fix attempt uses roughly:
- **Claude tokens**: 50K-200K depending on complexity (~$1-5 per fix)
- **GitHub Actions minutes**: 5-25 minutes (free tier: 2000 min/month)
- **OpenAI tokens**: Only if tests require real model calls

---

## Option B: Self-Hosted Webhook Server

### 1. Server Setup

```bash
# Create system user
sudo useradd -r -m -d /opt/definable -s /bin/bash definable

# Switch to that user
sudo su - definable

# Clone repo
git clone https://github.com/<your-org>/definable.ai.git /opt/definable/repo
cd /opt/definable/repo

# Create Python venv for the webhook server
python3.12 -m venv /opt/definable/venv
source /opt/definable/venv/bin/activate

# Install webhook server deps
pip install -r .claude/server/requirements.txt

# Install Claude Code CLI
npm install -g @anthropic-ai/claude-code

# Install GitHub CLI
# See: https://cli.github.com/manual/installation

# Authenticate gh
gh auth login

# Create library venv (separate from webhook)
cd /opt/definable/repo
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[readers,serve,jwt,cron,runtime]"
pip install pytest pytest-asyncio

# Create directories
mkdir -p /opt/definable/logs
```

### 2. Configure Environment

```bash
# Copy example env
cp /opt/definable/repo/.claude/server/.env.example /opt/definable/.env

# Edit with your values
nano /opt/definable/.env
```

Fill in:
- `GITHUB_WEBHOOK_SECRET` — generate with `openssl rand -hex 32`
- `ANTHROPIC_API_KEY` — your Anthropic key
- `OPENAI_API_KEY` — optional, for tests

### 3. Make Script Executable

```bash
chmod +x /opt/definable/repo/.claude/server/fix-issue.sh
```

### 4. Install Systemd Service

```bash
sudo cp /opt/definable/repo/.claude/server/definable-fixer.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable definable-fixer
sudo systemctl start definable-fixer

# Check status
sudo systemctl status definable-fixer
sudo journalctl -u definable-fixer -f
```

### 5. Configure Nginx (if behind reverse proxy)

```nginx
server {
    listen 443 ssl;
    server_name fixer.definable.ai;

    ssl_certificate /etc/letsencrypt/live/fixer.definable.ai/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/fixer.definable.ai/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:9876;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Webhook payloads can be large
        client_max_body_size 10m;

        # Long timeout for fix operations
        proxy_read_timeout 1800;
    }
}
```

### 6. Configure GitHub Webhook

Go to your repo → Settings → Webhooks → Add webhook:

| Field | Value |
|-------|-------|
| Payload URL | `https://fixer.definable.ai/webhook` |
| Content type | `application/json` |
| Secret | Same value as `GITHUB_WEBHOOK_SECRET` in `.env` |
| Events | Select "Issues" only |
| Active | ✅ |

### 7. Test the Webhook

```bash
# Health check
curl https://fixer.definable.ai/health

# Manual trigger (bypass webhook)
curl -X POST https://fixer.definable.ai/fix/42

# Check status
curl https://fixer.definable.ai/status
```

### 8. Monitor

```bash
# Live logs
sudo journalctl -u definable-fixer -f

# Fix logs
ls -la /opt/definable/logs/

# Tail latest fix log
tail -f /opt/definable/logs/fix-issue-42-*.log
```

---

## How the Agent Decides What to Do

```
Issue Opened/Labeled
        │
        ▼
┌──────────────────┐     ┌─────────────┐
│ Skip labels?     │──→  │ Ignore      │
│ (wontfix, dup..) │ yes │             │
└────────┬─────────┘     └─────────────┘
         │ no
         ▼
┌──────────────────┐     ┌─────────────┐
│ claude-fix label │──→  │ Start fix   │
│ or AUTOFIX=true? │ yes │             │
└────────┬─────────┘     └─────────────┘
         │ no
         ▼
    Ignore

Start fix
    │
    ▼
┌──────────────────┐     ┌─────────────────┐
│ Read issue       │──→  │ No repro steps? │──→ Comment "needs-info", stop
│ Parse content    │     │ Unclear?        │
└────────┬─────────┘     └─────────────────┘
         │
         ▼
┌──────────────────┐     ┌─────────────────┐
│ Reproduce bug    │──→  │ Can't repro?    │──→ Comment "can't reproduce", stop
└────────┬─────────┘     └─────────────────┘
         │
         ▼
┌──────────────────┐
│ Root cause       │
│ analysis         │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Create branch    │
│ fix/issue-<N>    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Implement fix    │
│ + write tests    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌─────────────────┐
│ All tests pass?  │──→  │ No?            │──→ Fix or revert
└────────┬─────────┘     └─────────────────┘
         │ yes
         ▼
┌──────────────────┐     ┌─────────────────────────┐
│ Confident in fix?│──→  │ No?                     │──→ Comment analysis,
└────────┬─────────┘     │ (too complex, design    │    label needs-human,
         │ yes           │  decision needed, etc.)  │    delete branch
         ▼               └─────────────────────────┘
┌──────────────────┐
│ Push + create PR │
│ referencing #N   │
└──────────────────┘
```

---

## Updating the Agent

To update the agent's behavior, edit these files:

| File | Controls |
|------|----------|
| `.claude/agents/issue-fixer.md` | Agent personality, conventions, workflow |
| `.claude/commands/fix-issue.md` | Command pipeline (steps, validation) |
| `.github/workflows/fix-issue.yml` | Trigger rules, GitHub Actions config |
| `.claude/server/webhook.py` | Self-hosted trigger rules, concurrency |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Workflow doesn't trigger | Check: issue must be `open`, no skip labels, `claude-fix` label present |
| Agent can't push branch | Check `GITHUB_TOKEN` permissions: needs `contents: write` |
| Agent can't create PR | Check `GITHUB_TOKEN` permissions: needs `pull-requests: write` |
| Tests fail on CI but pass locally | Check Python version and installed extras match CI |
| Agent keeps timing out | Increase `timeout-minutes` in workflow (default: 30) |
| Self-hosted: webhook not receiving | Check nginx config, firewall, GitHub webhook delivery log |
| Agent creates bad fix | Review the agent prompt in `.claude/agents/issue-fixer.md` |
| Too many API costs | Set `AUTOFIX_ON_OPEN=false`, use `claude-fix` label for selective triggering |
| Multiple agents fixing same issue | Concurrency control is built in (both Actions and webhook) |
