---
description: >
  One-time interactive setup. Collects API keys, validates them,
  creates .env.test, maps the codebase, saves everything to memory.
  This is the ONLY command that asks the user questions.
---

# /setup — One-Time Interactive Setup

This is the **only** command that talks to the user. After setup, everything is autonomous.

## Step 1: Collect Credentials

Ask for ALL of these. If the user doesn't have one, mark it as skipped.

```
Required (for basic evaluation):
  OPENAI_API_KEY — OpenAI API key (needed for most agent tests)

Optional (for full coverage):
  DEEPSEEK_API_KEY — DeepSeek API key
  MOONSHOT_API_KEY — Moonshot/Kimi API key
  XAI_API_KEY — xAI/Grok API key
  VOYAGE_API_KEY — Voyage AI embeddings
  COHERE_API_KEY — Cohere reranker
  SERPAPI_API_KEY — SerpAPI for web search
  GOOGLE_SEARCH_API_KEY — Google Custom Search
  DISCORD_BOT_TOKEN — Discord bot (interface testing)
  TELEGRAM_BOT_TOKEN — Telegram bot (interface testing)
  GITHUB_TOKEN — GitHub API (issue filing, defaults to gh CLI)
```

## Step 2: Validate Each Key

For each provided key, run a minimal API call to verify it works:
- OpenAI: `python -c "from openai import OpenAI; OpenAI(api_key='KEY').models.list()"`
- DeepSeek: Similar minimal call
- Voyage: Similar minimal call

Print ✅ or ❌ for each key.

## Step 3: Save to .env.test

```bash
cat > .env.test << 'EOF'
export OPENAI_API_KEY="sk-..."
export VOYAGE_API_KEY="..."
# ... only keys that were provided and validated
EOF
```

## Step 4: Explore Codebase

Run the **api-explorer** subagent to map the full API surface.
Save results to `.claude/memory/project-profile.md`.

## Step 5: Collect Preferences

Ask the user:
1. **Timeout per test** (default: 120s)
2. **Skip any providers?** (e.g., "skip deepseek")
3. **Max issues per run?** (default: 20)
4. **Extra labels for issues?** (e.g., "sprint-3")

## Step 6: Save Memory

Write all 6 memory files:
1. `.claude/memory/credentials.md` — key status table
2. `.claude/memory/project-profile.md` — from api-explorer
3. `.claude/memory/evaluation-history.md` — empty (first run)
4. `.claude/memory/known-issues.md` — empty
5. `.claude/memory/fix-history.md` — empty (written by issue-fixer agent)
6. `.claude/memory/user-preferences.md` — collected preferences

## Step 7: Confirm

```
✅ Setup complete!

Credentials: <N> validated, <N> skipped
Project: <version>, <N> modules, <N> public symbols
Memory: 6 files saved to .claude/memory/

You can now run /evaluate — it will be fully autonomous.
Or run: claude -p "/evaluate" --dangerously-skip-permissions
```
