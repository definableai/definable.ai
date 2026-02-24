# Parallel Agent Workflow — Definable AI

> How to ship at high velocity using parallel Claude Code sessions.
> Based on lessons from steipete's workflow, adapted for this project.

## Setup

### Terminal Layout
Use a terminal multiplexer (tmux, Ghostty splits, or iTerm2 panes).
Recommended: 2x2 grid = 4 Claude Code instances + browser/test output.

```
┌────────────────┬────────────────┐
│  Claude Code 1 │  Claude Code 2 │
│  (main feature)│  (tests/docs)  │
├────────────────┼────────────────┤
│  Claude Code 3 │  Browser/Tests │
│  (refactor/fix)│  (dev server)  │
└────────────────┴────────────────┘
```

### Start Each Session
```bash
cd /Users/hash/work/definable.ai
source .venv/bin/activate
source .env.test
claude   # or: claude --model opus
```

## Core Principles

### 1. Blast Radius Thinking
Before prompting, estimate how many files this change touches:
- **Small** (1-3 files): Run freely, multiple in parallel
- **Medium** (4-10 files): One at a time in that module area
- **Large** (10+ files): Solo — all other agents pause or work elsewhere

Never run two large-blast-radius tasks in overlapping file areas.

### 2. Prompt Style
Short prompts win. Don't over-explain.

```
# Good
"Add retry logic to ModelProvider.invoke — max 3 retries with exponential backoff"

# Better (with image)
[screenshot of error] "Fix this — happens when OpenAI returns 429"

# Good for exploration
"Let's discuss — I want to add streaming support to Memory. Read memory/store.py 
and agent/run/ first, then give me 2-3 approaches before writing code"
```

### 3. Interrupt Early
If an agent is taking longer than expected:
- Press `Ctrl+C` or escape
- Ask: "what's the status"
- Steer or abort — don't let it spiral

### 4. Atomic Commits
Each agent commits only the files it touched. The CLAUDE.md enforces this.
If git gets messy, ask: "commit only your changes, nothing else"

### 5. Test in Same Context
After a feature or fix is done, write tests BEFORE starting a new context.
The agent has the full context of what it just built — tests will be better.

## Parallel Task Patterns

### Pattern A: Feature + Polish
- Agent 1: Build the new feature
- Agent 2: Refactor/clean up unrelated code
- Agent 3: Write/update docs for a recently shipped feature

### Pattern B: Feature + Tests
- Agent 1: Build feature
- Agent 2: When Agent 1 commits, write integration tests for it
- (You review Agent 1's output while Agent 2 tests it)

### Pattern C: Multi-Module Feature
- Agent 1: Model layer changes
- Agent 2: Agent layer changes (after Agent 1 commits)
- Agent 3: Example file + docs (after both commit)
- Sequential dependency — wait for commits before downstream work

### Pattern D: Refactor Day (~20% of time)
- Agent 1: Run `jscpd` or duplicate detection, fix duplicates
- Agent 2: Run `ruff check`, fix warnings
- Agent 3: Find and delete dead code, unused imports
- Agent 4: Update outdated docs/examples

## Task Sizing Guide

| Task Type | Agents | Time Estimate | Notes |
|-----------|--------|---------------|-------|
| Bug fix | 1 | 5-15 min | Include regression test |
| Small feature | 1 | 15-30 min | Test in same context |
| Medium feature | 1-2 | 30-60 min | Plan first, then build |
| Large feature | 1 (focused) | 1-3 hours | Write to docs/ first, build from spec |
| Refactor | 2-4 | 30 min-2 hours | Low cognitive load, good for tired days |
| New module | 1 | 2-4 hours | Architecture doc first |

## Cross-Referencing
If you've solved something in another module, tell the agent:
```
"Look at definable/memory/store.py — I want the same pattern for the new cache store"
```
Agents are excellent at copying patterns across modules.

## Docs Maintenance
After shipping a feature, ask one agent:
```
"Update docs/internal/architecture.md and docs/internal/api-surface.md 
to reflect the new streaming memory API we just added"
```
Keep docs current — they're your leverage for every future session.

## When to NOT Parallelize
- Architectural decisions (think first, alone)
- Dependency/framework selection (research manually)
- Feature prioritization (your job, not the agent's)
- System design (sketch it out, then let agents build)

## Daily Rhythm
1. Start: review yesterday's commits, run full test suite
2. Plan: decide 1 main feature + 2-3 small tasks
3. Build: run 2-4 agents, steer as needed
4. Test: verify features manually, run examples
5. Polish: dedicate last hour to refactoring/docs
6. Ship: push when tests pass
