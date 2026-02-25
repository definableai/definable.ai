# Agent Team System — Definable AI

## Overview

This system runs an autonomous agent team using parallel Claude Code instances.
Each agent has a distinct role, mindset, and instruction set.
Communication happens via shared files in `.agents/`.

## Architecture

```
YOU (Driver)
 │
 ├── Write task brief → .agents/queue/task.md
 │
 ├── Phase 1: RESEARCH (1 agent)
 │   └── Output → .agents/handoffs/research.md
 │
 ├── Phase 2: PLAN (1 agent)
 │   └── Output → .agents/handoffs/plan.md
 │   └── ⏸️  YOU REVIEW & APPROVE
 │
 ├── Phase 3: DEVELOP (2-4 agents)
 │   └── Output → code changes + .agents/handoffs/dev-report.md
 │
 ├── Phase 4: TEST (1-2 agents, fresh mindset)
 │   └── Output → .agents/handoffs/test-report.md
 │
 ├── Phase 5: EVALUATE (1 agent)
 │   └── Output → example files + .agents/handoffs/eval-report.md
 │
 ├── Phase 6: DOCS (1 agent)
 │   └── Output → updated docs + .agents/handoffs/docs-report.md
 │
 └── ⏸️  YOU REVIEW & PUSH
```

## Terminal Layout (8 panes)

```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ Researcher   │ Planner      │ Developer 1  │ Developer 2  │
│ /researcher  │ /planner     │ /developer   │ /developer   │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ Tester 1     │ Tester 2     │ Evaluator    │ Docs Agent   │
│ /tester      │ /tester      │ /evaluator   │ /docs-agent  │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

## How to Use

### Step 1: Write a Task Brief
Edit `.agents/queue/task.md` with your feature request.

### Step 2: Launch Researcher
In terminal 1: `claude` → then type `/researcher`
Wait for it to finish (writes to `.agents/handoffs/research.md`)

### Step 3: Launch Planner
In terminal 2: `claude` → then type `/planner`
Wait for it to finish (writes to `.agents/handoffs/plan.md`)

### Step 4: Review the Plan
Read `.agents/handoffs/plan.md`. If approved:
```bash
echo "APPROVED" > .agents/queue/plan-status.txt
```
If changes needed, edit the plan file and re-run `/planner`.

### Step 5: Launch Developers
In terminals 3-4: `claude` → then type `/developer`
Each developer reads the plan, picks unassigned phases, builds.

### Step 6: Launch Testers
In terminals 5-6: `claude` → then type `/tester`
Testers work from a FRESH perspective — they don't read dev context.

### Step 7: Launch Evaluator
In terminal 7: `claude` → then type `/evaluator`

### Step 8: Launch Docs Agent
In terminal 8: `claude` → then type `/docs-agent`

### Step 9: Review & Push
Read all handoff reports. Run tests yourself. Push when satisfied.

## File Protocol

| File | Written By | Read By |
|------|-----------|---------|
| `.agents/queue/task.md` | You | All agents |
| `.agents/queue/plan-status.txt` | You | Developer, Tester |
| `.agents/handoffs/research.md` | Researcher | Planner |
| `.agents/handoffs/plan.md` | Planner | You, Developer |
| `.agents/handoffs/dev-report.md` | Developer | Tester, Evaluator |
| `.agents/handoffs/test-report.md` | Tester | You |
| `.agents/handoffs/eval-report.md` | Evaluator | You, Docs Agent |
| `.agents/handoffs/docs-report.md` | Docs Agent | You |

## Rules
- Agents NEVER push to git without your approval
- Agents commit to a feature branch, not main
- Developers claim tasks by writing their agent ID to the plan file
- Testers NEVER read developer code comments about "why" — they test behavior only
- All handoff files include timestamps and agent identity
