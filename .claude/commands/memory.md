---
description: >
  View, update, or clear evaluator memory.
  "What do you remember?" — show all stored data
  "Forget my keys" — delete credentials only
  "Forget everything" — wipe all memory
  "Update memory" — re-scan project, refresh all data
---

# /memory — Memory Management

## Commands

### "What do you remember?"
Read and summarize all 6 memory files:
- `.claude/memory/credentials.md` — which keys are available
- `.claude/memory/project-profile.md` — project structure
- `.claude/memory/evaluation-history.md` — past eval runs
- `.claude/memory/known-issues.md` — filed issues
- `.claude/memory/fix-history.md` — past fix attempts by issue-fixer
- `.claude/memory/user-preferences.md` — preferences

### "Forget my keys"
- Delete `.claude/memory/credentials.md`
- Delete `.env.test`
- Print: "Credentials cleared. Run /setup to reconfigure."

### "Forget everything"
- Delete all 6 files in `.claude/memory/`
- Delete `.env.test`
- Recreate empty `.claude/memory/` directory
- Print: "All memory cleared. Run /setup to start fresh."

### "Update memory"
- Re-run **api-explorer** to refresh project profile
- Check `.env.test` and update credentials status
- Verify known issues are still open: `gh issue view <N> --json state`
- Print summary of what changed
