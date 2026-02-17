---
description: >
  Manually file a GitHub issue for a specific bug.
  Usage: /file-issue <description of the bug>
  The issue-filer agent will investigate, write the issue, and file it.
---

# /file-issue — Manual Issue Filing

Use the **issue-filer** subagent to:

1. Investigate the described bug in the codebase
2. Write a developer-grade reproduction
3. Check `.claude/memory/known-issues.md` for duplicates
4. Check GitHub for existing issues
5. File via `gh issue create` (or fallback to `.claude/reports/unfiled-issues.md`)
6. Update `.claude/memory/known-issues.md`

Pass the bug description as the argument to this command.
