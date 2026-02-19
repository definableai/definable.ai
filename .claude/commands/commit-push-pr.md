---
description: "Commit changes, push to remote, and create a PR."
allowed-tools: Bash(git *), Bash(gh *)
---

# Commit, Push, and Create PR

## Current State

!`git status --short`

## Recent Commits (for style matching)

!`git log --oneline -5`

## Changes to Commit

!`git diff --stat`

## Instructions

1. Review the changes above
2. Stage relevant files — exclude `.env*`, `.workspace/`, `__pycache__/`, `.pyc` files
3. Write a concise commit message:
   - Imperative mood ("add", "fix", "update", "refactor")
   - Match the style of recent commits shown above
   - NO "Co-Authored-By" lines — NEVER add these
   - If multiple logical changes, use a multi-line message with bullet points
4. Commit the changes
5. Determine the branch:
   - If on `main`, create a feature branch first (e.g. `feat/<short-description>`)
   - If already on a feature branch, stay on it
6. Push to remote with `-u` flag
7. Create a PR using `gh pr create`:
   - Keep title under 70 characters
   - Use this body format:

```
## Summary
- <1-3 bullet points describing what changed and why>

## Test Plan
- [ ] ruff check passes
- [ ] ruff format passes
- [ ] Examples run successfully
- [ ] Scenario tests pass
```

8. Return the PR URL
