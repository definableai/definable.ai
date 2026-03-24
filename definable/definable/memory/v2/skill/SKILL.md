---
name: memory-manager
description: Manages two-tier user memory — working memory (always loaded) and archived memory (indexed, on-demand). Instructs the agent on when and how to store, recall, and forget information.
version: 1.0.0
tags: [memory, recall, archive, preferences, context]
---

## Memory System

You have a two-tier memory system for each user:

**Working memory** (visible in `<working_memory>` tags above) is your scratchpad — always loaded into every turn. It should contain ALL active facts about the current user: identity, preferences, current project state, team, schedule, active constraints. This is your primary source of truth.

**Archived memory** stores detail, history, and context that doesn't need to be in every prompt. Access it via `recall_memory` (search summaries) → `fetch_memory_entries` (load full content).

## Memory Categories

| Category | What to store | When to save |
|----------|--------------|-------------|
| `user` | Identity, role, expertise, preferences, contacts | User shares personal/professional info |
| `feedback` | How to approach work — corrections AND confirmations | User corrects you OR validates a non-obvious approach |
| `project` | Goals, decisions, deadlines, risks, status | User shares project/work context |
| `reference` | Pointers to external systems, URLs, tools | User mentions external resources |
| `conversation` | Key takeaways worth preserving | Archiving overflow from working memory |

## Critical Rules

### Rule 1: Working memory is comprehensive, not minimal
Working memory should contain ALL active facts — not just the latest one. When the user shares new info, ADD it to working memory alongside existing facts. Only remove items that are explicitly superseded, contradicted, or forgotten.

**Wrong:** Rewrite WM with only the latest fact, dropping earlier ones.
**Right:** Rewrite WM with the latest fact merged into everything already there.

### Rule 2: Always search before saying "I don't know"
If the user asks a question and the answer is NOT in your working memory, you MUST call `recall_memory` to search archived memory before responding. Never say "I don't have that information" or "Could you share that?" without searching first.

### Rule 3: Archive AND update working memory
When the user shares important information, do BOTH:
1. Call `update_working_memory` to add it to the scratchpad
2. Call `archive_to_memory` to preserve the detail long-term

Archive is not a substitute for working memory. Both serve different purposes.

### Rule 4: Use tools for actions, not text
When asked to send, draft, search, or check email — ALWAYS use the available email/MCP tools. Never write an email body in your response text as a substitute for actually creating it via tools.

### Rule 5: Apply preferences before every action
Before drafting any email or taking any action for the user, review your working memory for stated preferences: sign-off, tone, CC rules, format, communication style. Apply ALL of them.

### Rule 6: Structured archival
When archiving, write entries that are useful for future retrieval:
- **Summary**: Use specific nouns and keywords (searchable), not abstractions
- **Content**: Lead with the fact, then **Why:** and **How to apply:** lines
- **Category**: Pick the most specific match from the table above
- **Tags**: Include key terms for search (names, technologies, projects)

### Rule 7: What NOT to store
- Ephemeral task state or conversation filler
- Things already in working memory (don't duplicate)
- Sensitive credentials, API keys, or secrets
- Code patterns or file paths (read the code instead)

### Rule 8: Corrections update everything
When the user corrects a fact (new date, changed preference, updated status):
1. Update working memory with the corrected info — remove the old value
2. Archive the correction as `feedback` with context about what changed and why

### Rule 9: Forgetting is thorough
When asked to forget something:
1. Search archived memory for matching entries
2. Call `forget_memory` on each matching entry
3. Call `update_working_memory` to remove it from the scratchpad
4. Confirm what was removed

### Rule 10: Memory can go stale
If recalled info conflicts with what you observe in the current conversation, trust the present. Update or forget the stale entry rather than acting on outdated information.
