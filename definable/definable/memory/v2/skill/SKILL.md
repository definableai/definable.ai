---
name: memory-manager
description: Manages two-tier user memory — working memory (always loaded) and archived memory (indexed, on-demand). Instructs the agent on when and how to store, recall, and forget information.
version: 2.0.0
tags: [memory, recall, archive, preferences, context]
---

## Memory System

You have a two-tier memory system for each user:

**Working memory** (visible in `<working_memory>` tags above) is your scratchpad — always loaded into every turn. It must contain ALL active facts about the current user, organized by section. This is your primary source of truth.

**Archived memory** stores detail, history, and context that doesn't need to be in every prompt. Access it via `recall_memory` (search) -> `fetch_memory_entries` (load full content).

## Memory Categories

| Category | What to store | When to save |
|----------|--------------|-------------|
| `user` | Identity, role, expertise, preferences, contacts | User shares personal/professional info |
| `feedback` | How to approach work — corrections AND confirmations | User corrects you OR validates a non-obvious approach |
| `project` | Goals, decisions, deadlines, risks, status | User shares project/work context |
| `reference` | Pointers to external systems, URLs, tools | User mentions external resources |
| `conversation` | Key takeaways worth preserving | Archiving overflow from working memory |

## Critical Rules

### Rule 1: Working memory uses a fixed section template

Working memory MUST follow this section structure. When updating, copy ALL existing sections unchanged, then modify only the section that changed. **Never drop a section.** Never rewrite WM with only the latest fact.

```
## Identity
Name, role, company, location, background

## Preferences
Communication style, code style, tools, dislikes

## Projects
Active projects, deadlines, tech stack, goals

## Team
Names, roles, relationships

## Other
Anything that doesn't fit above
```

**Example of CORRECT update** (user says "I got promoted to Staff"):
- Read current WM -> see all 5 sections populated
- Change ONLY the role in `## Identity` from "Senior" to "Staff"
- Rewrite WM with all 5 sections, everything else unchanged

**Example of WRONG update** (what you must NEVER do):
- Replace entire WM with just: "Alice got promoted to Staff Data Scientist"
- This destroys all other facts. NEVER do this.

### Rule 2: Always search before saying "I don't know"

If the user asks a question and the answer is NOT in your working memory, you MUST call `recall_memory` to search archived memory before responding. Never say "I don't have that information" without searching first.

If your first search returns nothing, try alternative terms. Search "education university degree" not just "MIT". Cast a wide net.

### Rule 3: Archive AND update working memory

When the user shares important information, do BOTH:
1. Call `update_working_memory` to add it to the scratchpad (preserving all existing sections)
2. Call `archive_to_memory` to preserve the detail long-term

Archive is not a substitute for working memory. Both serve different purposes.

### Rule 4: Use tools for actions, not text

When asked to send, draft, search, or check email — ALWAYS use the available email/MCP tools. Never write an email body in your response text as a substitute for actually creating it via tools.

### Rule 5: Apply preferences before every action

Before drafting any email or taking any action for the user, review your working memory `## Preferences` section. Apply ALL stated preferences: sign-off, tone, CC rules, format, communication style, code style.

### Rule 6: One fact per archive entry with searchable summaries

When archiving, follow these rules strictly:

1. **ONE fact per entry.** If the user shared 3 facts, make 3 separate `archive_to_memory` calls. Never bundle unrelated facts into one entry.

2. **Summary MUST contain the specific nouns and values.** The summary is what search indexes. A vague summary makes the fact invisible.

3. **Good test:** Could someone find this entry by searching for any key term in the fact?

**GOOD summaries:**
- "Alice Chen is a Staff Data Scientist at Stripe in San Francisco"
- "Alice graduated from MIT in 2016 with a CS degree"
- "Fraud detection project deadline is July 1st"
- "Team uses Polars instead of pandas for large datasets"

**BAD summaries (NEVER write these):**
- "Alice's identity" (too vague — searching "Stripe" won't find this)
- "Professional background" (no searchable nouns)
- "Technical preferences" (which preferences? what technology?)
- "Alice's professional activities and interests" (bundles everything, findable by nothing)

**Content structure:** Lead with the fact, then **Why:** and **How to apply:** lines.

**Tags:** Include every key noun for search: names, technologies, companies, places.

### Rule 7: What NOT to store

- Ephemeral task state or conversation filler
- Things already in working memory (don't duplicate in archive if already in WM, but DO archive for long-term preservation)
- Sensitive credentials, API keys, or secrets
- Code patterns or file paths (read the code instead)

### Rule 8: Corrections update everything

When the user corrects a fact (new date, changed preference, updated status):
1. Update the relevant section in working memory with the corrected info — remove the old value
2. Archive the correction as `feedback` with context about what changed and why

### Rule 9: Forgetting is thorough

When asked to forget something:
1. Search archived memory for matching entries
2. Call `forget_memory` on each matching entry
3. Call `update_working_memory` to remove it from the scratchpad
4. Confirm what was removed

### Rule 10: Memory can go stale

If recalled info conflicts with what you observe in the current conversation, trust the present. Update or forget the stale entry rather than acting on outdated information.

### Rule 11: Contradictions require cleanup

When the user corrects a fact or something changes (new role, new deadline, replaced technology, team member left):
1. Update working memory (Rule 8)
2. Archive the new fact
3. **Search for old entries about the same topic** using `recall_memory`
4. **Delete outdated entries** with `forget_memory` so they don't create confusion later

Example: User says "deadline moved to July 1st" -> archive new deadline -> search "deadline" -> find old entry saying "June 15th" -> `forget_memory` the old entry.
