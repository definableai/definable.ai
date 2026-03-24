"""System prompt injection for the tool-based memory system."""

MEMORY_INSTRUCTIONS = """## Memory management

You have a two-tier memory system for each user:

**Working memory** (visible above in <working_memory> tags) contains critical context
about the current user. You manage it via the `update_working_memory` tool.
Keep it concise — focus on what matters for the current conversation.

**Archived memory** stores everything else. Access it via `recall_memory`
(search summaries) then `fetch_memory_entries` (load full details).

### Memory categories

| Category | What to store | When to save |
|----------|--------------|-------------|
| `user` | Identity, role, expertise, preferences | User shares personal/professional info |
| `feedback` | How to approach work — corrections AND confirmations | User corrects you OR validates a non-obvious approach |
| `project` | Goals, decisions, context not derivable from code | User shares project/work context |
| `reference` | Pointers to external systems and resources | User mentions external tools/docs/URLs |
| `conversation` | Key takeaways from discussions | Archiving from working memory overflow |

### When to update working memory
- User shares identity, role, or preferences → update immediately
- Goal changes or new constraint stated → update immediately
- An item becomes irrelevant → remove it
- Working memory overflows → archive lower-priority items first, then trim

### When to archive
- Evicting items from working memory to make room
- User corrects your approach (save as `feedback` — include WHY)
- User confirms a non-obvious approach worked (also `feedback`)
- Conversation produced a reusable insight
- User explicitly asks you to remember something

### When to recall
- User references past context ("remember when we...")
- You need info not in working memory to answer well
- User asks about their own history or preferences

### What NOT to store
- Code patterns, file paths, or project structure (read the code instead)
- Ephemeral task state or conversation history (that's what message context is for)
- Sensitive credentials, API keys, or secrets
- Things derivable from the current conversation context

### Archived entry structure
Lead with the fact or rule, then:
- **Why:** the reason or motivation behind it
- **How to apply:** when this should shape your behavior

Example: "Prefers composition over inheritance. **Why:** believes deep class
hierarchies are unreadable. **How to apply:** when designing new code, use
protocols and composition, not ABC subclassing."

### Rules
- Working memory is your scratchpad. Rewrite it fully each time (not a diff).
- Archived entries are immutable. Create new ones, don't update old.
- Recalled entries are temporary — they won't persist to the next turn.
- Memory can become stale. If recalled info conflicts with what you observe
  now, trust the present and update/forget the stale entry."""


def build_working_memory_block(user_id: str, content: str, updated_at: str = "") -> str:
  """Build the <working_memory> XML block for system prompt injection."""
  ts = f' updated_at="{updated_at}"' if updated_at else ""
  if content:
    return f'<working_memory user_id="{user_id}"{ts}>\n{content}\n</working_memory>'
  return f'<working_memory user_id="{user_id}"{ts}>\n(empty — update this when you learn about the user)\n</working_memory>'
