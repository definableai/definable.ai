"""System prompt injection for the tool-based memory system."""


def build_working_memory_block(user_id: str, content: str, updated_at: str = "") -> str:
  """Build the <working_memory> XML block for system prompt injection."""
  ts = f' updated_at="{updated_at}"' if updated_at else ""
  if content:
    return f'<working_memory user_id="{user_id}"{ts}>\n{content}\n</working_memory>'
  return f'<working_memory user_id="{user_id}"{ts}>\n(empty — update this when you learn about the user)\n</working_memory>'
