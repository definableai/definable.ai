"""Memory tools — `@tool`-decorated functions bound to a FileMemory.

Built per-agent at construction time and registered with the ToolRegistry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from definable.agent.toolkit.decorator import tool

if TYPE_CHECKING:
  from definable.agent.memory.file import FileMemory
  from definable.agent.toolkit.function import Function


def memory_tools(memory: FileMemory) -> list[Function]:
  """Build the four memory tools bound to a FileMemory instance.

  Returned tools: `read_memory`, `write_memory`, `list_memories`,
  `search_memory`. Each closes over `memory` so multiple agents may keep
  independent stores.
  """

  @tool
  def read_memory(name: str) -> str:
    """Read a memory file by name. Returns the file contents or a
    'not found' message."""
    try:
      return memory.read(name)
    except FileNotFoundError:
      return f"No memory found: {name!r}"

  @tool
  def write_memory(name: str, content: str) -> str:
    """Save content to a memory file. Overwrites if the file exists.
    Use a short descriptive name (e.g. 'profile', 'preferences')."""
    memory.write(name, content)
    return f"Saved memory: {name}"

  @tool
  def list_memories() -> list[str]:
    """List every memory file's name."""
    return memory.names()

  @tool
  def search_memory(query: str) -> str:
    """Search memory files for substring matches. Returns matching
    files and snippets around each match."""
    results = memory.search(query)
    if not results:
      return f"No memories matched: {query!r}"
    return "\n\n".join(f"## {name}\n{snippet}" for name, snippet in results)

  return [read_memory, write_memory, list_memories, search_memory]
