"""agent.memory — markdown-file memory.

Single primitive: `FileMemory(root)`. The agent reads and writes through
auto-generated tools (`read_memory`, `write_memory`, `list_memories`,
`search_memory`). No conversation history is persisted — events are the
source of truth, callers may export them externally.

Public surface::

    from definable.agent.memory import FileMemory, memory_tools, build_index
"""

from definable.agent.memory.auto_index import build_index
from definable.agent.memory.file import FileMemory
from definable.agent.memory.tools import memory_tools

__all__ = ["FileMemory", "build_index", "memory_tools"]
