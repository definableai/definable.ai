"""FileMemory — markdown-file memory store.

Each memory is a `.md` file under `root`. The agent uses the
`read_memory` / `write_memory` / `list_memories` / `search_memory` tools
to access it; nothing about the storage shape is hidden from the user.

If `root/INDEX.md` exists it is treated as the curated table of contents;
otherwise `index()` auto-derives one from each file's first line.
"""

from __future__ import annotations

from pathlib import Path


class FileMemory:
  """Markdown-file memory rooted at a directory."""

  def __init__(self, root: Path | str) -> None:
    self.root = Path(root)
    self.root.mkdir(parents=True, exist_ok=True)

  def read(self, name: str) -> str:
    path = self._path(name)
    if not path.exists():
      raise FileNotFoundError(f"No memory: {name!r}")
    return path.read_text(encoding="utf-8")

  def write(self, name: str, content: str) -> None:
    self._path(name).write_text(content, encoding="utf-8")

  def names(self) -> list[str]:
    """Return every memory file's stem (without `.md`), sorted."""
    return sorted(p.stem for p in self.root.glob("*.md"))

  def search(self, query: str, *, context: int = 80) -> list[tuple[str, str]]:
    """Substring search across all memory files.

    Returns `(name, snippet)` pairs where the snippet shows roughly
    `context` characters around the match.
    """
    if not query:
      return []
    needle = query.lower()
    results: list[tuple[str, str]] = []
    for path in sorted(self.root.glob("*.md")):
      text = path.read_text(encoding="utf-8")
      lower = text.lower()
      idx = lower.find(needle)
      if idx == -1:
        continue
      start = max(0, idx - context)
      end = min(len(text), idx + len(needle) + context)
      results.append((path.stem, text[start:end].strip()))
    return results

  def index(self) -> str:
    """Return INDEX.md if curated, else auto-build a one-line-per-file summary."""
    idx_path = self.root / "INDEX.md"
    if idx_path.exists():
      return idx_path.read_text(encoding="utf-8")
    from definable.agent.memory.auto_index import build_index

    return build_index(self)

  def _path(self, name: str) -> Path:
    """Resolve a name to a path. Reject anything that escapes `root`."""
    if not name or "/" in name or "\\" in name or name.startswith("."):
      raise ValueError(f"Invalid memory name: {name!r}")
    if not name.endswith(".md"):
      name = f"{name}.md"
    return self.root / name
