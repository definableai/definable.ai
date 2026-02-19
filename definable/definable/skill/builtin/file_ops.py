"""File operations skill — read, write, and list local files.

Gives the agent the ability to work with files on the local filesystem.
All operations are restricted to a configurable base directory for safety.

Example:
    from definable.skill.builtin import FileOperations

    agent = Agent(
        model=model,
        skills=[FileOperations(base_dir="./workspace")],
    )
    output = agent.run("Read the config.json file and summarize it.")
"""

from pathlib import Path
from typing import Optional

from definable.skill.base import Skill
from definable.tool.decorator import tool


class FileOperations(Skill):
  """Skill for reading, writing, and listing files.

  All file operations are sandboxed to a ``base_dir`` for safety.
  The agent cannot access files outside this directory.

  Args:
    base_dir: Root directory for all file operations.
        Defaults to the current working directory.
    allow_write: Whether to enable file writing (default: True).
    max_read_size: Maximum file size to read in bytes (default: 1MB).

  Example:
    agent = Agent(
        model=model,
        skills=[FileOperations(base_dir="./data", allow_write=False)],
    )
  """

  name = "file_operations"

  def __init__(
    self,
    *,
    base_dir: Optional[str] = None,
    allow_write: bool = True,
    max_read_size: int = 1_048_576,  # 1MB
  ):
    super().__init__()
    self._base_dir = Path(base_dir or ".").resolve()
    self._allow_write = allow_write
    self._max_read_size = max_read_size

  @property
  def instructions(self) -> str:  # type: ignore[override]
    write_note = "You can also write and append to files. " if self._allow_write else "File writing is disabled — you can only read. "
    return (
      f"You have access to file operation tools. All paths are relative to "
      f"the base directory: {self._base_dir}. "
      f"Use read_file to examine file contents, list_files to see what's available. "
      f"{write_note}"
      f"Always confirm with the user before overwriting existing files."
    )

  def _resolve_safe(self, path: str) -> Path:
    """Resolve a path safely within base_dir."""
    resolved = (self._base_dir / path).resolve()
    if not str(resolved).startswith(str(self._base_dir)):
      raise PermissionError(f"Path escapes base directory: {path}")
    return resolved

  @property
  def tools(self) -> list:
    skill = self  # capture for closures

    @tool
    def read_file(path: str) -> str:
      """Read the contents of a text file.

      Args:
        path: Relative path to the file within the workspace.

      Returns:
        The file contents as text, or an error message.
      """
      try:
        resolved = skill._resolve_safe(path)
        if not resolved.exists():
          return f"Error: File not found: {path}"
        if not resolved.is_file():
          return f"Error: Not a file: {path}"
        size = resolved.stat().st_size
        if size > skill._max_read_size:
          return f"Error: File too large ({size:,} bytes). Maximum is {skill._max_read_size:,} bytes."
        return resolved.read_text(encoding="utf-8", errors="replace")
      except PermissionError as e:
        return f"Error: {e}"
      except Exception as e:
        return f"Error reading file: {e}"

    @tool
    def list_files(path: str = ".") -> str:
      """List files and directories at the given path.

      Args:
        path: Relative directory path. Defaults to the workspace root.

      Returns:
        A formatted listing of files and directories with sizes.
      """
      try:
        resolved = skill._resolve_safe(path)
        if not resolved.exists():
          return f"Error: Directory not found: {path}"
        if not resolved.is_dir():
          return f"Error: Not a directory: {path}"

        entries = sorted(resolved.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines = []
        for entry in entries:
          rel = entry.relative_to(skill._base_dir)
          if entry.is_dir():
            count = sum(1 for _ in entry.iterdir())
            lines.append(f"📁 {rel}/  ({count} items)")
          else:
            size = entry.stat().st_size
            if size < 1024:
              size_str = f"{size} B"
            elif size < 1024 * 1024:
              size_str = f"{size / 1024:.1f} KB"
            else:
              size_str = f"{size / (1024 * 1024):.1f} MB"
            lines.append(f"📄 {rel}  ({size_str})")

        if not lines:
          return "Directory is empty."
        return "\n".join(lines)
      except PermissionError as e:
        return f"Error: {e}"
      except Exception as e:
        return f"Error listing files: {e}"

    result = [read_file, list_files]

    if skill._allow_write:

      @tool
      def write_file(path: str, content: str) -> str:
        """Write content to a file, creating it if it doesn't exist.

        Args:
          path: Relative path for the file within the workspace.
          content: The text content to write.

        Returns:
          Confirmation message or error.
        """
        try:
          resolved = skill._resolve_safe(path)
          resolved.parent.mkdir(parents=True, exist_ok=True)
          resolved.write_text(content, encoding="utf-8")
          return f"Successfully wrote {len(content):,} characters to {path}"
        except PermissionError as e:
          return f"Error: {e}"
        except Exception as e:
          return f"Error writing file: {e}"

      @tool
      def append_to_file(path: str, content: str) -> str:
        """Append content to an existing file.

        Args:
          path: Relative path to the file.
          content: The text content to append.

        Returns:
          Confirmation message or error.
        """
        try:
          resolved = skill._resolve_safe(path)
          if not resolved.exists():
            return f"Error: File not found: {path}. Use write_file to create it."
          with open(resolved, "a", encoding="utf-8") as f:
            f.write(content)
          return f"Successfully appended {len(content):,} characters to {path}"
        except PermissionError as e:
          return f"Error: {e}"
        except Exception as e:
          return f"Error appending to file: {e}"

      result.extend([write_file, append_to_file])

    return result
