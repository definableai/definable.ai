"""Code executor for running scripts bundled with Agent Skills.

Runs Python, bash, and Node.js scripts from a skill's ``scripts/``
directory in a subprocess with timeout, working directory control,
and output capture.

Usage::

    from definable.agent.skill.executor import SkillScriptExecutor
    from definable.agent.skill.markdown import SkillLoader

    skill = SkillLoader.load_skill_directory(Path("skills/pdf"))
    executor = SkillScriptExecutor(timeout=30)
    output = await executor.run(skill, "scripts/extract.py", ["input.pdf"])
"""

import asyncio
import sys
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
  from definable.agent.skill.markdown import MarkdownSkill

# Extension → interpreter mapping
_INTERPRETERS: dict[str, list[str]] = {
  ".py": [sys.executable],
  ".sh": ["bash"],
  ".bash": ["bash"],
  ".js": ["node"],
}


class SkillScriptExecutor:
  """Execute scripts bundled with Agent Skills.

  Runs Python and bash scripts from a skill's scripts/ directory
  in a subprocess with timeout, working directory, and output capture.

  Args:
    timeout: Maximum execution time in seconds (default 30).
    max_output: Maximum combined stdout+stderr characters (default 10000).
  """

  def __init__(self, *, timeout: int = 30, max_output: int = 10000):
    self._timeout = timeout
    self._max_output = max_output

  async def run(
    self,
    skill: "MarkdownSkill",
    script_path: str,
    args: Optional[List[str]] = None,
  ) -> str:
    """Run a script from a skill directory.

    Args:
      skill: The skill containing the script.
      script_path: Relative path to the script (e.g. "scripts/extract.py").
      args: Optional command-line arguments.

    Returns:
      Combined stdout+stderr output (truncated to max_output).
    """
    if skill.skill_directory is None:
      return f"Error: Skill '{skill.meta.name}' is not a directory skill"

    skill_root = skill.skill_directory.resolve()
    target = (skill_root / script_path).resolve()

    # Security: path-escape check
    if not str(target).startswith(str(skill_root)):
      return f"Error: Script path '{script_path}' escapes skill directory"

    if not target.is_file():
      return f"Error: Script not found: {script_path}"

    # Determine interpreter
    ext = target.suffix.lower()
    interpreter = _INTERPRETERS.get(ext)
    if interpreter is None:
      return f"Error: Unsupported script type '{ext}'. Supported: {', '.join(_INTERPRETERS.keys())}"

    cmd = [*interpreter, str(target), *(args or [])]

    try:
      proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(skill_root),
      )
      stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
    except asyncio.TimeoutError:
      try:
        proc.kill()  # type: ignore[union-attr]
        await proc.wait()  # type: ignore[union-attr]
      except Exception:
        pass
      return f"Error: Script timed out after {self._timeout}s"
    except Exception as e:
      return f"Error: Failed to run script: {e}"

    output = ""
    if stdout:
      output += stdout.decode("utf-8", errors="replace")
    if stderr:
      output += stderr.decode("utf-8", errors="replace")

    if proc.returncode != 0:
      output = f"[exit code {proc.returncode}]\n{output}"

    # Truncate to max_output
    if len(output) > self._max_output:
      output = output[: self._max_output] + f"\n... (truncated at {self._max_output} chars)"

    return output
