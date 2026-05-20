"""Python execution skill — run Python code in a controlled environment."""

from __future__ import annotations

import json
import subprocess
import sys
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Optional

from definable.agent.skill.base import Skill
from definable.agent.toolkit.decorator import tool
from definable.utils.log import log_warning


class PythonExec(Skill):
  """Execute Python code and scripts in a controlled environment.

  WARNING: This skill allows arbitrary code execution. Use with human
  supervision and appropriate security measures.

  Args:
      base_dir: Working directory for file operations. Default current dir.
      restrict_to_base_dir: Prevent file operations outside base_dir. Default True.
      timeout: Maximum execution time in seconds. Default 30.
      enable_file_ops: Enable save/run file tools. Default True.
      enable_pip: Enable pip install tool. Default False.

  Example::

      from definable.agent.skill.builtin import PythonExec
      agent = Agent(model=model, skills=[PythonExec(base_dir="./workspace")])
  """

  name = "python_exec"
  instructions = (
    "You can execute Python code using run_python. The code runs in an isolated namespace. "
    "Use variable_to_return to get a specific variable's value back. "
    "Standard output (print statements) is captured and returned."
  )

  def __init__(
    self,
    *,
    base_dir: Optional[str] = None,
    restrict_to_base_dir: bool = True,
    timeout: int = 30,
    enable_file_ops: bool = True,
    enable_pip: bool = False,
  ):
    super().__init__()
    self._base_dir = Path(base_dir or ".").resolve()
    self._restrict = restrict_to_base_dir
    self._timeout = timeout
    self._enable_file_ops = enable_file_ops
    self._enable_pip = enable_pip
    log_warning("PythonExec skill loaded — allows arbitrary code execution. Ensure human supervision.")

  def _check_path(self, path: str) -> Path:
    resolved = (self._base_dir / path).resolve()
    if self._restrict and not str(resolved).startswith(str(self._base_dir)):
      raise PermissionError(f"Path escapes base directory: {path}")
    return resolved

  @property
  def tools(self) -> list:
    skill = self
    result: list = []

    @tool
    def run_python(code: str, variable_to_return: str = "") -> str:
      """Execute Python code and return output. Optionally return a specific variable's value."""
      try:
        namespace: Dict[str, Any] = {}
        old_stdout = sys.stdout
        sys.stdout = captured = StringIO()
        try:
          exec(code, namespace)  # noqa: S102
        finally:
          sys.stdout = old_stdout

        stdout = captured.getvalue()

        if variable_to_return and variable_to_return in namespace:
          value = namespace[variable_to_return]
          return json.dumps({"stdout": stdout, "result": str(value)}, default=str)
        return json.dumps({"stdout": stdout}, default=str) if stdout else json.dumps({"status": "ok"})
      except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})

    result.append(run_python)

    if self._enable_file_ops:

      @tool
      def save_and_run(filename: str, code: str, variable_to_return: str = "") -> str:
        """Save Python code to a file and execute it."""
        try:
          path = skill._check_path(filename)
          path.write_text(code, encoding="utf-8")
          proc = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=skill._timeout,
            cwd=str(skill._base_dir),
          )
          output = proc.stdout
          if proc.returncode != 0:
            output += f"\nSTDERR:\n{proc.stderr}"
          return json.dumps({"file": str(path), "returncode": proc.returncode, "output": output[:10000]})
        except subprocess.TimeoutExpired:
          return json.dumps({"error": f"Script timed out after {skill._timeout}s"})
        except PermissionError as e:
          return json.dumps({"error": str(e)})
        except Exception as e:
          return json.dumps({"error": str(e)})

      result.append(save_and_run)

    if self._enable_pip:

      @tool
      def pip_install(package: str) -> str:
        """Install a Python package via pip."""
        try:
          proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True,
            timeout=120,
          )
          if proc.returncode == 0:
            return json.dumps({"status": "ok", "package": package})
          return json.dumps({"error": proc.stderr[:5000]})
        except Exception as e:
          return json.dumps({"error": str(e)})

      result.append(pip_install)

    return result
