"""Shell skill — execute shell commands in a sandboxed environment.

Gives the agent the ability to run shell commands for tasks like
file manipulation, system info gathering, and script execution.
Includes configurable safety controls.

⚠️ SECURITY: This skill executes real shell commands. Always configure
``allowed_commands`` or ``blocked_commands`` in production.

Example:
    from definable.skill.builtin import Shell

    agent = Agent(
        model=model,
        skills=[Shell(allowed_commands={"ls", "cat", "grep", "wc", "head", "tail"})],
    )
    output = agent.run("How many Python files are in the current directory?")
"""

import subprocess
from typing import Optional, Set

from definable.skill.base import Skill
from definable.tool.decorator import tool


class Shell(Skill):
  """Skill for executing shell commands with safety controls.

  Provides a ``run_command`` tool that executes shell commands and
  returns their output. Includes configurable restrictions for
  safe operation.

  Args:
    allowed_commands: If set, only these command names are permitted
        (e.g. ``{"ls", "cat", "grep"}``). Checked against the first
        word of the command.
    blocked_commands: Commands that are always blocked (e.g. ``{"rm", "sudo"}``).
        Default blocks dangerous commands.
    timeout: Maximum execution time in seconds (default: 30).
    max_output_size: Maximum output size in characters (default: 10000).
    working_dir: Working directory for commands (default: current dir).

  Example:
    # Read-only shell access
    agent = Agent(
        model=model,
        skills=[Shell(allowed_commands={"ls", "cat", "grep", "find", "wc"})],
    )

    # Unrestricted (use with caution)
    agent = Agent(model=model, skills=[Shell()])
  """

  name = "shell"

  _DEFAULT_BLOCKED = {
    "rm",
    "rmdir",
    "mkfs",
    "dd",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "init",
    "systemctl",
    "chmod",
    "chown",
    "chgrp",
    "sudo",
    "su",
    "passwd",
    "kill",
    "killall",
    "pkill",
    "iptables",
    "ufw",
    "format",
    "fdisk",
    "parted",
  }

  def __init__(
    self,
    *,
    allowed_commands: Optional[Set[str]] = None,
    blocked_commands: Optional[Set[str]] = None,
    timeout: int = 30,
    max_output_size: int = 10000,
    working_dir: Optional[str] = None,
  ):
    super().__init__()
    self._allowed_commands = allowed_commands
    self._blocked_commands = blocked_commands if blocked_commands is not None else self._DEFAULT_BLOCKED
    self._timeout = timeout
    self._max_output_size = max_output_size
    self._working_dir = working_dir

  @property
  def instructions(self) -> str:  # type: ignore[override]
    parts = ["You have access to a shell command tool for executing system commands."]
    if self._allowed_commands:
      parts.append(f"Allowed commands: {', '.join(sorted(self._allowed_commands))}.")
    if self._blocked_commands:
      parts.append("Some dangerous commands are blocked for safety.")
    parts.append("Always explain what a command does before running it. Prefer simple, composable commands over complex one-liners.")
    return " ".join(parts)

  def _validate_command(self, command: str) -> Optional[str]:
    """Validate a command against safety rules. Returns error or None."""
    cmd_parts = command.strip().split()
    if not cmd_parts:
      return "Error: Empty command."

    base_cmd = cmd_parts[0]

    # Handle pipes and chains — check all commands
    for separator in ["|", "&&", "||", ";"]:
      if separator in command:
        segments = command.split(separator)
        for segment in segments:
          segment = segment.strip()
          if segment:
            err = self._validate_command(segment)
            if err:
              return err
        return None

    if self._allowed_commands and base_cmd not in self._allowed_commands:
      return f"Error: Command '{base_cmd}' is not in the allowed list: {sorted(self._allowed_commands)}"

    if base_cmd in self._blocked_commands:
      return f"Error: Command '{base_cmd}' is blocked for safety."

    return None

  @property
  def tools(self) -> list:
    skill = self

    @tool
    def run_command(command: str) -> str:
      """Execute a shell command and return its output.

      Args:
        command: The shell command to execute (e.g. "ls -la", "grep -r 'TODO' .").

      Returns:
        The command's stdout output, or stderr if the command failed.
      """
      validation_error = skill._validate_command(command)
      if validation_error:
        return validation_error

      try:
        result = subprocess.run(
          command,
          shell=True,
          capture_output=True,
          text=True,
          timeout=skill._timeout,
          cwd=skill._working_dir,
        )

        output = result.stdout
        if result.returncode != 0:
          output = f"[Exit code: {result.returncode}]\n"
          if result.stderr:
            output += f"stderr: {result.stderr}\n"
          if result.stdout:
            output += f"stdout: {result.stdout}"

        # Truncate large output
        if len(output) > skill._max_output_size:
          output = output[: skill._max_output_size] + "\n... [truncated]"

        return output or "(no output)"

      except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {skill._timeout} seconds."
      except Exception as e:
        return f"Error executing command: {e}"

    return [run_command]
