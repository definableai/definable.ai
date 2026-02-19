"""SubprocessTransport — manages Claude Code CLI subprocess and JSONL communication.

Spawns `claude` CLI with stream-json format, handles bidirectional JSONL
protocol including control messages for hooks/permissions/tools.

No external dependencies — only stdlib + asyncio.
"""

import asyncio
import json
import os
import shutil
from typing import AsyncIterator, Dict, List, Optional

from definable.utils.log import log_debug, log_error, log_warning


class SubprocessTransport:
  """Manages Claude Code CLI subprocess and JSONL communication.

  Spawns `claude` CLI with stream-json IO, reads/writes newline-delimited JSON.
  Handles concurrent writes via an async lock, buffers partial JSONL lines,
  and provides a graceful shutdown sequence.
  """

  def __init__(
    self,
    cli_path: Optional[str] = None,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
  ):
    self._cli_path = cli_path or self._find_cli()
    self._cwd = cwd
    self._env = env or {}
    self._process: Optional[asyncio.subprocess.Process] = None
    self._write_lock = asyncio.Lock()
    self._buffer = ""
    self._max_buffer_size = 1_048_576  # 1 MB

  @staticmethod
  def _find_cli() -> str:
    """Find the `claude` CLI binary on PATH."""
    found = shutil.which("claude")
    if found:
      return found
    # Common global install locations
    for candidate in [
      os.path.expanduser("~/.npm-global/bin/claude"),
      "/usr/local/bin/claude",
    ]:
      if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return "claude"  # fallback — will fail at connect() if not found

  async def connect(self, args: List[str]) -> None:
    """Spawn CLI subprocess with given args.

    Args:
      args: CLI arguments (e.g. ["--output-format", "stream-json", ...]).

    Raises:
      FileNotFoundError: If the CLI binary is not found.
      RuntimeError: If the subprocess fails to start.
    """
    cmd = [self._cli_path] + args

    # Build environment: inherit current + user overrides + entrypoint marker.
    # Remove CLAUDECODE to avoid "nested session" detection when spawned
    # from within an existing Claude Code session.
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    env.update(self._env)
    env["CLAUDE_CODE_ENTRYPOINT"] = "definable"

    log_debug(f"Spawning Claude CLI: {' '.join(cmd)}")

    try:
      self._process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=self._cwd,
        env=env,
      )
    except FileNotFoundError:
      raise FileNotFoundError(f"Claude Code CLI not found at '{self._cli_path}'. Install with: npm install -g @anthropic-ai/claude-code")
    except Exception as exc:
      raise RuntimeError(f"Failed to start Claude Code CLI: {exc}") from exc

    self._buffer = ""
    log_debug(f"Claude CLI started (pid={self._process.pid})")

  async def send(self, message: dict) -> None:
    """Write a JSON message to stdin (newline-terminated).

    Thread-safe via async lock to prevent interleaved writes from
    concurrent tasks (e.g. parallel control responses).
    """
    if not self.is_running:
      raise RuntimeError("Transport is not connected")

    async with self._write_lock:
      data = json.dumps(message, separators=(",", ":")) + "\n"
      assert self._process is not None
      assert self._process.stdin is not None
      self._process.stdin.write(data.encode())
      await self._process.stdin.drain()

  async def receive(self) -> AsyncIterator[dict]:
    """Read JSONL messages from stdout. Yields parsed dicts.

    Buffers partial lines until a newline is seen, then attempts
    JSON parsing. Silently skips non-JSON lines (e.g. CLI warnings).
    """
    if not self.is_running:
      raise RuntimeError("Transport is not connected")

    assert self._process is not None
    assert self._process.stdout is not None

    while True:
      chunk = await self._process.stdout.read(8192)
      if not chunk:
        # EOF — process has closed stdout
        break

      self._buffer += chunk.decode("utf-8", errors="replace")

      # Guard against unbounded buffer growth
      if len(self._buffer) > self._max_buffer_size:
        log_warning("JSONL buffer exceeded max size, dropping oldest data")
        # Keep only the last portion
        self._buffer = self._buffer[-self._max_buffer_size // 2 :]

      # Process complete lines
      while "\n" in self._buffer:
        line, self._buffer = self._buffer.split("\n", 1)
        line = line.strip()
        if not line:
          continue
        try:
          yield json.loads(line)
        except json.JSONDecodeError:
          # CLI sometimes emits non-JSON text (warnings, progress)
          log_debug(f"Non-JSON CLI output: {line[:200]}")

  async def close(self) -> None:
    """Gracefully terminate subprocess.

    Shutdown sequence: close stdin → wait(5s) → terminate → kill.
    """
    if self._process is None:
      return

    pid = self._process.pid

    # 1. Close stdin to signal we're done
    if self._process.stdin and not self._process.stdin.is_closing():
      try:
        self._process.stdin.close()
        await self._process.stdin.wait_closed()
      except Exception:
        pass

    # 2. Wait for graceful exit
    if self._process.returncode is None:
      try:
        await asyncio.wait_for(self._process.wait(), timeout=5.0)
      except asyncio.TimeoutError:
        # 3. Terminate
        log_warning(f"Claude CLI (pid={pid}) did not exit in 5s, terminating")
        try:
          self._process.terminate()
          await asyncio.wait_for(self._process.wait(), timeout=3.0)
        except asyncio.TimeoutError:
          # 4. Kill as last resort
          log_error(f"Claude CLI (pid={pid}) did not terminate, killing")
          self._process.kill()
          await self._process.wait()

    # Drain stderr for debugging
    if self._process.stderr:
      try:
        stderr_data = await asyncio.wait_for(self._process.stderr.read(), timeout=1.0)
        if stderr_data:
          stderr_text = stderr_data.decode("utf-8", errors="replace").strip()
          if stderr_text:
            log_debug(f"Claude CLI stderr: {stderr_text[:500]}")
      except (asyncio.TimeoutError, Exception):
        pass

    rc = self._process.returncode
    self._process = None
    self._buffer = ""
    log_debug(f"Claude CLI (pid={pid}) exited with code {rc}")

  @property
  def is_running(self) -> bool:
    """Whether the subprocess is currently alive."""
    return self._process is not None and self._process.returncode is None
