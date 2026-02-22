"""Async REPL input handler for the CLI interface."""

from __future__ import annotations

import asyncio
import contextlib
import sys
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Optional

if TYPE_CHECKING:
  from prompt_toolkit.completion import Completer

  from definable.agent.interface.cli.config import CLIConfig


class InputHandler:
  """Non-blocking stdin reader with optional slash-command completions.

  When a ``completer`` is provided, input is read via ``prompt_toolkit``'s
  ``PromptSession.prompt_async`` — giving the user a dropdown, arrow-key
  navigation, and input history for free.

  When ``completer`` is ``None`` (disabled or import failure), falls back
  to ``run_in_executor(input)`` with zero behavior change.

  Args:
    config: CLI configuration (for prompt string).
    on_input: Async callback invoked with each line of user input.
    on_eof: Async callback invoked when stdin reaches EOF (Ctrl+D).
    completer: Optional prompt_toolkit ``Completer`` for slash commands.
  """

  def __init__(
    self,
    *,
    config: "CLIConfig",
    on_input: Callable[[str], Coroutine[Any, Any, None]],
    on_eof: Callable[[], Coroutine[Any, Any, None]],
    completer: Optional["Completer"] = None,
  ) -> None:
    self._config = config
    self._on_input = on_input
    self._on_eof = on_eof
    self._completer = completer
    self._task: Optional[asyncio.Task[None]] = None
    self._running = False
    self._session: Any = None  # lazily created PromptSession

  async def start(self) -> None:
    """Start the input loop as a background task."""
    self._running = True
    self._task = asyncio.create_task(self._loop())

  async def stop(self) -> None:
    """Cancel the input loop."""
    self._running = False
    if self._task is not None:
      self._task.cancel()
      with contextlib.suppress(asyncio.CancelledError):
        await self._task
      self._task = None

  def _get_session(self) -> Any:
    """Lazily create a prompt_toolkit PromptSession."""
    if self._session is None:
      from prompt_toolkit import PromptSession
      from prompt_toolkit.history import InMemoryHistory

      self._session = PromptSession(
        completer=self._completer,
        history=InMemoryHistory(),
        complete_while_typing=True,
      )
    return self._session

  async def _loop(self) -> None:
    """Main REPL loop — reads stdin via prompt_toolkit or executor."""
    while self._running:
      try:
        line = await self._read_line_async()
      except EOFError:
        await self._on_eof()
        break
      except asyncio.CancelledError:
        break

      if line is None:
        await self._on_eof()
        break

      await self._on_input(line)

  async def _read_line_async(self) -> Optional[str]:
    """Read a line of input asynchronously.

    Uses prompt_toolkit when a completer is available, otherwise
    falls back to ``run_in_executor(input)``.
    """
    if self._completer is not None:
      try:
        session = self._get_session()
        return await session.prompt_async(self._config.prompt)
      except KeyboardInterrupt:
        print("", file=sys.stderr)
        raise EOFError
    else:
      return await self._read_line_blocking()

  async def _read_line_blocking(self) -> Optional[str]:
    """Blocking read from stdin via executor. Returns None on EOF."""
    loop = asyncio.get_running_loop()
    try:
      return await loop.run_in_executor(None, lambda: input(self._config.prompt))
    except EOFError:
      return None
    except KeyboardInterrupt:
      print("", file=sys.stderr)
      raise EOFError
