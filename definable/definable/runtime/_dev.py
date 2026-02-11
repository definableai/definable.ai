"""Dev mode — hot-reload watcher using watchfiles.

When ``agent.serve(dev=True)`` is called, the parent process uses
``watchfiles.run_process`` to re-execute the user's script whenever
``.py`` files change.  The child process is identified by the
``_DEFINABLE_DEV_CHILD`` environment variable.
"""

from __future__ import annotations

import os
import runpy
import sys

from definable.utils.log import log_info, log_warning

_ENV_KEY = "_DEFINABLE_DEV_CHILD"


def is_dev_child() -> bool:
  """Return True if running inside a dev-mode child process."""
  return os.environ.get(_ENV_KEY) == "1"


def _dev_target(script_path: str) -> None:
  """Execute the user script in a child process.

  Must be a module-level function so it is picklable for
  ``multiprocessing.spawn``.
  """
  runpy.run_path(script_path, run_name="__main__")


def _on_reload(changes: set) -> None:  # type: ignore[type-arg]
  """Log changed files on each reload cycle."""
  paths = [str(c[1]) for c in changes]
  log_info(f"[dev] Reloading — changed files: {', '.join(paths)}")


def run_dev_mode(watch_dir: str | None = None) -> None:
  """Start the dev-mode watcher (parent process).

  Resolves the calling script from ``sys.argv[0]``, sets
  ``_DEFINABLE_DEV_CHILD=1`` in the environment, and delegates to
  ``watchfiles.run_process`` which spawns a child that re-runs the
  script on every ``.py`` file change.

  Args:
    watch_dir: Directory to watch. Defaults to the parent directory
      of the calling script.

  Raises:
    ImportError: If ``watchfiles`` is not installed.
    RuntimeError: If the calling script cannot be resolved.
  """
  try:
    from watchfiles import PythonFilter, run_process
  except ImportError as e:
    raise ImportError("watchfiles is required for dev mode. Install it with: pip install 'definable-ai[serve]'") from e

  script = os.path.abspath(sys.argv[0])
  if not os.path.isfile(script):
    raise RuntimeError(f"Cannot resolve script path for dev mode: {sys.argv[0]!r}")

  if watch_dir is None:
    watch_dir = os.path.dirname(script)

  os.environ[_ENV_KEY] = "1"

  log_info(f"[dev] Watching for changes in {watch_dir}")
  log_info(f"[dev] Script: {script}")
  log_warning("[dev] Press Ctrl+C to stop")

  run_process(
    watch_dir,
    target=_dev_target,
    args=(script,),
    callback=_on_reload,
    watch_filter=PythonFilter(),
  )
