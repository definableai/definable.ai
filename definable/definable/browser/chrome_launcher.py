"""Chrome subprocess management for Playwright CDP connections.

Handles finding Chrome executables, launching with stealth flags,
polling for CDP readiness, and clean shutdown. Ported from openclaw chrome.ts.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from definable.utils.log import log_debug, log_info, log_warning

if TYPE_CHECKING:
  from definable.browser.config import BrowserConfig


@dataclass
class RunningChrome:
  """Represents a running Chrome process with CDP enabled."""

  pid: int
  process: subprocess.Popen  # type: ignore[type-arg]
  cdp_port: int
  cdp_url: str
  user_data_dir: str
  _temp_dir: str | None = field(default=None, repr=False)


# -- Executable discovery -------------------------------------------------

_MACOS_PATHS = (
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
  "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
  "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
  "/Applications/Chromium.app/Contents/MacOS/Chromium",
)

_LINUX_NAMES = (
  "google-chrome",
  "google-chrome-stable",
  "chromium-browser",
  "chromium",
  "brave-browser",
  "microsoft-edge",
)


async def find_chrome_executable() -> str:
  """Find a Chrome-compatible executable on the current platform.

  Raises:
      FileNotFoundError: If no suitable browser is found.
  """
  system = platform.system()

  if system == "Darwin":
    for path in _MACOS_PATHS:
      if os.path.isfile(path):
        return path
    # Fallback to which
    found = shutil.which("google-chrome") or shutil.which("chromium")
    if found:
      return found

  elif system == "Linux":
    for name in _LINUX_NAMES:
      found = shutil.which(name)
      if found:
        return found

  elif system == "Windows":
    # Check common Windows paths
    program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
    program_files_x86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
    for base in (program_files, program_files_x86):
      chrome_path = os.path.join(base, "Google", "Chrome", "Application", "chrome.exe")
      if os.path.isfile(chrome_path):
        return chrome_path
    found = shutil.which("chrome") or shutil.which("chrome.exe")
    if found:
      return found

  raise FileNotFoundError(
    "No Chrome-compatible browser found. Install Chrome, Brave, or Edge, or set BrowserConfig(executable_path='/path/to/chrome')."
  )


# -- CDP readiness --------------------------------------------------------


async def is_cdp_ready(cdp_url: str, timeout: float = 0.5) -> bool:
  """Check if a Chrome CDP endpoint is responding."""
  import aiohttp

  http_url = cdp_url.replace("ws://", "http://").replace("wss://", "https://")
  if not http_url.startswith("http"):
    http_url = f"http://{http_url}"
  version_url = f"{http_url.rstrip('/')}/json/version"

  try:
    async with aiohttp.ClientSession() as session:
      async with session.get(version_url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
        return resp.status == 200
  except Exception:
    return False


async def get_chrome_ws_url(cdp_url: str, timeout: float = 0.5) -> str | None:
  """Get the WebSocket debugger URL from a Chrome CDP endpoint."""
  import aiohttp

  http_url = cdp_url.replace("ws://", "http://").replace("wss://", "https://")
  if not http_url.startswith("http"):
    http_url = f"http://{http_url}"
  version_url = f"{http_url.rstrip('/')}/json/version"

  try:
    async with aiohttp.ClientSession() as session:
      async with session.get(version_url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
        if resp.status != 200:
          return None
        data = await resp.json()
        ws_url = str(data.get("webSocketDebuggerUrl", "")).strip()
        return ws_url or None
  except Exception:
    return None


# -- Launch / Stop --------------------------------------------------------


async def launch_chrome(config: "BrowserConfig") -> RunningChrome:
  """Launch a Chrome process with CDP enabled and stealth flags.

  Waits for CDP to become ready (polls every 200ms for up to 15s).

  Raises:
      FileNotFoundError: If no Chrome executable is found.
      RuntimeError: If CDP doesn't come up within timeout.
  """
  exe = config.executable_path or await find_chrome_executable()

  # User data directory
  temp_dir: str | None = None
  if config.user_data_dir:
    user_data_dir = config.user_data_dir
    os.makedirs(user_data_dir, exist_ok=True)
  else:
    temp_dir = tempfile.mkdtemp(prefix="definable-browser-")
    user_data_dir = temp_dir

  cdp_port = config.cdp_port

  # Build Chrome CLI args (from openclaw chrome.ts)
  args: list[str] = [
    exe,
    f"--remote-debugging-port={cdp_port}",
    f"--user-data-dir={user_data_dir}",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-sync",
    "--disable-background-networking",
    "--disable-component-update",
    "--disable-features=Translate,MediaRouter",
    "--disable-session-crashed-bubble",
    "--hide-crash-restore-bubble",
    "--password-store=basic",
    "--disable-infobars",
    "--noerrdialogs",
  ]

  if config.headless:
    args.append("--headless=new")
    args.append("--disable-gpu")

  if config.no_sandbox:
    args.append("--no-sandbox")
    args.append("--disable-setuid-sandbox")

  if platform.system() == "Linux":
    args.append("--disable-dev-shm-usage")

  # Stealth: hide navigator.webdriver
  if config.stealth:
    args.append("--disable-blink-features=AutomationControlled")

  if config.proxy:
    args.append(f"--proxy-server={config.proxy}")

  if config.user_agent:
    args.append(f"--user-agent={config.user_agent}")

  # User-provided extra args
  if config.extra_args:
    args.extend(config.extra_args)

  # Open about:blank on fresh profiles to ensure a CDP target exists.
  # Skip for persistent profiles — Chrome will restore previous session tabs.
  if not config.user_data_dir:
    args.append("about:blank")

  log_debug(f"Launching Chrome: {exe} on port {cdp_port}")
  proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

  # Wait for CDP to come up
  cdp_url = f"http://127.0.0.1:{cdp_port}"
  deadline = asyncio.get_event_loop().time() + 15.0
  while asyncio.get_event_loop().time() < deadline:
    if await is_cdp_ready(cdp_url, timeout=0.5):
      break
    await asyncio.sleep(0.2)
  else:
    # Timeout — kill the process
    with contextlib.suppress(Exception):
      proc.kill()
    raise RuntimeError(f"Failed to start Chrome CDP on port {cdp_port}. Chrome may not be installed or the port may be in use.")

  pid = proc.pid or -1
  log_info(f"Chrome started (pid {pid}) on 127.0.0.1:{cdp_port}")

  return RunningChrome(
    pid=pid,
    process=proc,
    cdp_port=cdp_port,
    cdp_url=cdp_url,
    user_data_dir=user_data_dir,
    _temp_dir=temp_dir,
  )


async def stop_chrome(running: RunningChrome, timeout: float = 2.5) -> None:
  """Stop a running Chrome process. SIGTERM first, then SIGKILL after timeout."""
  proc = running.process
  if proc.poll() is not None:
    return  # Already exited

  with contextlib.suppress(Exception):
    proc.terminate()

  # Poll for exit
  deadline = asyncio.get_event_loop().time() + timeout
  while asyncio.get_event_loop().time() < deadline:
    if proc.poll() is not None:
      break
    await asyncio.sleep(0.1)

  # Force kill if still running
  if proc.poll() is None:
    with contextlib.suppress(Exception):
      proc.kill()
    log_warning(f"Chrome (pid {running.pid}) did not exit gracefully, sent SIGKILL")

  # Clean up temp directory
  if running._temp_dir:
    try:
      import shutil as _shutil

      _shutil.rmtree(running._temp_dir, ignore_errors=True)
    except Exception:
      pass
