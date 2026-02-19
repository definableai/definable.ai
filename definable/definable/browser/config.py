"""BrowserConfig — frozen configuration dataclass for BrowserToolkit (SeleniumBase CDP)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BrowserConfig:
  """Configuration for BrowserToolkit (SeleniumBase CDP mode).

  Immutable once created. Follows the same frozen-dataclass convention as
  AgentConfig, Memory, Knowledge, and Thinking.

  SeleniumBase CDP mode drives Chrome directly via Chrome DevTools Protocol —
  no WebDriver, no automation fingerprint. It is the most stealth-capable
  browser automation approach available in Python.

  Connection modes (in priority order):
    1. ``host`` + ``port``: Attach to an already-running Chrome with
       ``--remote-debugging-port=<port>``. No new browser window is opened.
    2. ``user_data_dir``: Launch with a persistent user profile (cookies,
       localStorage, logged-in sessions are preserved between runs).
    3. Default: Launch a fresh Chrome instance (ephemeral).

  Attributes:
      headless: Run Chrome without a visible window. Default False — CDP mode
                works best headed for maximum stealth.
      user_data_dir: Path to a Chrome user data directory for session
                     persistence. Default None (ephemeral).
      proxy: Proxy server in ``"host:port"`` or ``"user:pass@host:port"``
             format. Default None.
      user_agent: Override the browser's User-Agent string. Default None.
      lang: Language locale code, e.g. ``"en"``, ``"fr"``, ``"zh-CN"``.
            Default ``"en"``.
      sandbox: Whether to enable Chrome's sandbox (default True).
               Set False only if Chrome refuses to launch in restricted envs.
      browser_executable_path: Path to a specific Chrome/Chromium binary.
                                Default None (SeleniumBase finds Chrome).
      browser_args: Extra Chrome CLI args passed at launch. Default None.
      host: Remote-debugging host to attach to an existing Chrome session.
            Must be combined with ``port``. Default None.
      port: Remote-debugging port to attach to an existing Chrome session.
            Must be combined with ``host``. Default None.
      incognito: Launch in incognito mode. Default False.
      mobile: Enable mobile emulation mode. Default False.
      ad_block: Enable SeleniumBase's built-in ad blocker. Default False.
      timeout: Default per-operation timeout in seconds. Default 30.0.

  Examples::

      # Fresh stealth Chrome (most common)
      config = BrowserConfig(headless=False)

      # Persistent profile — stays logged in between runs
      config = BrowserConfig(user_data_dir="/tmp/my-profile")

      # Attach to your running Chrome
      # (launch Chrome with: --remote-debugging-port=9222 --no-first-run)
      config = BrowserConfig(host="127.0.0.1", port=9222)

      # With proxy
      config = BrowserConfig(proxy="user:pass@proxy.example.com:8080")
  """

  headless: bool = False
  user_data_dir: Optional[str] = None
  proxy: Optional[str] = None
  user_agent: Optional[str] = None
  lang: str = "en"
  sandbox: bool = True
  browser_executable_path: Optional[str] = None
  browser_args: Optional[tuple[str, ...]] = None
  host: Optional[str] = None
  port: Optional[int] = None
  incognito: bool = False
  mobile: bool = False
  ad_block: bool = False
  timeout: float = 30.0
