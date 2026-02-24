"""BrowserConfig — frozen configuration dataclass for BrowserToolkit (Playwright)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BrowserConfig:
  """Configuration for BrowserToolkit (Playwright CDP mode).

  Immutable once created. Follows the same frozen-dataclass convention as
  AgentConfig, Memory, Knowledge, and Thinking.

  Playwright connects to Chrome via Chrome DevTools Protocol (CDP), giving
  full control with native async support and no automation fingerprints.

  Connection modes (in priority order):
    1. ``cdp_url``: Connect to an already-running Chrome via WebSocket
       (e.g. ``ws://127.0.0.1:9222``).
    2. ``user_data_dir``: Launch with a persistent user profile (cookies,
       localStorage, logged-in sessions are preserved between runs).
    3. Default: Launch a fresh Chrome instance (ephemeral).

  Examples::

      # Fresh stealth Chrome (most common)
      config = BrowserConfig(headless=False)

      # Persistent profile — stays logged in between runs
      config = BrowserConfig(user_data_dir="/tmp/my-profile")

      # Attach to your running Chrome
      # (launch Chrome with: --remote-debugging-port=9222 --no-first-run)
      config = BrowserConfig(cdp_url="ws://127.0.0.1:9222")

      # CI / Docker
      config = BrowserConfig(headless=True, no_sandbox=True)

      # With proxy
      config = BrowserConfig(proxy="user:pass@proxy.example.com:8080")
  """

  # Connection (priority: cdp_url > user_data_dir > fresh launch)
  cdp_url: Optional[str] = None
  user_data_dir: Optional[str] = None

  # Launch options
  headless: bool = False
  executable_path: Optional[str] = None
  extra_args: tuple[str, ...] = ()
  no_sandbox: bool = False
  proxy: Optional[str] = None
  stealth: bool = True

  # Behavior
  timeout: float = 30.0
  viewport_width: int = 1280
  viewport_height: int = 720
  locale: str = "en-US"
  timezone: Optional[str] = None
  user_agent: Optional[str] = None

  # Page state ring buffer sizes
  max_console_messages: int = 500
  max_page_errors: int = 200
  max_network_requests: int = 500

  # Downloads
  downloads_dir: Optional[str] = None

  # Advanced
  cdp_port: int = 9222
  slow_mo: float = 0.0

  # Deprecated compat (emit DeprecationWarning in __post_init__)
  host: Optional[str] = None
  port: Optional[int] = None

  # Deprecated SeleniumBase compat
  sandbox: Optional[bool] = None
  browser_executable_path: Optional[str] = None
  browser_args: Optional[tuple[str, ...]] = None

  def __post_init__(self) -> None:
    # host/port -> cdp_url migration
    if self.host and self.port:
      warnings.warn(
        "host/port are deprecated. Use cdp_url='ws://host:port' instead.",
        DeprecationWarning,
        stacklevel=3,
      )
      if not self.cdp_url:
        object.__setattr__(self, "cdp_url", f"ws://{self.host}:{self.port}")

    # sandbox -> no_sandbox migration
    if self.sandbox is not None:
      warnings.warn(
        "sandbox is deprecated. Use no_sandbox=True instead of sandbox=False.",
        DeprecationWarning,
        stacklevel=3,
      )
      if not self.no_sandbox and self.sandbox is False:
        object.__setattr__(self, "no_sandbox", True)

    # browser_executable_path -> executable_path migration
    if self.browser_executable_path is not None:
      warnings.warn(
        "browser_executable_path is deprecated. Use executable_path instead.",
        DeprecationWarning,
        stacklevel=3,
      )
      if not self.executable_path:
        object.__setattr__(self, "executable_path", self.browser_executable_path)

    # browser_args -> extra_args migration
    if self.browser_args is not None:
      warnings.warn(
        "browser_args is deprecated. Use extra_args instead.",
        DeprecationWarning,
        stacklevel=3,
      )
      if not self.extra_args:
        object.__setattr__(self, "extra_args", self.browser_args)
