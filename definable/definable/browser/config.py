"""BrowserConfig — frozen configuration dataclass for BrowserToolkit (Playwright)."""

from __future__ import annotations

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
