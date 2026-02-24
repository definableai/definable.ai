"""SSRF navigation protection for browser operations.

Validates URLs before and after navigation to prevent Server-Side Request
Forgery attacks. Ported from openclaw navigation-guard.ts.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlparse


class NavigationBlockedError(Exception):
  """Raised when a navigation URL is blocked by SSRF policy."""


ALLOWED_PROTOCOLS = {"http", "https"}
SAFE_NON_NETWORK_URLS = {"about:blank"}


def _is_private_ip(addr: str) -> bool:
  """Check if an IP address is in a private/reserved range."""
  try:
    ip = ipaddress.ip_address(addr)
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
  except ValueError:
    return False


async def _resolve_hostname(hostname: str) -> list[str]:
  """Resolve hostname to IP addresses using asyncio executor."""
  loop = asyncio.get_running_loop()
  try:
    results = await loop.run_in_executor(
      None,
      lambda: socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM),
    )
    return [str(r[4][0]) for r in results]
  except socket.gaierror:
    return []


async def assert_navigation_allowed(url: str) -> None:
  """Validate a URL before navigation. Blocks private IPs and non-http protocols.

  Raises:
      NavigationBlockedError: If the URL is blocked.
  """
  raw = (url or "").strip()
  if not raw:
    raise NavigationBlockedError("URL is required")

  # Allow safe non-network URLs
  if raw in SAFE_NON_NETWORK_URLS:
    return

  try:
    parsed = urlparse(raw)
  except Exception:
    raise NavigationBlockedError(f"Invalid URL: {raw}")

  if not parsed.scheme:
    raise NavigationBlockedError(f"Invalid URL (no scheme): {raw}")

  if parsed.scheme not in ALLOWED_PROTOCOLS:
    raise NavigationBlockedError(f'Navigation blocked: unsupported protocol "{parsed.scheme}://"')

  hostname = parsed.hostname
  if not hostname:
    raise NavigationBlockedError(f"Invalid URL (no hostname): {raw}")

  # Resolve and check for private IPs
  addresses = await _resolve_hostname(hostname)
  if not addresses:
    raise NavigationBlockedError(f"Cannot resolve hostname: {hostname}")

  for addr in addresses:
    if _is_private_ip(addr):
      raise NavigationBlockedError(f"Navigation blocked: {hostname} resolves to private IP {addr}")


async def assert_navigation_result_allowed(url: str) -> None:
  """Best-effort post-navigation guard for final page URLs.

  Only validates network URLs (http/https) to avoid false positives
  on browser-internal error pages (e.g. chrome-error://).
  """
  raw = (url or "").strip()
  if not raw:
    return

  try:
    parsed = urlparse(raw)
  except Exception:
    return

  # Only validate network URLs
  if parsed.scheme in ALLOWED_PROTOCOLS:
    await assert_navigation_allowed(raw)
  # Skip chrome-error://, devtools://, etc.
