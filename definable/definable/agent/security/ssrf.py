"""SSRF (Server-Side Request Forgery) protection.

Validates outbound URLs to ensure they don't resolve to private/internal
IP addresses. Used by tools that make HTTP requests with untrusted URLs.

Usage::

    from definable.agent.security.ssrf import SSRFGuard, is_private_ip

    guard = SSRFGuard()
    response = await guard.get("https://example.com/api")  # OK
    response = await guard.get("http://169.254.169.254/metadata")  # SSRFBlockedError
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass, field
from typing import Optional, Set
from urllib.parse import urlparse


# ------------------------------------------------------------------
# Private / reserved IP ranges
# ------------------------------------------------------------------

PRIVATE_RANGES: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
  # RFC 1918 private
  ipaddress.IPv4Network("10.0.0.0/8"),
  ipaddress.IPv4Network("172.16.0.0/12"),
  ipaddress.IPv4Network("192.168.0.0/16"),
  # Loopback
  ipaddress.IPv4Network("127.0.0.0/8"),
  ipaddress.IPv6Network("::1/128"),
  # Link-local
  ipaddress.IPv4Network("169.254.0.0/16"),
  ipaddress.IPv6Network("fe80::/10"),
  # Cloud metadata endpoints
  ipaddress.IPv4Network("169.254.169.254/32"),
  # IPv6 unique local
  ipaddress.IPv6Network("fc00::/7"),
]


class SSRFBlockedError(Exception):
  """Raised when a URL resolves to a private/reserved IP address."""

  def __init__(self, url: str, resolved_ip: str, reason: str = "") -> None:
    self.url = url
    self.resolved_ip = resolved_ip
    self.reason = reason or f"URL '{url}' resolves to private IP {resolved_ip}"
    super().__init__(self.reason)


def is_private_ip(ip_str: str) -> bool:
  """Check if an IP address falls within a private/reserved range.

  Args:
    ip_str: IP address string (IPv4 or IPv6).

  Returns:
    True if the IP is private/reserved, False otherwise.
  """
  try:
    addr = ipaddress.ip_address(ip_str)
  except ValueError:
    return False  # Invalid IP — not private
  return any(addr in network for network in PRIVATE_RANGES)


def resolve_and_check(
  url: str,
  *,
  allowed_private: Optional[Set[str]] = None,
) -> str:
  """DNS-resolve a URL's hostname and verify it's not a private IP.

  Args:
    url: The URL to check.
    allowed_private: Set of hostnames that are allowed to resolve to
      private IPs (e.g. ``{"localhost"}`` for known-safe local services).

  Returns:
    The original URL if safe.

  Raises:
    SSRFBlockedError: If the resolved IP is private/reserved.
    ValueError: If the URL cannot be parsed.
  """
  parsed = urlparse(url)
  hostname = parsed.hostname

  if not hostname:
    raise ValueError(f"Cannot extract hostname from URL: {url}")

  # Allow known-safe private hosts
  if allowed_private and hostname in allowed_private:
    return url

  # Resolve DNS
  try:
    results = socket.getaddrinfo(hostname, parsed.port or 80, proto=socket.IPPROTO_TCP)
  except socket.gaierror as exc:
    raise ValueError(f"Cannot resolve hostname '{hostname}': {exc}") from exc

  for family, _type, _proto, _canonname, sockaddr in results:
    ip_str = str(sockaddr[0])
    if is_private_ip(ip_str):
      raise SSRFBlockedError(
        url=url,
        resolved_ip=ip_str,
        reason=f"URL '{url}' resolves to private IP {ip_str} (hostname: {hostname})",
      )

  return url


# ------------------------------------------------------------------
# SSRFGuard — safe HTTP client wrapper
# ------------------------------------------------------------------


@dataclass
class SSRFGuardConfig:
  """Configuration for SSRF protection.

  Attributes:
    enabled: Whether SSRF checks are active.
    allowed_private_hosts: Hostnames that may resolve to private IPs.
  """

  enabled: bool = True
  allowed_private_hosts: Set[str] = field(default_factory=set)


class SSRFGuard:
  """HTTP client wrapper that enforces SSRF protection.

  Validates URLs before making requests, blocking those that resolve
  to private/reserved IP ranges.

  Args:
    config: SSRF guard configuration.
  """

  def __init__(self, config: Optional[SSRFGuardConfig] = None) -> None:
    self._config = config or SSRFGuardConfig()

  def check_url(self, url: str) -> str:
    """Validate a URL against SSRF rules.

    Returns the URL if safe, raises SSRFBlockedError if not.
    """
    if not self._config.enabled:
      return url
    return resolve_and_check(url, allowed_private=self._config.allowed_private_hosts)

  async def get(self, url: str, **kwargs: object) -> object:
    """SSRF-safe HTTP GET request.

    Validates the URL before delegating to httpx.
    """
    import httpx

    self.check_url(url)
    async with httpx.AsyncClient() as client:
      return await client.get(url, **kwargs)  # type: ignore[arg-type]

  async def post(self, url: str, **kwargs: object) -> object:
    """SSRF-safe HTTP POST request."""
    import httpx

    self.check_url(url)
    async with httpx.AsyncClient() as client:
      return await client.post(url, **kwargs)  # type: ignore[arg-type]
