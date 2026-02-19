"""API key authentication provider."""

from typing import Any, Mapping, Optional, Set, Union

from definable.agent.auth.base import AuthContext


def _get_header(headers: Any, name: str) -> str:
  """Case-insensitive header lookup.

  Starlette Headers are natively case-insensitive, but plain dicts
  (e.g. from AuthRequest) are not. This helper normalises lookup so
  both work identically.
  """
  # Try exact match first (fast path for Starlette Headers / matching case)
  value = headers.get(name, "")
  if value:
    return value
  # Fall back to case-insensitive scan for plain dicts
  name_lower = name.lower()
  for key, val in headers.items() if isinstance(headers, Mapping) else []:
    if key.lower() == name_lower:
      return val
  return ""


class APIKeyAuth:
  """Validates requests against a set of allowed API keys.

  Checks the specified header for a key, stripping optional ``"Bearer "``
  prefix.

  Args:
    keys: A single API key string or a set of allowed keys.
    header: HTTP header name to read the key from (default ``"X-API-Key"``).

  Example::

    agent.auth = APIKeyAuth(keys={"sk-abc123", "sk-def456"})
    agent.serve(enable_server=True)
  """

  def __init__(
    self,
    keys: Union[str, Set[str]],
    *,
    header: str = "X-API-Key",
  ) -> None:
    self.keys: Set[str] = {keys} if isinstance(keys, str) else set(keys)
    self.header = header

  def authenticate(self, request: Any) -> Optional[AuthContext]:
    """Check the request header for a valid API key.

    Args:
      request: An AuthRequest or HTTP request object with a ``headers`` attribute.

    Returns:
      AuthContext with a hashed user_id on success, None on failure.
    """
    headers = getattr(request, "headers", {})
    value = _get_header(headers, self.header)
    if not value:
      # Fall back to Authorization header
      value = _get_header(headers, "authorization")

    # Strip "Bearer " prefix
    if value.lower().startswith("bearer "):
      value = value[7:]

    value = value.strip()
    if not value or value not in self.keys:
      return None

    # Use a truncated hash as user_id (don't expose full key)
    import hashlib

    key_hash = hashlib.sha256(value.encode()).hexdigest()[:12]
    return AuthContext(user_id=f"apikey_{key_hash}")
