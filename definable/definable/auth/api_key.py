"""API key authentication provider."""

from typing import Any, Optional, Set, Union

from definable.auth.base import AuthContext


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
      request: HTTP request object with a ``headers`` attribute.

    Returns:
      AuthContext with a hashed user_id on success, None on failure.
    """
    value = getattr(request, "headers", {}).get(self.header, "")
    if not value:
      # Fall back to Authorization header
      value = getattr(request, "headers", {}).get("authorization", "")

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
