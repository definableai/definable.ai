"""Auth base types — AuthProvider protocol and AuthContext."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@dataclass
class AuthContext:
  """Authenticated request context.

  Attributes:
    user_id: Canonical user identifier.
    metadata: Arbitrary key-value metadata from the auth provider.
  """

  user_id: str
  metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class AuthProvider(Protocol):
  """Protocol for pluggable authentication providers.

  Implementations must define ``authenticate(request)`` which returns
  an :class:`AuthContext` on success or ``None`` on failure.

  The method may be sync or async.
  """

  def authenticate(self, request: Any) -> Optional[AuthContext]:
    """Authenticate an incoming HTTP request.

    Args:
      request: The framework request object (e.g. FastAPI Request).

    Returns:
      AuthContext on success, None on failure.
    """
    ...
