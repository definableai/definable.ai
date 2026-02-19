"""Auth base types — AuthProvider protocol, AuthContext, and AuthRequest."""

import inspect
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


@dataclass
class AuthRequest:
  """Transport-agnostic authentication request.

  Normalizes auth input from any transport (HTTP, Telegram, Discord, etc.)
  so auth providers receive a consistent interface.

  Attributes:
    platform: Transport origin — "http", "telegram", "discord", "signal", etc.
    user_id: Pre-known user identity (messaging interfaces provide this).
    username: Display name of the sender.
    chat_id: Chat/conversation identifier.
    headers: HTTP headers (empty dict for messaging transports).
    metadata: Arbitrary transport-specific data.
  """

  platform: str
  user_id: Optional[str] = None
  username: Optional[str] = None
  chat_id: Optional[str] = None
  headers: Dict[str, str] = field(default_factory=dict)
  metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class AuthProvider(Protocol):
  """Protocol for pluggable authentication providers.

  Implementations must define ``authenticate(request)`` which returns
  an :class:`AuthContext` on success or ``None`` on failure.

  The *request* may be an :class:`AuthRequest` (transport-agnostic) or
  a framework-specific object (e.g. FastAPI ``Request``).  The method
  may be sync or async.
  """

  def authenticate(self, request: Any) -> Optional[AuthContext]:
    """Authenticate an incoming request.

    Args:
      request: An AuthRequest or framework request object.

    Returns:
      AuthContext on success, None on failure.
    """
    ...


async def resolve_auth(auth_provider: Any, request: Any) -> Optional[AuthContext]:
  """Call an auth provider's ``authenticate`` method.

  Handles both sync and async providers transparently.

  Args:
    auth_provider: Object with an ``authenticate(request)`` method.
    request: The request to authenticate (AuthRequest or framework request).

  Returns:
    AuthContext on success, None on failure.
  """
  result = auth_provider.authenticate(request)
  if inspect.isawaitable(result):
    return await result
  return result
