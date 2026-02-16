"""User-ID-based allowlist authentication provider for messaging interfaces."""

from typing import Any, Optional, Set

from definable.auth.base import AuthContext, AuthRequest


class AllowlistAuth:
  """Validates requests against a set of allowed user IDs.

  Designed for messaging interfaces (Telegram, Discord, Signal) where
  user identity is provided by the platform rather than by HTTP headers.

  Args:
    user_ids: Set of allowed platform user IDs.
    chat_ids: Optional set of allowed chat/conversation IDs.
    platforms: Optional set of platforms this provider applies to.
      When set, requests from other platforms return None (not applicable).

  Example::

    from definable.auth import AllowlistAuth

    telegram = TelegramInterface(
      config=TelegramConfig(bot_token=token),
      auth=AllowlistAuth(user_ids={"12345", "67890"}),
    )
  """

  def __init__(
    self,
    user_ids: Set[str],
    *,
    chat_ids: Optional[Set[str]] = None,
    platforms: Optional[Set[str]] = None,
  ) -> None:
    self.user_ids = set[str](user_ids)
    self.chat_ids = set[str](chat_ids) if chat_ids is not None else None
    self.platforms = set[str](platforms) if platforms is not None else None

  def authenticate(self, request: Any) -> Optional[AuthContext]:
    """Check if the request's user ID is in the allowlist.

    Only applies to :class:`AuthRequest` instances. Returns None for
    non-AuthRequest objects (not applicable — lets other providers handle it).

    Args:
      request: An AuthRequest from a messaging interface.

    Returns:
      AuthContext on success, None on failure or non-applicable request.
    """
    if not isinstance(request, AuthRequest):
      return None

    # Platform scoping: skip if this provider doesn't cover the platform
    if self.platforms is not None and request.platform not in self.platforms:
      return None

    # Chat ID filter
    if self.chat_ids is not None and request.chat_id not in self.chat_ids:
      return None

    # User ID check
    if request.user_id is None or request.user_id not in self.user_ids:
      return None

    return AuthContext(
      user_id=request.user_id,
      metadata={
        "platform": request.platform,
        "auth_method": "allowlist",
      },
    )
