"""Composite authentication provider that chains multiple providers."""

from typing import Any, Optional

from definable.auth.base import AuthContext, resolve_auth


class CompositeAuth:
  """Chains multiple auth providers, returning the first successful result.

  Tries each provider in order and returns the first non-None
  :class:`AuthContext`. Supports mixed sync and async providers.

  Args:
    *providers: Auth provider instances to chain.

  Raises:
    ValueError: If no providers are given.

  Example::

    from definable.auth import APIKeyAuth, AllowlistAuth, CompositeAuth

    agent.auth = CompositeAuth(
      APIKeyAuth(keys={"sk-123"}),
      AllowlistAuth(user_ids={"12345"}, platforms={"telegram"}),
    )
  """

  def __init__(self, *providers: Any) -> None:
    if not providers:
      raise ValueError("CompositeAuth requires at least one provider")
    self._providers = list(providers)

  async def authenticate(self, request: Any) -> Optional[AuthContext]:
    """Try each provider in order, return first success.

    Args:
      request: The request to authenticate.

    Returns:
      AuthContext from the first provider that succeeds, or None.
    """
    for provider in self._providers:
      result = await resolve_auth(provider, request)
      if result is not None:
        return result
    return None
