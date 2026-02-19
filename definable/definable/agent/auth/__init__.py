"""Pluggable authentication for the agent HTTP server and messaging interfaces."""

from definable.agent.auth.api_key import APIKeyAuth
from definable.agent.auth.base import AuthContext, AuthProvider, AuthRequest, resolve_auth

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.agent.auth.allowlist import AllowlistAuth
  from definable.agent.auth.composite import CompositeAuth
  from definable.agent.auth.jwt import JWTAuth


# Lazy imports for optional providers
def __getattr__(name: str):
  if name == "JWTAuth":
    from definable.agent.auth.jwt import JWTAuth

    return JWTAuth
  if name == "AllowlistAuth":
    from definable.agent.auth.allowlist import AllowlistAuth

    return AllowlistAuth
  if name == "CompositeAuth":
    from definable.agent.auth.composite import CompositeAuth

    return CompositeAuth
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
  "AuthProvider",
  "AuthContext",
  "AuthRequest",
  "resolve_auth",
  "APIKeyAuth",
  "JWTAuth",
  "AllowlistAuth",
  "CompositeAuth",
]
