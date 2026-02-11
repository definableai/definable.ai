"""Pluggable authentication for the agent HTTP server."""

from definable.auth.api_key import APIKeyAuth
from definable.auth.base import AuthContext, AuthProvider


# Lazy import for JWTAuth (requires pyjwt)
def __getattr__(name: str):
  if name == "JWTAuth":
    from definable.auth.jwt import JWTAuth

    return JWTAuth
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


JWTAuth: type

__all__ = [
  "AuthProvider",
  "AuthContext",
  "APIKeyAuth",
  "JWTAuth",
]
