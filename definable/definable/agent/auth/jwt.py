"""JWT authentication provider."""

from typing import Any, Optional

from definable.agent.auth.api_key import _get_header
from definable.agent.auth.base import AuthContext


class JWTAuth:
  """Validates requests using JWT Bearer tokens.

  Requires ``pyjwt`` (lazy-imported).

  Args:
    secret: Secret key or public key for token validation.
    algorithm: JWT algorithm (default ``"HS256"``).
    audience: Optional expected audience claim.
    issuer: Optional expected issuer claim.

  Example::

    agent.auth = JWTAuth(secret="my-secret-key")
    agent.serve(enable_server=True)
  """

  def __init__(
    self,
    secret: str,
    *,
    algorithm: str = "HS256",
    audience: Optional[str] = None,
    issuer: Optional[str] = None,
  ) -> None:
    self.secret = secret
    self.algorithm = algorithm
    self.audience = audience
    self.issuer = issuer

  def authenticate(self, request: Any) -> Optional[AuthContext]:
    """Validate a JWT Bearer token from the Authorization header.

    Extracts ``user_id`` from the ``sub``, ``user_id``, or ``id``
    claim (checked in that order).

    Args:
      request: HTTP request object with a ``headers`` attribute.

    Returns:
      AuthContext on success, None on failure.
    """
    try:
      import jwt
    except ImportError as e:
      raise ImportError("pyjwt is required for JWTAuth. Install it with: pip install 'definable[jwt]'") from e

    auth_header = _get_header(getattr(request, "headers", {}), "authorization")
    if not auth_header.lower().startswith("bearer "):
      return None

    token = auth_header[7:].strip()
    if not token:
      return None

    try:
      payload = jwt.decode(
        token,
        self.secret,
        algorithms=[self.algorithm],
        audience=self.audience,
        issuer=self.issuer,
      )
    except jwt.InvalidTokenError:
      return None

    # Extract user_id from standard claims
    user_id = payload.get("sub") or payload.get("user_id") or payload.get("id")
    if not user_id:
      return None

    return AuthContext(
      user_id=str(user_id),
      metadata={k: v for k, v in payload.items() if k not in ("sub", "user_id", "id")},
    )
