"""Exception hierarchy for the interfaces module."""

from typing import Optional

from definable.exceptions import DefinableError


class InterfaceError(DefinableError):
  """Base exception for all interface errors."""

  def __init__(self, message: str, status_code: int = 500, platform: str = ""):
    super().__init__(message, status_code)
    self.platform = platform
    self.type = "interface_error"
    self.error_id = "interface_error"


class InterfaceConnectionError(InterfaceError):
  """Raised when connection to the platform fails."""

  def __init__(self, message: str, platform: str = ""):
    super().__init__(message, status_code=503, platform=platform)
    self.type = "interface_connection_error"
    self.error_id = "interface_connection_error"


class InterfaceAuthenticationError(InterfaceError):
  """Raised when bot token or credentials are invalid."""

  def __init__(self, message: str, platform: str = ""):
    super().__init__(message, status_code=401, platform=platform)
    self.type = "interface_authentication_error"
    self.error_id = "interface_authentication_error"


class InterfaceRateLimitError(InterfaceError):
  """Raised when the platform rate limit is exceeded."""

  def __init__(self, message: str, platform: str = "", retry_after: Optional[float] = None):
    super().__init__(message, status_code=429, platform=platform)
    self.retry_after = retry_after
    self.type = "interface_rate_limit_error"
    self.error_id = "interface_rate_limit_error"


class InterfaceMessageError(InterfaceError):
  """Raised when message send/receive fails."""

  def __init__(self, message: str, platform: str = ""):
    super().__init__(message, status_code=400, platform=platform)
    self.type = "interface_message_error"
    self.error_id = "interface_message_error"
