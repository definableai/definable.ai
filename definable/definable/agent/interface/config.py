"""Base configuration for interfaces."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class InterfaceConfig:
  """Base configuration for all interface implementations.

  Uses frozen dataclass for immutability, following the AgentConfig pattern.

  Attributes:
    platform: Platform identifier (e.g. "telegram").
    max_session_history: Maximum messages to keep in session history.
    session_ttl_seconds: Session time-to-live in seconds.
    max_concurrent_requests: Maximum concurrent agent requests.
    error_message: Default error message sent to users on failure.
    typing_indicator: Whether to show typing indicator while processing.
    max_message_length: Maximum message length for the platform.
    rate_limit_messages_per_minute: Rate limit per user per minute.
  """

  platform: str = ""
  max_session_history: int = 50
  session_ttl_seconds: int = 3600
  max_concurrent_requests: int = 10
  error_message: str = "Sorry, something went wrong. Please try again."
  typing_indicator: bool = True
  max_message_length: int = 4096
  rate_limit_messages_per_minute: int = 30

  def with_updates(self, **kwargs: object) -> "InterfaceConfig":
    """Create a new config with updated values (immutable pattern).

    Args:
      **kwargs: Fields to update in the new config.

    Returns:
      New InterfaceConfig instance with updated values.
    """
    current = asdict(self)
    current.update(kwargs)
    return self.__class__(**current)
