"""Events emitted by the resilience layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KeyRotatedEvent:
  """Emitted when a key is rotated after a rate limit."""

  old_key_prefix: str  # first 8 chars
  new_key_prefix: str
  reason: str = "rate_limited"


@dataclass
class ProviderFailoverEvent:
  """Emitted when failover to a different provider occurs."""

  from_model_id: str
  to_model_id: str
  reason: str = ""
