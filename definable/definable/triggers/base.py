"""Base trigger types for the agent runtime."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Optional, Union

if TYPE_CHECKING:
  from definable.agents.agent import Agent

TriggerResult = Union[None, str, Dict[str, Any], Awaitable]


@dataclass
class TriggerEvent:
  """Event passed to trigger handlers.

  Attributes:
    body: Parsed request body (dict for webhooks, None for cron).
    headers: HTTP headers (webhooks only).
    source: Human-readable trigger source, e.g. ``"POST /webhook"``.
    timestamp: Unix timestamp when the event was created.
    raw: Raw request object (framework-specific).
  """

  body: Optional[Dict[str, Any]] = None
  headers: Optional[Dict[str, str]] = None
  source: str = ""
  timestamp: float = field(default_factory=time)
  raw: Any = None


class BaseTrigger(ABC):
  """Abstract base for all trigger types.

  Subclasses must implement :attr:`name` to provide a human-readable
  identifier.
  """

  handler: Optional[Callable] = None
  agent: Optional["Agent"] = None
  auth: Optional[Any] = None  # Per-trigger auth override (None = use agent default)

  @property
  @abstractmethod
  def name(self) -> str:
    """Human-readable trigger identifier, e.g. ``'POST /webhook'``."""
    ...
