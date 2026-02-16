"""Trigger system for webhooks, cron jobs, and programmatic events."""

from definable.triggers.base import BaseTrigger, TriggerEvent, TriggerResult
from definable.triggers.event import EventTrigger
from definable.triggers.executor import TriggerExecutor
from definable.triggers.webhook import Webhook

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.triggers.cron import Cron


# Lazy import for Cron (requires croniter)
def __getattr__(name: str):
  if name == "Cron":
    from definable.triggers.cron import Cron

    return Cron
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
  "BaseTrigger",
  "TriggerEvent",
  "TriggerResult",
  "Webhook",
  "Cron",
  "EventTrigger",
  "TriggerExecutor",
]
