"""Trigger system for webhooks, cron jobs, and programmatic events."""

from definable.agent.trigger.base import BaseTrigger, TriggerEvent, TriggerResult
from definable.agent.trigger.event import EventTrigger
from definable.agent.trigger.executor import TriggerExecutor
from definable.agent.trigger.webhook import Webhook

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.agent.trigger.cron import Cron
  from definable.agent.trigger.interval import Interval
  from definable.agent.trigger.oneshot import OneShot


# Lazy imports for optional-dependency triggers
def __getattr__(name: str):
  if name == "Cron":
    from definable.agent.trigger.cron import Cron

    return Cron
  if name == "Interval":
    from definable.agent.trigger.interval import Interval

    return Interval
  if name == "OneShot":
    from definable.agent.trigger.oneshot import OneShot

    return OneShot
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
  "BaseTrigger",
  "TriggerEvent",
  "TriggerResult",
  "Webhook",
  "Cron",
  "EventTrigger",
  "TriggerExecutor",
  "Interval",  # noqa: F822
  "OneShot",  # noqa: F822
]
