"""Model resilience — key rotation, failover chains, resilient wrappers."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.model.resilience.failover import FailoverChain, FailoverEntry
  from definable.model.resilience.key_pool import KeyHealth, KeyPool, SelectionStrategy
  from definable.model.resilience.resilient import ResilientModel

__all__ = [
  "KeyPool",
  "KeyHealth",
  "SelectionStrategy",
  "FailoverChain",
  "FailoverEntry",
  "ResilientModel",
]


def __getattr__(name: str):
  if name in ("KeyPool", "KeyHealth", "SelectionStrategy"):
    from definable.model.resilience import key_pool as _kp

    return getattr(_kp, name)
  if name in ("FailoverChain", "FailoverEntry"):
    from definable.model.resilience import failover as _fo

    return getattr(_fo, name)
  if name == "ResilientModel":
    from definable.model.resilience.resilient import ResilientModel

    return ResilientModel
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
