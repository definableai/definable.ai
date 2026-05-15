"""USD cost calculation from a flat usage dict.

Thin adapter on top of :class:`definable.model.pricing.PricingRegistry`
(which loads ``model_pricing.json``). Observability stores token counts
in a plain ``dict[str, int]`` on each :class:`ModelResponded` event;
this module turns that dict into a number for the UI.

Unknown ``(provider, model)`` pairs return ``0.0`` and emit a one-time
warning so unrecognized models still render cleanly in the dashboard.
"""

from __future__ import annotations

import logging

from definable.model.metrics import Metrics
from definable.model.pricing import PricingRegistry

log = logging.getLogger(__name__)

_WARNED: set[tuple[str, str]] = set()


def cost(provider: str | None, model: str | None, usage: dict[str, int] | None) -> float:
  """Compute USD cost for one model response.

  Returns ``0.0`` when any of provider/model/usage is missing or when the
  registry has no rate for the pair. Designed to be safe to call from the
  hot path: zero allocations on the unknown-model path beyond the
  warning-once bookkeeping.
  """
  if not provider or not model or not usage:
    return 0.0
  rates = PricingRegistry().get_pricing(provider, model)
  if rates is None:
    key = (provider, model)
    if key not in _WARNED:
      _WARNED.add(key)
      log.warning("No pricing for %s/%s — cost will render as $0.00", provider, model)
    return 0.0
  metrics = Metrics(
    input_tokens=int(usage.get("input_tokens", 0)),
    output_tokens=int(usage.get("output_tokens", 0)),
    total_tokens=int(usage.get("total_tokens", 0)),
    cache_read_tokens=int(usage.get("cache_read_tokens", 0)),
    cache_write_tokens=int(usage.get("cache_write_tokens", 0)),
    reasoning_tokens=int(usage.get("reasoning_tokens", 0)),
  )
  return rates.calculate_cost(metrics)
