from __future__ import annotations

from definable.observability import pricing as pricing_mod
from definable.observability.pricing import cost


def setup_function() -> None:
  pricing_mod._WARNED.clear()


def test_known_model_returns_positive_cost() -> None:
  usd = cost("openai", "gpt-4o", {"input_tokens": 1_000_000, "output_tokens": 1_000_000})
  # Per pricing JSON: input $2.50/M, output $10.00/M → $12.50 total.
  assert usd == 12.50


def test_cached_input_path() -> None:
  usd_full = cost("openai", "gpt-4o", {"input_tokens": 1_000_000, "output_tokens": 0})
  usd_cached = cost(
    "openai",
    "gpt-4o",
    {"input_tokens": 1_000_000, "cache_read_tokens": 500_000, "output_tokens": 0},
  )
  assert usd_cached < usd_full


def test_unknown_model_returns_zero_and_warns_once() -> None:
  assert cost("nope_prov", "nope_mod", {"input_tokens": 100}) == 0.0
  assert cost("nope_prov", "nope_mod", {"input_tokens": 100}) == 0.0
  # Warn-once cache holds exactly one entry for the same pair regardless of repeat calls.
  assert ("nope_prov", "nope_mod") in pricing_mod._WARNED
  assert len(pricing_mod._WARNED) == 1


def test_missing_inputs_return_zero() -> None:
  assert cost(None, "gpt-4o", {"input_tokens": 1}) == 0.0
  assert cost("openai", None, {"input_tokens": 1}) == 0.0
  assert cost("openai", "gpt-4o", None) == 0.0
  assert cost("openai", "gpt-4o", {}) == 0.0
