"""Ollama metrics — null-safe tokens + timing breakdown in provider_metrics."""

import pytest

pytest.importorskip("ollama")

from definable.model.ollama.chat import Ollama  # noqa: E402


def test_get_metrics_null_safe_tokens():
  model = Ollama(id="qwen3:0.6b")
  metrics = model._get_metrics({"done": True})
  assert metrics.input_tokens == 0
  assert metrics.output_tokens == 0
  assert metrics.total_tokens == 0


def test_get_metrics_with_tokens_and_timings():
  model = Ollama(id="qwen3:0.6b")
  resp = {
    "prompt_eval_count": 10,
    "eval_count": 25,
    "total_duration": 5_000_000_000,
    "load_duration": 1_000_000_000,
    "prompt_eval_duration": 500_000_000,
    "eval_duration": 3_500_000_000,
  }
  metrics = model._get_metrics(resp)
  assert metrics.input_tokens == 10
  assert metrics.output_tokens == 25
  assert metrics.total_tokens == 35
  assert metrics.provider_metrics is not None
  assert metrics.provider_metrics["total_duration"] == 5_000_000_000
  assert metrics.provider_metrics["load_duration"] == 1_000_000_000


def test_get_metrics_omits_missing_timings():
  model = Ollama(id="qwen3:0.6b")
  resp = {"prompt_eval_count": 5, "eval_count": 10, "total_duration": 1_000_000}
  metrics = model._get_metrics(resp)
  assert metrics.provider_metrics is not None
  assert "total_duration" in metrics.provider_metrics
  assert "load_duration" not in metrics.provider_metrics


def test_get_metrics_no_timings_no_provider_metrics():
  model = Ollama(id="qwen3:0.6b")
  resp = {"prompt_eval_count": 5, "eval_count": 10}
  metrics = model._get_metrics(resp)
  assert metrics.provider_metrics is None
