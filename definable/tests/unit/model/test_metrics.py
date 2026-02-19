"""
Unit tests for the Metrics dataclass.

Tests construction with defaults, field storage, accumulation via __add__
and __radd__, to_dict() serialization, and timer integration.
No API calls. No external dependencies.

Covers:
  - Metrics creation with all defaults
  - Token counting fields (input, output, total, audio, cache, reasoning)
  - Time tracking fields (duration, time_to_first_token)
  - Cost field
  - Provider and additional metrics dicts
  - Metrics.__add__() accumulates token counts
  - Metrics.__add__() merges optional fields (cost, duration, provider_metrics)
  - Metrics.__radd__() supports sum() with start=0
  - Metrics.to_dict() filters out zero/None/empty values
  - Metrics.to_dict() excludes the timer field
  - Timer integration (start_timer, stop_timer, set_time_to_first_token)
"""

import pytest

from definable.model.metrics import Metrics


# ---------------------------------------------------------------------------
# Metrics: creation with defaults
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMetricsDefaults:
  """Metrics created with no arguments has sensible zero/None defaults."""

  def test_input_tokens_default_zero(self):
    m = Metrics()
    assert m.input_tokens == 0

  def test_output_tokens_default_zero(self):
    m = Metrics()
    assert m.output_tokens == 0

  def test_total_tokens_default_zero(self):
    m = Metrics()
    assert m.total_tokens == 0

  def test_audio_input_tokens_default_zero(self):
    m = Metrics()
    assert m.audio_input_tokens == 0

  def test_audio_output_tokens_default_zero(self):
    m = Metrics()
    assert m.audio_output_tokens == 0

  def test_audio_total_tokens_default_zero(self):
    m = Metrics()
    assert m.audio_total_tokens == 0

  def test_cache_read_tokens_default_zero(self):
    m = Metrics()
    assert m.cache_read_tokens == 0

  def test_cache_write_tokens_default_zero(self):
    m = Metrics()
    assert m.cache_write_tokens == 0

  def test_reasoning_tokens_default_zero(self):
    m = Metrics()
    assert m.reasoning_tokens == 0

  def test_cost_default_none(self):
    m = Metrics()
    assert m.cost is None

  def test_duration_default_none(self):
    m = Metrics()
    assert m.duration is None

  def test_time_to_first_token_default_none(self):
    m = Metrics()
    assert m.time_to_first_token is None

  def test_timer_default_none(self):
    m = Metrics()
    assert m.timer is None

  def test_provider_metrics_default_none(self):
    m = Metrics()
    assert m.provider_metrics is None

  def test_additional_metrics_default_none(self):
    m = Metrics()
    assert m.additional_metrics is None


# ---------------------------------------------------------------------------
# Metrics: field storage
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMetricsFieldStorage:
  """Metrics stores all fields accurately when provided."""

  def test_input_tokens_stored(self):
    m = Metrics(input_tokens=100)
    assert m.input_tokens == 100

  def test_output_tokens_stored(self):
    m = Metrics(output_tokens=200)
    assert m.output_tokens == 200

  def test_total_tokens_stored(self):
    m = Metrics(total_tokens=300)
    assert m.total_tokens == 300

  def test_cost_stored(self):
    m = Metrics(cost=0.005)
    assert m.cost == 0.005

  def test_audio_tokens_stored(self):
    m = Metrics(audio_input_tokens=10, audio_output_tokens=20, audio_total_tokens=30)
    assert m.audio_input_tokens == 10
    assert m.audio_output_tokens == 20
    assert m.audio_total_tokens == 30

  def test_cache_tokens_stored(self):
    m = Metrics(cache_read_tokens=50, cache_write_tokens=25)
    assert m.cache_read_tokens == 50
    assert m.cache_write_tokens == 25

  def test_reasoning_tokens_stored(self):
    m = Metrics(reasoning_tokens=500)
    assert m.reasoning_tokens == 500

  def test_duration_stored(self):
    m = Metrics(duration=1.5)
    assert m.duration == 1.5

  def test_time_to_first_token_stored(self):
    m = Metrics(time_to_first_token=0.25)
    assert m.time_to_first_token == 0.25

  def test_provider_metrics_stored(self):
    m = Metrics(provider_metrics={"model_id": "gpt-4o"})
    assert m.provider_metrics == {"model_id": "gpt-4o"}

  def test_additional_metrics_stored(self):
    m = Metrics(additional_metrics={"custom": 42})
    assert m.additional_metrics == {"custom": 42}


# ---------------------------------------------------------------------------
# Metrics: __add__() accumulation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMetricsAddition:
  """Metrics.__add__() sums token counts and merges optional fields."""

  def test_adds_input_tokens(self):
    a = Metrics(input_tokens=10)
    b = Metrics(input_tokens=20)
    result = a + b
    assert result.input_tokens == 30

  def test_adds_output_tokens(self):
    a = Metrics(output_tokens=100)
    b = Metrics(output_tokens=200)
    result = a + b
    assert result.output_tokens == 300

  def test_adds_total_tokens(self):
    a = Metrics(total_tokens=50)
    b = Metrics(total_tokens=60)
    result = a + b
    assert result.total_tokens == 110

  def test_adds_audio_tokens(self):
    a = Metrics(audio_total_tokens=5, audio_input_tokens=2, audio_output_tokens=3)
    b = Metrics(audio_total_tokens=10, audio_input_tokens=4, audio_output_tokens=6)
    result = a + b
    assert result.audio_total_tokens == 15
    assert result.audio_input_tokens == 6
    assert result.audio_output_tokens == 9

  def test_adds_cache_tokens(self):
    a = Metrics(cache_read_tokens=10, cache_write_tokens=5)
    b = Metrics(cache_read_tokens=20, cache_write_tokens=15)
    result = a + b
    assert result.cache_read_tokens == 30
    assert result.cache_write_tokens == 20

  def test_adds_reasoning_tokens(self):
    a = Metrics(reasoning_tokens=100)
    b = Metrics(reasoning_tokens=200)
    result = a + b
    assert result.reasoning_tokens == 300

  def test_adds_cost_both_present(self):
    a = Metrics(cost=0.01)
    b = Metrics(cost=0.02)
    result = a + b
    assert result.cost == pytest.approx(0.03)

  def test_cost_left_only(self):
    a = Metrics(cost=0.01)
    b = Metrics()
    result = a + b
    assert result.cost == 0.01

  def test_cost_right_only(self):
    a = Metrics()
    b = Metrics(cost=0.02)
    result = a + b
    assert result.cost == 0.02

  def test_cost_neither_stays_none(self):
    a = Metrics()
    b = Metrics()
    result = a + b
    assert result.cost is None

  def test_adds_duration_both_present(self):
    a = Metrics(duration=1.0)
    b = Metrics(duration=2.0)
    result = a + b
    assert result.duration == pytest.approx(3.0)

  def test_duration_left_only(self):
    a = Metrics(duration=1.5)
    b = Metrics()
    result = a + b
    assert result.duration == 1.5

  def test_duration_neither_stays_none(self):
    a = Metrics()
    b = Metrics()
    result = a + b
    assert result.duration is None

  def test_merges_provider_metrics(self):
    a = Metrics(provider_metrics={"key_a": "val_a"})
    b = Metrics(provider_metrics={"key_b": "val_b"})
    result = a + b
    assert result.provider_metrics == {"key_a": "val_a", "key_b": "val_b"}

  def test_merges_additional_metrics(self):
    a = Metrics(additional_metrics={"x": 1})
    b = Metrics(additional_metrics={"y": 2})
    result = a + b
    assert result.additional_metrics == {"x": 1, "y": 2}

  def test_adds_time_to_first_token_both_present(self):
    a = Metrics(time_to_first_token=0.1)
    b = Metrics(time_to_first_token=0.2)
    result = a + b
    assert result.time_to_first_token == pytest.approx(0.3)

  def test_returns_new_instance(self):
    a = Metrics(input_tokens=10)
    b = Metrics(input_tokens=20)
    result = a + b
    assert result is not a
    assert result is not b


# ---------------------------------------------------------------------------
# Metrics: __radd__() for sum()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMetricsRadd:
  """Metrics.__radd__() supports sum() with integer start value."""

  def test_radd_with_int_returns_self(self):
    m = Metrics(input_tokens=50)
    result = 0 + m
    assert result.input_tokens == 50

  def test_sum_of_metrics_list(self):
    metrics_list = [
      Metrics(input_tokens=10, output_tokens=5),
      Metrics(input_tokens=20, output_tokens=10),
      Metrics(input_tokens=30, output_tokens=15),
    ]
    total = sum(metrics_list)
    assert total.input_tokens == 60
    assert total.output_tokens == 30


# ---------------------------------------------------------------------------
# Metrics: to_dict()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMetricsToDict:
  """Metrics.to_dict() returns a clean dict with zero/None/empty values filtered out."""

  def test_empty_metrics_returns_empty_dict(self):
    m = Metrics()
    d = m.to_dict()
    assert d == {}

  def test_nonzero_tokens_included(self):
    m = Metrics(input_tokens=100, output_tokens=50, total_tokens=150)
    d = m.to_dict()
    assert d["input_tokens"] == 100
    assert d["output_tokens"] == 50
    assert d["total_tokens"] == 150

  def test_zero_tokens_excluded(self):
    m = Metrics(input_tokens=0, output_tokens=100)
    d = m.to_dict()
    assert "input_tokens" not in d
    assert d["output_tokens"] == 100

  def test_none_cost_excluded(self):
    m = Metrics()
    d = m.to_dict()
    assert "cost" not in d

  def test_nonzero_cost_included(self):
    m = Metrics(cost=0.005)
    d = m.to_dict()
    assert d["cost"] == 0.005

  def test_timer_excluded(self):
    m = Metrics(input_tokens=10)
    m.start_timer()
    m.stop_timer()
    d = m.to_dict()
    assert "timer" not in d

  def test_duration_included_when_set(self):
    m = Metrics(duration=2.5)
    d = m.to_dict()
    assert d["duration"] == 2.5

  def test_provider_metrics_included_when_nonempty(self):
    m = Metrics(provider_metrics={"key": "val"})
    d = m.to_dict()
    assert d["provider_metrics"] == {"key": "val"}

  def test_empty_provider_metrics_excluded(self):
    m = Metrics(provider_metrics={})
    d = m.to_dict()
    assert "provider_metrics" not in d

  def test_none_duration_excluded(self):
    m = Metrics()
    d = m.to_dict()
    assert "duration" not in d


# ---------------------------------------------------------------------------
# Metrics: timer integration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMetricsTimer:
  """Metrics timer start/stop and time_to_first_token."""

  def test_start_timer_creates_timer(self):
    m = Metrics()
    m.start_timer()
    assert m.timer is not None

  def test_stop_timer_sets_duration(self):
    m = Metrics()
    m.start_timer()
    m.stop_timer(set_duration=True)
    assert m.duration is not None
    assert m.duration >= 0

  def test_stop_timer_without_setting_duration(self):
    m = Metrics()
    m.start_timer()
    m.stop_timer(set_duration=False)
    assert m.duration is None

  def test_stop_timer_without_start_is_noop(self):
    m = Metrics()
    m.stop_timer()  # timer is None, should not raise
    assert m.duration is None

  def test_set_time_to_first_token(self):
    m = Metrics()
    m.start_timer()
    m.set_time_to_first_token()
    assert m.time_to_first_token is not None
    assert m.time_to_first_token >= 0

  def test_set_time_to_first_token_without_timer_is_noop(self):
    m = Metrics()
    m.set_time_to_first_token()  # timer is None
    assert m.time_to_first_token is None
