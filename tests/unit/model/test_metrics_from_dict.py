"""Metrics deserialization safety + ToolCallMetrics shape."""

from definable.model.metrics import MessageMetrics, Metrics, ToolCallMetrics


def test_metrics_from_dict_roundtrip():
  m = Metrics(input_tokens=10, output_tokens=20, total_tokens=30)
  d = m.to_dict()
  recovered = Metrics.from_dict(d)
  assert recovered.input_tokens == 10
  assert recovered.output_tokens == 20
  assert recovered.total_tokens == 30


def test_metrics_from_dict_drops_unknown_keys():
  m = Metrics.from_dict({"input_tokens": 5, "future_field": "ignored"})
  assert m.input_tokens == 5


def test_metrics_from_dict_skips_timer():
  # Timer should never come in via dict — ignore if it does
  m = Metrics.from_dict({"input_tokens": 1, "timer": "something"})
  assert m.timer is None


def test_message_metrics_alias():
  assert MessageMetrics is Metrics


def test_tool_call_metrics_basic():
  tc = ToolCallMetrics(tool_name="search", tool_call_id="call_abc", start_time=1.0, end_time=2.0, duration=1.0)
  assert tc.tool_name == "search"
  assert tc.duration == 1.0


def test_tool_call_metrics_to_dict_drops_none():
  tc = ToolCallMetrics(tool_name="search", duration=0.5)
  d = tc.to_dict()
  assert "tool_name" in d
  assert "duration" in d
  assert "error" not in d


def test_tool_call_metrics_from_dict_roundtrip():
  tc = ToolCallMetrics(tool_name="search", duration=1.5, error="boom")
  recovered = ToolCallMetrics.from_dict(tc.to_dict())
  assert recovered.tool_name == "search"
  assert recovered.duration == 1.5
  assert recovered.error == "boom"
