"""Status bar — bottom bar showing model, status, metrics, cost."""

from __future__ import annotations

import contextlib
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

# Pricing per 1M tokens: (input_price, output_price)
_MODEL_PRICING: dict[str, tuple[float, float]] = {
  # OpenAI
  "gpt-4o": (2.50, 10.00),
  "gpt-4o-mini": (0.15, 0.60),
  "gpt-4.1": (2.00, 8.00),
  "gpt-4.1-mini": (0.40, 1.60),
  "gpt-4.1-nano": (0.10, 0.40),
  "o3": (10.00, 40.00),
  "o3-mini": (1.10, 4.40),
  "o4-mini": (1.10, 4.40),
  # Anthropic
  "claude-opus-4": (15.00, 75.00),
  "claude-sonnet-4": (3.00, 15.00),
  "claude-haiku": (0.80, 4.00),
  # DeepSeek
  "deepseek-chat": (0.14, 0.28),
  "deepseek-reasoner": (0.55, 2.19),
  # Google
  "gemini-2.0-flash": (0.10, 0.40),
  "gemini-2.5-pro": (1.25, 10.00),
  "gemini-2.5-flash": (0.15, 0.60),
  # Mistral
  "mistral-large": (2.00, 6.00),
  "mistral-small": (0.10, 0.30),
  # xAI
  "grok-3": (3.00, 15.00),
  "grok-3-mini": (0.30, 0.50),
}


def _lookup_pricing(model_name: str) -> tuple[float, float] | None:
  """Find pricing for a model by substring match (longest key wins)."""
  lower = model_name.lower()
  best_key = ""
  best_pricing: tuple[float, float] | None = None
  for key, pricing in _MODEL_PRICING.items():
    if key in lower and len(key) > len(best_key):
      best_key = key
      best_pricing = pricing
  return best_pricing


class StatusBar(Widget):
  """Bottom status bar showing agent state and metrics.

  Displays: model name | status | token count | cost | latency | total time | turn | phase
  Tokens and cost accumulate across turns within a run and reset on new runs.
  """

  DEFAULT_CSS = """
  StatusBar {
    dock: bottom;
    height: 1;
    background: $surface;
    color: $text-muted;
  }

  StatusBar Horizontal {
    height: 1;
    width: 100%;
  }

  StatusBar .status-model {
    padding: 0 1;
    width: auto;
    color: $accent;
  }

  StatusBar .status-separator {
    width: 3;
    content-align: center middle;
    color: $text-disabled;
  }

  StatusBar .status-state {
    padding: 0 1;
    width: auto;
  }

  StatusBar .status-state.ready {
    color: $success;
  }

  StatusBar .status-state.running {
    color: $warning;
  }

  StatusBar .status-state.error {
    color: $error;
  }

  StatusBar .status-tokens {
    padding: 0 1;
    width: auto;
    color: $text-muted;
  }

  StatusBar .status-cost {
    padding: 0 1;
    width: auto;
    color: $success-darken-1;
  }

  StatusBar .status-latency {
    padding: 0 1;
    width: auto;
    color: $text-muted;
  }

  StatusBar .status-total-time {
    padding: 0 1;
    width: auto;
    color: $text-muted;
  }

  StatusBar .status-turn {
    padding: 0 1;
    width: auto;
    color: $text-muted;
  }

  StatusBar .status-spacer {
    width: 1fr;
  }

  StatusBar .status-session {
    padding: 0 1;
    width: auto;
    color: $text-disabled;
  }

  StatusBar .status-phase {
    padding: 0 1;
    width: auto;
    color: $text-muted;
    text-style: italic;
  }
  """

  model_name: reactive[str] = reactive("", layout=True)
  session_name: reactive[str] = reactive("")
  status: reactive[str] = reactive("Ready", layout=True)
  total_tokens: reactive[int] = reactive(0)
  input_tokens: reactive[int] = reactive(0)
  output_tokens: reactive[int] = reactive(0)
  ttft_ms: reactive[Optional[float]] = reactive(None)
  total_time_ms: reactive[Optional[float]] = reactive(None)
  turn: reactive[int] = reactive(0)
  phase: reactive[str] = reactive("")

  def __init__(self, model_name: str = "") -> None:
    super().__init__()
    self.model_name = model_name

  def compose(self) -> ComposeResult:
    with Horizontal():
      yield Static("", id="status-model", classes="status-model")
      yield Static(" \u2502 ", classes="status-separator")
      yield Static("", id="status-state", classes="status-state ready")
      yield Static("", id="status-tokens", classes="status-tokens")
      yield Static("", id="status-cost", classes="status-cost")
      yield Static("", id="status-latency", classes="status-latency")
      yield Static("", id="status-total-time", classes="status-total-time")
      yield Static("", id="status-turn", classes="status-turn")
      yield Static("", classes="status-spacer")
      yield Static("", id="status-session", classes="status-session")
      yield Static("", id="status-phase", classes="status-phase")

  def watch_model_name(self, value: str) -> None:
    with contextlib.suppress(Exception):
      self.query_one("#status-model", Static).update(value)

  def watch_session_name(self, value: str) -> None:
    with contextlib.suppress(Exception):
      text = f"\u2502 {value}" if value else ""
      self.query_one("#status-session", Static).update(text)

  def watch_status(self, value: str) -> None:
    try:
      widget = self.query_one("#status-state", Static)
      widget.remove_class("ready", "running", "error")
      # Text indicators alongside colors for accessibility
      if value == "Ready":
        widget.update("\u2713 Ready")
        widget.add_class("ready")
      elif value in ("Running", "Thinking"):
        widget.update("\u21bb " + value)
        widget.add_class("running")
      elif value == "Error":
        widget.update("\u2717 Error")
        widget.add_class("error")
      else:
        widget.update(value)
    except Exception:
      pass

  def watch_total_tokens(self, value: int) -> None:
    try:
      text = f"{value:,} tokens" if value > 0 else ""
      self.query_one("#status-tokens", Static).update(f" \u2502 {text}" if text else "")
    except Exception:
      pass

  def watch_ttft_ms(self, value: Optional[float]) -> None:
    try:
      text = f"TTFT {value:.0f}ms" if value is not None else ""
      self.query_one("#status-latency", Static).update(f" \u2502 {text}" if text else "")
    except Exception:
      pass

  def watch_total_time_ms(self, value: Optional[float]) -> None:
    try:
      if value is not None:
        if value >= 1000:
          text = f"{value / 1000:.1f}s"
        else:
          text = f"{value:.0f}ms"
      else:
        text = ""
      self.query_one("#status-total-time", Static).update(f" \u2502 {text}" if text else "")
    except Exception:
      pass

  def watch_turn(self, value: int) -> None:
    try:
      text = f"Turn {value}" if value > 0 else ""
      self.query_one("#status-turn", Static).update(f" \u2502 {text}" if text else "")
    except Exception:
      pass

  def watch_phase(self, value: str) -> None:
    with contextlib.suppress(Exception):
      self.query_one("#status-phase", Static).update(value)

  def add_turn_tokens(self, input_tokens: int, output_tokens: int) -> None:
    """Accumulate token counts from a model call."""
    self.input_tokens += input_tokens
    self.output_tokens += output_tokens
    self.total_tokens = self.input_tokens + self.output_tokens
    self._update_cost()

  def _update_cost(self) -> None:
    """Recalculate and display estimated cost."""
    try:
      pricing = _lookup_pricing(self.model_name)
      if pricing is None or (self.input_tokens == 0 and self.output_tokens == 0):
        self.query_one("#status-cost", Static).update("")
        return
      input_price, output_price = pricing
      cost = (self.input_tokens / 1_000_000) * input_price + (self.output_tokens / 1_000_000) * output_price
      if cost < 0.01:
        text = f"${cost:.4f}"
      elif cost < 1.00:
        text = f"${cost:.3f}"
      else:
        text = f"${cost:.2f}"
      self.query_one("#status-cost", Static).update(f"\u2502 {text}")
    except Exception:
      pass

  @property
  def estimated_cost(self) -> float | None:
    """Return the estimated cost in USD, or None if unknown model."""
    pricing = _lookup_pricing(self.model_name)
    if pricing is None:
      return None
    input_price, output_price = pricing
    return (self.input_tokens / 1_000_000) * input_price + (self.output_tokens / 1_000_000) * output_price

  def set_running(self) -> None:
    """Set status to running and reset per-run metrics."""
    self.status = "Running"
    self.phase = ""
    self.turn = 0
    self.total_tokens = 0
    self.input_tokens = 0
    self.output_tokens = 0
    self.ttft_ms = None
    self.total_time_ms = None
    with contextlib.suppress(Exception):
      self.query_one("#status-cost", Static).update("")

  def set_ready(self) -> None:
    """Set status to ready."""
    self.status = "Ready"
    self.phase = ""

  def set_error(self) -> None:
    """Set status to error."""
    self.status = "Error"
    self.phase = ""
