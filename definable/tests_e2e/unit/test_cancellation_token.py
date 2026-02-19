"""Unit tests for CancellationToken cooperative cancellation."""

import pytest

from definable.agent.cancellation import AgentCancelled, CancellationToken


@pytest.mark.unit
class TestCancellationToken:
  def test_token_not_cancelled_by_default(self):
    token = CancellationToken()
    assert token.is_cancelled is False
    # Should not raise when not cancelled
    token.raise_if_cancelled()

  def test_cancel_sets_flag(self):
    token = CancellationToken()
    token.cancel()
    assert token.is_cancelled is True

  def test_raise_if_cancelled_raises(self):
    token = CancellationToken()
    token.cancel()
    with pytest.raises(AgentCancelled):
      token.raise_if_cancelled()

  def test_raise_if_cancelled_message(self):
    token = CancellationToken()
    token.cancel()
    with pytest.raises(AgentCancelled, match="cancelled"):
      token.raise_if_cancelled()
