"""Anthropic _handle_api_error: 529→rate limit, ctx-too-long→ContextWindowExceededError."""

from unittest.mock import MagicMock

import pytest

from definable.exceptions import ContextWindowExceededError, ModelProviderError, ModelRateLimitError
from definable.model.anthropic.claude import Claude


def _make_status_error(status_code, message):
  from anthropic import APIStatusError

  response = MagicMock()
  response.status_code = status_code
  err = APIStatusError(message=message, response=response, body=None)
  err.status_code = status_code
  err.message = message
  return err


def test_overloaded_529_raises_rate_limit():
  model = Claude(id="claude-sonnet-4-5-20250929", api_key="test")
  err = _make_status_error(529, "Service overloaded, please retry")
  with pytest.raises(ModelRateLimitError):
    model._handle_api_error(err)


def test_overloaded_text_raises_rate_limit_even_without_529():
  model = Claude(id="claude-sonnet-4-5-20250929", api_key="test")
  err = _make_status_error(500, "Backend overloaded right now")
  with pytest.raises(ModelRateLimitError):
    model._handle_api_error(err)


def test_prompt_too_long_raises_context_window_error():
  model = Claude(id="claude-sonnet-4-5-20250929", api_key="test")
  err = _make_status_error(400, "prompt is too long: 250000 tokens")
  with pytest.raises(ContextWindowExceededError):
    model._handle_api_error(err)


def test_other_status_raises_generic_provider_error():
  model = Claude(id="claude-sonnet-4-5-20250929", api_key="test")
  err = _make_status_error(400, "invalid_request_error: missing field")
  with pytest.raises(ModelProviderError) as exc:
    model._handle_api_error(err)
  assert not isinstance(exc.value, (ModelRateLimitError, ContextWindowExceededError))
