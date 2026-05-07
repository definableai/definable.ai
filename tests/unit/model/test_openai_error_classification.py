"""OpenAI APIStatusError classification — context_length_exceeded → ContextWindowExceededError."""

from unittest.mock import MagicMock

import pytest

from definable.exceptions import ContextWindowExceededError, ModelProviderError
from definable.model.openai.chat import OpenAIChat


def _make_api_status_error(error_body, status_code=400, raw_text="raw"):
  """Build a fake openai.APIStatusError-shaped object."""
  from openai import APIStatusError

  response = MagicMock()
  response.status_code = status_code
  if error_body is None:
    response.json.side_effect = ValueError("not json")
    response.text = raw_text
  else:
    response.json.return_value = {"error": error_body}
    response.text = raw_text

  err = APIStatusError(message="boom", response=response, body=error_body)
  return err


def test_context_length_exceeded_raises_context_window_error():
  model = OpenAIChat(id="gpt-5.4-mini", api_key="test")
  err = _make_api_status_error({"code": "context_length_exceeded", "message": "Too long"})
  with pytest.raises(ContextWindowExceededError) as exc:
    model._raise_for_api_status(err)
  assert exc.value.error_id == "context_window_exceeded"
  assert "Too long" in exc.value.message


def test_other_4xx_raises_generic_provider_error():
  model = OpenAIChat(id="gpt-5.4-mini", api_key="test")
  err = _make_api_status_error({"code": "invalid_request_error", "message": "Bad param"})
  with pytest.raises(ModelProviderError) as exc:
    model._raise_for_api_status(err)
  # Not the context-window subclass
  assert not isinstance(exc.value, ContextWindowExceededError)
  assert "Bad param" in exc.value.message


def test_non_json_body_falls_back_to_text():
  model = OpenAIChat(id="gpt-5.4-mini", api_key="test")
  err = _make_api_status_error(None, raw_text="upstream html error page")
  with pytest.raises(ModelProviderError) as exc:
    model._raise_for_api_status(err)
  assert "upstream html error page" in exc.value.message


def test_collect_metrics_on_completion_default_false():
  m = OpenAIChat(id="gpt-4o", api_key="test")
  assert m.collect_metrics_on_completion is False
