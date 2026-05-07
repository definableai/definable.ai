"""ContextWindowExceededError — distinct from generic provider error."""

import pytest

from definable.exceptions import ContextWindowExceededError, ModelProviderError


def test_subclass_of_model_provider_error():
  err = ContextWindowExceededError("ctx exceeded")
  assert isinstance(err, ModelProviderError)


def test_default_status_400():
  err = ContextWindowExceededError("ctx exceeded")
  assert err.status_code == 400


def test_error_id():
  err = ContextWindowExceededError("ctx exceeded")
  assert err.error_id == "context_window_exceeded"


def test_carries_model_metadata():
  err = ContextWindowExceededError("ctx exceeded", model_name="OpenAIChat", model_id="gpt-5.4-mini")
  assert err.model_name == "OpenAIChat"
  assert err.model_id == "gpt-5.4-mini"


def test_distinguishable_from_provider_error():
  generic = ModelProviderError("oops")
  ctx = ContextWindowExceededError("ctx exceeded")
  assert generic.error_id != ctx.error_id


def test_raises_and_catches_specifically():
  with pytest.raises(ContextWindowExceededError):
    raise ContextWindowExceededError("ctx exceeded")
