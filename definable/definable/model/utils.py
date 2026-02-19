from typing import Optional, Union

from definable.model.base import Model


def _get_model_class(model_id: str, model_provider: str) -> Model:
  if model_provider == "aimlapi":
    raise ValueError(f"Model provider '{model_provider}' is not yet implemented.")

  raise ValueError(f"Model provider '{model_provider}' is not supported.")


def _parse_model_string(model_string: str) -> Model:
  if not model_string or not isinstance(model_string, str):
    raise ValueError(f"Model string must be a non-empty string, got: {model_string}")

  if ":" not in model_string:
    raise ValueError(f"Invalid model string format: '{model_string}'. Model strings should be in format '<provider>:<model_id>' e.g. 'openai:gpt-4o'")

  parts = model_string.split(":", 1)
  if len(parts) != 2:
    raise ValueError(f"Invalid model string format: '{model_string}'. Model strings should be in format '<provider>:<model_id>' e.g. 'openai:gpt-4o'")

  model_provider, model_id = parts
  model_provider = model_provider.strip().lower()
  model_id = model_id.strip()

  if not model_provider or not model_id:
    raise ValueError(f"Invalid model string format: '{model_string}'. Model strings should be in format '<provider>:<model_id>' e.g. 'openai:gpt-4o'")

  return _get_model_class(model_id, model_provider)


def get_model(model: Union[Model, str, None]) -> Optional[Model]:
  if model is None:
    return None
  elif isinstance(model, Model):
    return model
  elif isinstance(model, str):
    return _parse_model_string(model)
  else:
    raise ValueError("Model must be a Model instance, string, or None")
