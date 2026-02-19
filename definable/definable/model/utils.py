"""Model string resolution — maps 'provider/model-id' to Model instances."""

from importlib import import_module
from typing import Dict, Optional, Tuple, Union

from definable.model.base import Model

# Provider registry: provider_name → (module_path, class_name)
# Lazy imports — only loads the provider module when actually requested.
_PROVIDER_MAP: Dict[str, Tuple[str, str]] = {
  "openai": ("definable.model.openai.chat", "OpenAIChat"),
  "deepseek": ("definable.model.deepseek.chat", "DeepSeekChat"),
  "moonshot": ("definable.model.moonshot.chat", "MoonshotChat"),
  "xai": ("definable.model.xai.xai", "xAI"),
  "anthropic": ("definable.model.anthropic.claude", "Claude"),
  "mistral": ("definable.model.mistral.mistral", "MistralChat"),
  "google": ("definable.model.google.gemini", "Gemini"),
  "perplexity": ("definable.model.perplexity.perplexity", "Perplexity"),
  "ollama": ("definable.model.ollama.chat", "Ollama"),
  "openrouter": ("definable.model.openrouter.openrouter", "OpenRouter"),
}


def get_supported_providers() -> list[str]:
  """Return sorted list of supported provider names."""
  return sorted(_PROVIDER_MAP.keys())


def resolve_model_string(model_string: str) -> Model:
  """Resolve a 'provider/model-id' string into a Model instance.

  Args:
    model_string: Format 'provider/model-id', e.g. 'openai/gpt-4o', 'deepseek/deepseek-chat'.

  Returns:
    An instantiated Model subclass with the given model ID.

  Raises:
    ValueError: If the format is invalid or the provider is not supported.
  """
  if not model_string or not isinstance(model_string, str):
    raise ValueError(f"Model string must be a non-empty string, got: {model_string!r}")

  if "/" not in model_string:
    # Bare model ID without provider prefix — default to OpenAI
    # e.g. "gpt-4o-mini" → OpenAIChat(id="gpt-4o-mini")
    provider = "openai"
    model_id = model_string.strip()
    module_path, class_name = _PROVIDER_MAP[provider]
    module = import_module(module_path)
    model_class = getattr(module, class_name)
    return model_class(id=model_id)

  provider, _, model_id = model_string.partition("/")
  provider = provider.strip().lower()
  model_id = model_id.strip()

  if not provider or not model_id:
    raise ValueError(f"Invalid model string '{model_string}'. Both provider and model-id are required: 'provider/model-id'")

  if provider not in _PROVIDER_MAP:
    supported = ", ".join(get_supported_providers())
    raise ValueError(f"Unknown model provider '{provider}'. Supported providers: {supported}")

  module_path, class_name = _PROVIDER_MAP[provider]
  module = import_module(module_path)
  model_class = getattr(module, class_name)
  return model_class(id=model_id)


def get_model(model: Union[Model, str, None]) -> Optional[Model]:
  """Resolve a model argument to a Model instance.

  Accepts a Model instance (passthrough), a 'provider/model-id' string
  (resolved via resolve_model_string), or None (returns None).
  """
  if model is None:
    return None
  if isinstance(model, Model):
    return model
  if isinstance(model, str):
    return resolve_model_string(model)
  raise ValueError(f"Model must be a Model instance, 'provider/model-id' string, or None — got {type(model).__name__}")
