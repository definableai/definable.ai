import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Type, Union
from uuid import UUID, uuid4


class ModelCapability(Enum):
  TEXT_GENERATION = "text_generation"
  IMAGE_UNDERSTANDING = "image_understanding"
  IMAGE_GENERATION = "image_generation"
  AUDIO_TRANSCRIPTION = "audio_transcription"
  AUDIO_GENERATION = "audio_generation"
  EMBEDDING = "embedding"
  FUNCTION_CALLING = "function_calling"
  CODE_GENERATION = "code_generation"
  STRUCTURED_OUTPUT = "structured_output"
  REASONING = "reasoning"


def requires_capability(capability: ModelCapability):
  """
  Decorator to check if a model has a required capability before executing a method.

  Args:
      capability: The capability required to execute the method

  Returns:
      Decorated function that checks for the capability
  """

  def decorator(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
      if not self.has_capability(capability):
        raise UnsupportedCapabilityError(f"Model {self.name} (v{self.version}) does not support {capability.value}")
      return func(self, *args, **kwargs)

    @wraps(func)
    async def async_wrapper(self, *args, **kwargs):
      if not self.has_capability(capability):
        raise UnsupportedCapabilityError(f"Model {self.name} (v{self.version}) does not support {capability.value}")
      return await func(self, *args, **kwargs)

    # Return appropriate wrapper based on if the function is async
    if asyncio.iscoroutinefunction(func):
      return async_wrapper
    return wrapper

  return decorator


class UnsupportedCapabilityError(Exception):
  """Raised when attempting to use a capability not supported by the model."""

  pass


@dataclass
class ModelMetrics:
  """Track performance metrics of the model during operation."""

  start_time: Optional[float] = None
  end_time: Optional[float] = None
  time_to_first_token: Optional[float] = None

  input_tokens: int = 0
  output_tokens: int = 0
  total_tokens: int = 0
  cached_tokens: int = 0

  input_audio_tokens: int = 0
  output_audio_tokens: int = 0
  reasoning_tokens: int = 0

  additional_metrics: Dict[str, Any] = field(default_factory=dict)

  def start_timer(self) -> None:
    self.start_time = time.time()

  def stop_timer(self) -> None:
    self.end_time = time.time()

  def set_time_to_first_token(self) -> None:
    if self.start_time and not self.time_to_first_token:
      self.time_to_first_token = time.time() - self.start_time

  @property
  def elapsed(self) -> float:
    """Total elapsed time in seconds."""
    if self.start_time is None:
      return 0.0
    end = self.end_time or time.time()
    return end - self.start_time

  @property
  def tokens_per_second(self) -> float:
    """Token generation throughput."""
    if not self.elapsed or not self.output_tokens:
      return 0.0
    return self.output_tokens / self.elapsed


@dataclass
class ProviderInfo:
  """Information about the model provider."""

  name: str
  version: Optional[str] = None
  api_endpoint: Optional[str] = None
  api_key: Optional[str] = None
  organization_id: Optional[str] = None
  rate_limits: Optional[Dict[str, Any]] = None


@dataclass
class ModelConfig:
  """Configuration parameters for model generation."""

  temperature: float = 0.7
  top_p: float = 1.0
  top_k: Optional[int] = None
  max_tokens: Optional[int] = None
  presence_penalty: float = 0.0
  frequency_penalty: float = 0.0
  stop_sequences: List[str] = field(default_factory=list)
  response_format: Optional[Dict[str, Any]] = None
  seed: Optional[int] = None
  stream: bool = False
  timeout: Optional[float] = None
  additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
  """Structured response from a model."""

  id: UUID = field(default_factory=uuid4)
  created_at: float = field(default_factory=time.time)

  content: Optional[str] = None
  role: Optional[str] = None
  finish_reason: Optional[str] = None


@dataclass
class Model(ABC):
  """Base class for all LLM implementations."""

  # Core identity
  id: str
  name: Optional[str] = None
  version: Optional[str] = None

  # Provider information
  provider: ProviderInfo = field(default_factory=lambda: ProviderInfo(name="generic"))

  # Technical specifications
  parameter_count: Optional[int] = None
  context_window: int = 4096

  # Capabilities
  capabilities: List[ModelCapability] = field(default_factory=list)

  # Default configuration
  default_config: ModelConfig = field(default_factory=ModelConfig)

  # System behavior
  system_prompt: Optional[str] = None
  instructions: Optional[List[str]] = None

  # Message roles
  system_role: str = "system"
  user_role: str = "user"
  assistant_role: str = "assistant"
  tool_role: str = "tool"

  # Internal state
  _function_call_stack: Optional[List[Any]] = None
  _tool_choice: Optional[Union[str, Dict[str, Any]]] = None

  # Extension point for plugins
  _plugins: List[Any] = field(default_factory=list)

  def __post_init__(self):
    """Initialize after dataclass fields are set."""
    if not self.name and self.id:
      self.name = self.id

  def to_dict(self) -> Dict[str, Any]:
    """Convert core model information to dictionary."""
    fields = {"id", "name", "version", "family", "parameter_count", "context_window"}
    return {field: getattr(self, field) for field in fields if getattr(self, field) is not None}

  def has_capability(self, capability: ModelCapability) -> bool:
    """
    Check if the model has a specific capability.

    Args:
        capability: The capability to check for

    Returns:
        bool: True if the model has the capability, False otherwise
    """
    return capability in self.capabilities

  @property
  def supports_streaming(self) -> bool:
    """Check if the model supports streaming."""
    # In this improved architecture, we derive this from capabilities
    # rather than having a separate flag
    # But we keep the property for backward compatibility
    return self.has_capability(ModelCapability.TEXT_GENERATION)

  @property
  def supports_vision(self) -> bool:
    """Check if the model supports vision."""
    return self.has_capability(ModelCapability.IMAGE_UNDERSTANDING)

  @property
  def supports_audio(self) -> bool:
    """Check if the model supports audio."""
    return self.has_capability(ModelCapability.AUDIO_TRANSCRIPTION) or self.has_capability(ModelCapability.AUDIO_GENERATION)

  @property
  def supports_function_calling(self) -> bool:
    """Check if the model supports function calling."""
    return self.has_capability(ModelCapability.FUNCTION_CALLING)

  @property
  def supports_native_structured_outputs(self) -> bool:
    """Check if the model supports structured outputs."""
    return self.has_capability(ModelCapability.STRUCTURED_OUTPUT)

  @property
  def supports_json_schema_outputs(self) -> bool:
    """Check if the model supports JSON schema outputs."""
    return self.has_capability(ModelCapability.STRUCTURED_OUTPUT)

  @property
  def supports_agent_workflow(self) -> bool:
    """Check if the model supports agent workflow."""
    return True  # This is a base capability we assume all models support

  @abstractmethod
  def invoke(
    self,
    messages: List[Any],
    config: Optional[ModelConfig] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    response_format: Optional[Union[Dict, Type[Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
  ) -> Any:
    """
    Synchronously generate a completion from the model.

    Args:
        messages: List of messages in the conversation
        config: Configuration parameters for generation
        tools: List of tool definitions
        response_format: Structured output format
        tool_choice: Controls which tools can be called

    Returns:
        Raw response from the model provider
    """
    pass

  @abstractmethod
  async def ainvoke(
    self,
    messages: List[Any],
    config: Optional[ModelConfig] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    response_format: Optional[Union[Dict, Type[Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
  ) -> Any:
    """
    Asynchronously generate a completion from the model.

    Args:
        messages: List of messages in the conversation
        config: Configuration parameters for generation
        tools: List of tool definitions
        response_format: Structured output format
        tool_choice: Controls which tools can be called

    Returns:
        Raw response from the model provider
    """
    pass

  @abstractmethod
  def invoke_stream(
    self,
    messages: List[Any],
    config: Optional[ModelConfig] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    response_format: Optional[Union[Dict, Type[Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
  ) -> Iterator[Any]:
    """
    Synchronously stream a completion from the model.

    Args:
        messages: List of messages in the conversation
        config: Configuration parameters for generation
        tools: List of tool definitions
        response_format: Structured output format
        tool_choice: Controls which tools can be called

    Returns:
        Iterator of response chunks from the model provider
    """
    pass

  @abstractmethod
  async def ainvoke_stream(
    self,
    messages: List[Any],
    config: Optional[ModelConfig] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    response_format: Optional[Union[Dict, Type[Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
  ) -> AsyncIterator[Any]:
    """
    Asynchronously stream a completion from the model.

    Args:
        messages: List of messages in the conversation
        config: Configuration parameters for generation
        tools: List of tool definitions
        response_format: Structured output format
        tool_choice: Controls which tools can be called

    Returns:
        Async iterator of response chunks from the model provider
    """
    pass

  @abstractmethod
  def parse_provider_response(self, response: Any, **kwargs) -> ModelResponse:
    """
    Parse the raw response from the model provider.

    Args:
        response: Raw response from the model provider

    Returns:
        ModelResponse: Structured response data
    """
    pass

  @abstractmethod
  def parse_provider_response_delta(self, response: Any) -> ModelResponse:
    """
    Parse a streaming response chunk from the model provider.

    Args:
        response: Raw response chunk from the model provider

    Returns:
        ModelResponse: Parsed response delta
    """
    pass

  @requires_capability(ModelCapability.EMBEDDING)
  def create_embedding(self, input_text: Union[str, List[str]]) -> Any:
    """
    Generate embeddings for input text.

    Args:
        input_text: Single text or list of texts to embed

    Returns:
        Embedding vector(s)
    """
    raise NotImplementedError("Embedding generation must be implemented in derived class")

  @requires_capability(ModelCapability.EMBEDDING)
  async def acreate_embedding(self, input_text: Union[str, List[str]]) -> Any:
    """
    Asynchronously generate embeddings for input text.

    Args:
        input_text: Single text or list of texts to embed

    Returns:
        Embedding vector(s)
    """
    raise NotImplementedError("Async embedding generation must be implemented in derived class")

  @requires_capability(ModelCapability.IMAGE_GENERATION)
  def generate_image(self, prompt: str, **kwargs) -> Any:
    """
    Generate an image from a text prompt.

    Args:
        prompt: Text description of the image to generate
        **kwargs: Additional parameters for image generation

    Returns:
        Generated image data
    """
    raise NotImplementedError("Image generation must be implemented in derived class")

  @requires_capability(ModelCapability.IMAGE_GENERATION)
  async def agenerate_image(self, prompt: str, **kwargs) -> Any:
    """
    Asynchronously generate an image from a text prompt.

    Args:
        prompt: Text description of the image to generate
        **kwargs: Additional parameters for image generation

    Returns:
        Generated image data
    """
    raise NotImplementedError("Async image generation must be implemented in derived class")

  @requires_capability(ModelCapability.AUDIO_GENERATION)
  def generate_audio(self, text: str, **kwargs) -> Any:
    """
    Generate audio from text (text-to-speech).

    Args:
        text: Text to convert to speech
        **kwargs: Additional parameters for audio generation

    Returns:
        Generated audio data
    """
    raise NotImplementedError("Audio generation must be implemented in derived class")

  @requires_capability(ModelCapability.AUDIO_GENERATION)
  async def agenerate_audio(self, text: str, **kwargs) -> Any:
    """
    Asynchronously generate audio from text (text-to-speech).

    Args:
        text: Text to convert to speech
        **kwargs: Additional parameters for audio generation

    Returns:
        Generated audio data
    """
    raise NotImplementedError("Async audio generation must be implemented in derived class")

  @requires_capability(ModelCapability.AUDIO_TRANSCRIPTION)
  def transcribe_audio(self, audio_data: Any, **kwargs) -> str:
    """
    Transcribe audio to text.

    Args:
        audio_data: Audio data to transcribe
        **kwargs: Additional parameters for transcription

    Returns:
        Transcribed text
    """
    raise NotImplementedError("Audio transcription must be implemented in derived class")

  @requires_capability(ModelCapability.AUDIO_TRANSCRIPTION)
  async def atranscribe_audio(self, audio_data: Any, **kwargs) -> str:
    """
    Asynchronously transcribe audio to text.

    Args:
        audio_data: Audio data to transcribe
        **kwargs: Additional parameters for transcription

    Returns:
        Transcribed text
    """
    raise NotImplementedError("Async audio transcription must be implemented in derived class")

  def count_tokens(self, text: str) -> int:
    """
    Count the number of tokens in a text string.

    Args:
        text: Text to count tokens for

    Returns:
        int: Token count
    """
    raise NotImplementedError("Token counting must be implemented in derived class")

  def estimate_cost(self, input_tokens: int, output_tokens: int = 0, image_tokens: int = 0, audio_seconds: float = 0.0) -> float:
    """
    Estimate the cost of a model invocation.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        image_tokens: Number of image tokens
        audio_seconds: Duration of audio in seconds

    Returns:
        float: Estimated cost in USD
    """
    cost = (
      input_tokens * self.cost_per_input_token
      + output_tokens * self.cost_per_output_token
      + image_tokens * self.cost_per_image_token
      + audio_seconds * self.cost_per_audio_second
    )
    return cost

  def set_default_parameters(self, **params) -> None:
    """Update the default parameters for the model."""
    for key, value in params.items():
      if hasattr(self.default_config, key):
        setattr(self.default_config, key, value)
      else:
        self.default_config.additional_params[key] = value

  def clear(self) -> None:
    """Reset internal state of the model."""
    self._function_call_stack = None

  def get_system_message(self) -> Optional[str]:
    """Get the system message for this model."""
    return self.system_prompt

  def get_provider_name(self) -> str:
    """Get the name of the model provider."""
    if self.provider and self.provider.name:
      return self.provider.name
    return "unknown"

  # API compatibility methods
  def response(self, messages: List[Any], *args, **kwargs) -> ModelResponse:
    """Generate a response using the complete agent workflow."""
    # Implementation would handle all the message parsing, tool calling, etc.
    raise NotImplementedError("Agent workflow must be implemented in derived class")

  async def aresponse(self, messages: List[Any], *args, **kwargs) -> ModelResponse:
    """Generate an async response using the complete agent workflow."""
    raise NotImplementedError("Async agent workflow must be implemented in derived class")

  def response_stream(self, messages: List[Any], *args, **kwargs) -> Iterator[ModelResponse]:
    """Stream a response using the complete agent workflow."""
    raise NotImplementedError("Streaming agent workflow must be implemented in derived class")

  async def aresponse_stream(self, messages: List[Any], *args, **kwargs) -> AsyncIterator[ModelResponse]:
    """Stream an async response using the complete agent workflow."""
    raise NotImplementedError("Async streaming agent workflow must be implemented in derived class")

  # Extension point for plugins and custom behavior
  def register_plugin(self, plugin: Any) -> None:
    """
    Register a plugin to extend model functionality.

    Args:
        plugin: Plugin instance to register
    """
    if plugin not in self._plugins:
      self._plugins.append(plugin)
      if hasattr(plugin, "initialize"):
        plugin.initialize(self)

  def unregister_plugin(self, plugin: Any) -> None:
    """
    Unregister a plugin.

    Args:
        plugin: Plugin instance to unregister
    """
    if plugin in self._plugins:
      if hasattr(plugin, "cleanup"):
        plugin.cleanup(self)
      self._plugins.remove(plugin)

  def get_plugins(self) -> List[Any]:
    """
    Get all registered plugins.

    Returns:
        List of registered plugin instances
    """
    return self._plugins.copy()


class ModelFactory:
  """Factory to create model instances with proper version handling."""

  @staticmethod
  def create_model(provider: str, model_id: str, version: Optional[str] = None, **kwargs) -> BaseModel:
    """
    Create a model instance based on provider and model ID.

    Args:
        provider: Provider name (e.g., "openai", "anthropic")
        model_id: Model identifier (e.g., "gpt-4", "claude-3")
        version: Optional version override
        **kwargs: Additional parameters for model initialization

    Returns:
        BaseModel: Instance of appropriate model class
    """
    # This would be implemented to return the appropriate model class
    # based on the provider, model ID, and version
    raise NotImplementedError("Model factory implementation required")
