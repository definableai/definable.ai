from dataclasses import dataclass, field
from os import getenv
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from definable.exceptions import ModelAuthenticationError
from definable.model.message import Citations, UrlCitation
from definable.model.openai.like import OpenAILike
from definable.model.response import ModelResponse
from definable.utils.log import log_debug


@dataclass
class Perplexity(OpenAILike):
  """
  A class for using models hosted on Perplexity.

  Attributes:
    id (str): The model id. Defaults to "sonar".
    name (str): The model name. Defaults to "Perplexity".
    provider (str): The provider name. Defaults to "Perplexity".
    api_key (Optional[str]): The API key.
    base_url (str): The base URL. Defaults to "https://api.perplexity.ai/".
    max_tokens (int): The maximum number of tokens. Defaults to 1024.
  """

  id: str = "sonar"
  name: str = "Perplexity"
  provider: str = "Perplexity"
  # Perplexity returns cumulative token counts in each streaming chunk, so only collect on final chunk
  collect_metrics_on_completion: bool = True

  api_key: Optional[str] = field(default_factory=lambda: getenv("PERPLEXITY_API_KEY"))
  base_url: str = "https://api.perplexity.ai/"
  max_tokens: int = 1024
  top_k: Optional[float] = None

  supports_native_structured_outputs: bool = False
  supports_json_schema_outputs: bool = True

  def _get_client_params(self) -> Dict[str, Any]:
    if not self.api_key:
      self.api_key = getenv("PERPLEXITY_API_KEY")
      if not self.api_key:
        raise ModelAuthenticationError(
          message="PERPLEXITY_API_KEY not set. Please set the PERPLEXITY_API_KEY environment variable.",
          model_name=self.name,
        )
    return super()._get_client_params()

  def get_request_params(
    self,
    response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    **kwargs: Any,
  ) -> Dict[str, Any]:
    base_params: Dict[str, Any] = {
      "max_tokens": self.max_tokens,
      "temperature": self.temperature,
      "top_p": self.top_p,
      "top_k": self.top_k,
      "presence_penalty": self.presence_penalty,
      "frequency_penalty": self.frequency_penalty,
    }

    if response_format is not None:
      base_params["response_format"] = response_format

    # Filter out None values
    request_params = {k: v for k, v in base_params.items() if v is not None}
    if self.request_params:
      request_params.update(self.request_params)

    if request_params:
      log_debug(f"Calling {self.provider} with request parameters: {request_params}", log_level=2)
    return request_params

  def _augment_response(self, raw: Dict[str, Any], model_response: ModelResponse) -> None:
    """Attach Perplexity search citations to a parsed response."""
    if raw.get("citations"):
      model_response.citations = Citations(urls=[UrlCitation(url=c) for c in raw["citations"]])

  def _augment_delta(self, raw_delta: Dict[str, Any], model_response: ModelResponse) -> None:
    """Attach Perplexity search citations to a streaming delta."""
    if raw_delta.get("citations"):
      model_response.citations = Citations(urls=[UrlCitation(url=c) for c in raw_delta["citations"]])
