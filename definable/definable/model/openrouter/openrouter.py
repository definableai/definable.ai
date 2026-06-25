from dataclasses import dataclass
from os import getenv
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from definable.exceptions import ModelAuthenticationError
from definable.model.openai.like import OpenAILike
from definable.model.response import ModelResponse


@dataclass
class OpenRouter(OpenAILike):
  """
  A class for using models hosted on OpenRouter.

  Attributes:
    id (str): The model id. Defaults to "gpt-4o".
    name (str): The model name. Defaults to "OpenRouter".
    provider (str): The provider name. Defaults to "OpenRouter".
    api_key (Optional[str]): The API key.
    base_url (str): The base URL. Defaults to "https://openrouter.ai/api/v1".
    max_tokens (int): The maximum number of tokens. Defaults to 1024.
    models (Optional[List[str]]): List of fallback model IDs for dynamic model routing.
  """

  id: str = "gpt-4o"
  name: str = "OpenRouter"
  provider: str = "OpenRouter"

  api_key: Optional[str] = None
  base_url: str = "https://openrouter.ai/api/v1"
  max_tokens: int = 1024
  models: Optional[List[str]] = None  # Dynamic model routing https://openrouter.ai/docs/features/model-routing

  def _get_client_params(self) -> Dict[str, Any]:
    if not self.api_key:
      self.api_key = getenv("OPENROUTER_API_KEY")
      if not self.api_key:
        raise ModelAuthenticationError(
          message="OPENROUTER_API_KEY not set. Please set the OPENROUTER_API_KEY environment variable.",
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
    request_params = super().get_request_params(response_format=response_format, tools=tools, tool_choice=tool_choice)

    # Add fallback models to extra_body if specified
    if self.models:
      extra_body = request_params.get("extra_body") or {}
      extra_body["models"] = self.models
      request_params["extra_body"] = extra_body

    return request_params

  def _augment_response(self, raw: Dict[str, Any], model_response: ModelResponse) -> None:
    choices = raw.get("choices") or []
    if choices and (choices[0].get("message") or {}).get("reasoning_details"):
      if model_response.provider_data is None:
        model_response.provider_data = {}
      model_response.provider_data["reasoning_details"] = choices[0]["message"]["reasoning_details"]

  def _augment_delta(self, raw_delta: Dict[str, Any], model_response: ModelResponse) -> None:
    choices = raw_delta.get("choices") or []
    if choices and (choices[0].get("delta") or {}).get("reasoning_details"):
      if model_response.provider_data is None:
        model_response.provider_data = {}
      model_response.provider_data["reasoning_details"] = choices[0]["delta"]["reasoning_details"]
