from dataclasses import dataclass
from os import getenv
from typing import Any, Dict, List, Optional, Type, Union

from definable.exceptions import ModelAuthenticationError
from definable.model.message import Citations, UrlCitation
from definable.model.openai.like import OpenAILike
from definable.model.response import ModelResponse
from definable.utils.log import log_debug
from pydantic import BaseModel


@dataclass
class xAI(OpenAILike):
  """
  Class for interacting with the xAI API.

  Attributes:
      id (str): The ID of the language model. Defaults to "grok-3".
      name (str): The name of the API. Defaults to "xAI".
      provider (str): The provider of the API. Defaults to "xAI".
      api_key (Optional[str]): The API key for the xAI API.
      base_url (Optional[str]): The base URL for the xAI API. Defaults to "https://api.x.ai/v1".
      search_parameters (Optional[Dict[str, Any]]): Search parameters for enabling live search.
  """

  id: str = "grok-4.3-latest"
  name: str = "xAI"
  provider: str = "xAI"
  supports_native_structured_outputs: bool = False

  api_key: Optional[str] = None
  base_url: str = "https://api.x.ai/v1"

  search_parameters: Optional[Dict[str, Any]] = None

  def _get_client_params(self) -> Dict[str, Any]:
    """
    Returns client parameters for API requests, checking for XAI_API_KEY.

    Returns:
        Dict[str, Any]: A dictionary of client parameters for API requests.
    """
    if not self.api_key:
      self.api_key = getenv("XAI_API_KEY")
      if not self.api_key:
        raise ModelAuthenticationError(
          message="XAI_API_KEY not set. Please set the XAI_API_KEY environment variable.",
          model_name=self.name,
        )
    return super()._get_client_params()

  def get_request_params(  # type: ignore[override]
    self,
    response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    **kwargs: Any,
  ) -> Dict[str, Any]:
    """
    Returns keyword arguments for API requests, including search parameters.

    Returns:
        Dict[str, Any]: A dictionary of keyword arguments for API requests.
    """
    request_params = super().get_request_params(response_format=response_format, tools=tools, tool_choice=tool_choice)

    if self.search_parameters:
      existing_body = request_params.get("extra_body") or {}
      existing_body.update({"search_parameters": self.search_parameters})
      request_params["extra_body"] = existing_body

    if request_params:
      log_debug(f"Calling {self.provider} with request parameters: {request_params}", log_level=2)

    return request_params

  def _augment_response(self, raw: Dict[str, Any], model_response: ModelResponse) -> None:
    """Attach xAI live-search citations to a parsed response."""
    self._attach_citations(raw, model_response)

  def _augment_delta(self, raw_delta: Dict[str, Any], model_response: ModelResponse) -> None:
    """Attach xAI live-search citations to a streaming delta."""
    self._attach_citations(raw_delta, model_response)

  @staticmethod
  def _attach_citations(raw: Dict[str, Any], model_response: ModelResponse) -> None:
    if raw.get("citations"):
      citations = Citations()
      citations.urls = [UrlCitation(url=str(c)) for c in raw["citations"]]
      citations.raw = raw["citations"]
      model_response.citations = citations
