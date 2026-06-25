from dataclasses import dataclass
from os import getenv
from typing import Any, Dict, List, Literal, Optional, Type, Union

import httpx
from pydantic import BaseModel

from definable.exceptions import ModelAuthenticationError
from definable.model.base import Model
from definable.model.openai.transform import OPENAI_TRANSFORM
from definable.types import ToolCallDict
from definable.utils.log import log_debug

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


@dataclass
class OpenAIChat(Model):
  """OpenAI Chat Completions over raw HTTP. The dialect lives in `transform.py`;
  this class is config + the few hooks the OpenAI transform calls. The whole
  OpenAI-schema family (xAI/DeepSeek/Moonshot/OpenRouter/Perplexity/Ollama)
  subclasses this and only changes fields/endpoint/auth/parse-extras.
  """

  id: str = "gpt-5.4-mini"
  name: str = "OpenAIChat"
  provider: str = "OpenAI"
  supports_native_structured_outputs: bool = True

  transform = OPENAI_TRANSFORM

  # Request parameters
  store: Optional[bool] = None
  reasoning_effort: Optional[str] = None
  verbosity: Optional[Literal["low", "medium", "high"]] = None
  metadata: Optional[Dict[str, Any]] = None
  frequency_penalty: Optional[float] = None
  logit_bias: Optional[Dict[str, int]] = None
  logprobs: Optional[bool] = None
  top_logprobs: Optional[int] = None
  max_tokens: Optional[int] = None
  max_completion_tokens: Optional[int] = None
  modalities: Optional[List[str]] = None
  audio: Optional[Dict[str, Any]] = None
  presence_penalty: Optional[float] = None
  seed: Optional[int] = None
  stop: Optional[Union[str, List[str]]] = None
  temperature: Optional[float] = None
  user: Optional[str] = None
  top_p: Optional[float] = None
  service_tier: Optional[str] = None
  strict_output: bool = True
  extra_headers: Optional[Dict[str, str]] = None
  extra_query: Optional[Dict[str, object]] = None
  extra_body: Optional[Dict[str, object]] = None
  request_params: Optional[Dict[str, Any]] = None
  role_map: Optional[Dict[str, str]] = None
  # Some OpenAI-compat providers (DeepSeek, Moonshot) require assistant `reasoning_content`
  # echoed back on follow-up turns. Off by default (OpenAI rejects unknown fields).
  send_reasoning_content: bool = False

  # Client parameters
  organization: Optional[str] = None
  max_retries: Optional[int] = None
  default_headers: Optional[Dict[str, str]] = None
  default_query: Optional[Dict[str, object]] = None
  http_client: Optional[Union[httpx.Client, httpx.AsyncClient]] = None
  client_params: Optional[Dict[str, Any]] = None
  async_client: Optional[Any] = None

  default_role_map = {
    "system": "developer",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
    "model": "assistant",
  }

  # --- auth + endpoint hooks ------------------------------------------------

  def _resolve_auth(self) -> None:
    self._get_client_params()

  def _get_client_params(self) -> Dict[str, Any]:
    """Resolve + validate the API key (subclasses read their own env var)."""
    if not self.api_key:
      self.api_key = getenv("OPENAI_API_KEY")
      if not self.api_key:
        raise ModelAuthenticationError(
          message="OPENAI_API_KEY not set. Please set the OPENAI_API_KEY environment variable.",
          model_name=self.name,
        )
    return {}

  def _chat_endpoint(self) -> str:
    base = str(self.base_url or DEFAULT_OPENAI_BASE_URL).rstrip("/")
    return f"{base}/chat/completions"

  def _build_headers(self, extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
    if self.organization:
      headers["OpenAI-Organization"] = self.organization
    if self.default_headers:
      headers.update(self.default_headers)
    if extra_headers:
      headers.update(extra_headers)
    return headers

  # --- request building hooks (called by the transform) ---------------------

  def get_request_params(
    self,
    response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    **kwargs: Any,
  ) -> Dict[str, Any]:
    base_params = {
      "store": self.store,
      "reasoning_effort": self.reasoning_effort,
      "verbosity": self.verbosity,
      "frequency_penalty": self.frequency_penalty,
      "logit_bias": self.logit_bias,
      "logprobs": self.logprobs,
      "top_logprobs": self.top_logprobs,
      "max_tokens": self.max_tokens,
      "max_completion_tokens": self.max_completion_tokens,
      "modalities": self.modalities,
      "audio": self.audio,
      "presence_penalty": self.presence_penalty,
      "seed": self.seed,
      "stop": self.stop,
      "temperature": self.temperature,
      "user": self.user,
      "top_p": self.top_p,
      "extra_headers": self.extra_headers,
      "extra_query": self.extra_query,
      "extra_body": self.extra_body,
      "metadata": self.metadata,
      "service_tier": self.service_tier,
    }

    if response_format is not None:
      if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        from definable.utils.models.schema_utils import get_response_schema_for_provider

        schema = get_response_schema_for_provider(response_format, "openai")
        base_params["response_format"] = {
          "type": "json_schema",
          "json_schema": {"name": response_format.__name__, "schema": schema, "strict": self.strict_output},
        }
      else:
        base_params["response_format"] = response_format

    request_params: Dict[str, Any] = {k: v for k, v in base_params.items() if v is not None}
    if tools:
      request_params["tools"] = tools
      if tool_choice is not None:
        request_params["tool_choice"] = tool_choice
    if self.request_params:
      request_params.update(self.request_params)
    if request_params:
      log_debug(f"Calling {self.provider} with request parameters: {request_params}", log_level=2)
    return request_params

  # --- response parse hooks (overridden by xAI/OpenRouter/Perplexity) -------

  def _augment_response(self, raw: Dict[str, Any], model_response: Any) -> None:
    """Hook to enrich a parsed response (e.g. citations). Default: no-op."""
    return None

  def _augment_delta(self, raw_delta: Dict[str, Any], model_response: Any) -> None:
    """Hook to enrich a parsed streaming delta. Default: no-op."""
    return None

  def parse_tool_calls(self, tool_calls_data: list[ToolCallDict]) -> list[ToolCallDict]:
    """Accumulate streamed tool-call fragments (keyed by `index`) into complete calls."""
    tool_calls: list[ToolCallDict] = []
    for _tool_call in tool_calls_data:
      _index = _tool_call.get("index") or 0
      _id = _tool_call.get("id")
      _type = _tool_call.get("type")
      _function = _tool_call.get("function") or {}
      _name = _function.get("name")
      _arguments = _function.get("arguments")
      if len(tool_calls) <= _index:
        tool_calls.extend([{}] * (_index - len(tool_calls) + 1))
      entry = tool_calls[_index]
      if not entry:
        entry["id"] = _id
        entry["type"] = _type
        entry["function"] = {"name": _name or "", "arguments": _arguments or ""}
      else:
        if _name:
          entry["function"]["name"] += _name
        if _arguments:
          entry["function"]["arguments"] += _arguments
        if _id:
          entry["id"] = _id
        if _type:
          entry["type"] = _type
    return tool_calls

  def to_dict(self) -> Dict[str, Any]:
    model_dict = super().to_dict()
    model_dict.update({
      "store": self.store,
      "reasoning_effort": self.reasoning_effort,
      "verbosity": self.verbosity,
      "frequency_penalty": self.frequency_penalty,
      "logit_bias": self.logit_bias,
      "logprobs": self.logprobs,
      "top_logprobs": self.top_logprobs,
      "max_tokens": self.max_tokens,
      "max_completion_tokens": self.max_completion_tokens,
      "modalities": self.modalities,
      "audio": self.audio,
      "presence_penalty": self.presence_penalty,
      "seed": self.seed,
      "stop": self.stop,
      "temperature": self.temperature,
      "top_p": self.top_p,
      "user": self.user,
      "service_tier": self.service_tier,
    })
    return {k: v for k, v in model_dict.items() if v is not None}
