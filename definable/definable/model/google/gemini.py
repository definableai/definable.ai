from dataclasses import dataclass
from os import getenv
from typing import Any, Dict, List, Optional, Sequence, Type, Union

from pydantic import BaseModel

from definable.exceptions import ModelAuthenticationError
from definable.model.base import Model
from definable.model.google.transform import GEMINI_TRANSFORM
from definable.model.http import apost_json, post_json
from definable.model.message import Message

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


@dataclass
class Gemini(Model):
  """Google Gemini (Developer API) over raw HTTP. The dialect lives in
  `transform.py`; this class is config + the hooks the transform calls.

  Model id rides in the URL path; sampling params nest under `generationConfig`.
  """

  id: str = "gemini-2.0-flash-001"
  name: str = "Gemini"
  provider: str = "Google"

  supports_native_structured_outputs: bool = True
  # Gemini streams cumulative usage per chunk — keep the latest, don't sum.
  collect_metrics_on_completion: bool = True

  transform = GEMINI_TRANSFORM

  # Request parameters (mapped to generationConfig camelCase by get_request_params)
  temperature: Optional[float] = None
  top_p: Optional[float] = None
  top_k: Optional[int] = None
  max_output_tokens: Optional[int] = None
  stop_sequences: Optional[List[str]] = None
  presence_penalty: Optional[float] = None
  frequency_penalty: Optional[float] = None
  seed: Optional[int] = None
  thinking_budget: Optional[int] = None  # -1 dynamic, 0 disable, >0 cap (Gemini 2.5)
  include_thoughts: Optional[bool] = None
  safety_settings: Optional[List[Dict[str, Any]]] = None
  request_params: Optional[Dict[str, Any]] = None

  # Client parameters
  default_headers: Optional[Dict[str, str]] = None

  def __post_init__(self) -> None:
    super().__post_init__()
    if self.thinking_budget is not None or self.include_thoughts is not None:
      self.supports_native_thinking = True

  # --- auth + endpoint ------------------------------------------------------

  def _resolve_auth(self) -> None:
    if not self.api_key:
      self.api_key = getenv("GEMINI_API_KEY") or getenv("GOOGLE_API_KEY")
      if not self.api_key:
        raise ModelAuthenticationError(
          message="GEMINI_API_KEY not set. Please set the GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable.",
          model_name=self.name,
        )

  def _endpoint(self, stream: bool = False, count_tokens: bool = False) -> str:
    base = str(self.base_url or DEFAULT_GEMINI_BASE_URL).rstrip("/")
    if count_tokens:
      return f"{base}/models/{self.id}:countTokens"
    if stream:
      return f"{base}/models/{self.id}:streamGenerateContent?alt=sse"
    return f"{base}/models/{self.id}:generateContent"

  def _build_headers(self) -> Dict[str, str]:
    headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key or ""}
    if self.default_headers:
      headers.update(self.default_headers)
    return headers

  # --- generationConfig (called by the transform) ---------------------------

  def get_request_params(self, response_format: Optional[Union[Dict, Type[BaseModel]]] = None) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if self.temperature is not None:
      cfg["temperature"] = self.temperature
    if self.max_output_tokens is not None:
      cfg["maxOutputTokens"] = self.max_output_tokens
    if self.top_p is not None:
      cfg["topP"] = self.top_p
    if self.top_k is not None:
      cfg["topK"] = self.top_k
    if self.stop_sequences:
      cfg["stopSequences"] = self.stop_sequences
    if self.presence_penalty is not None:
      cfg["presencePenalty"] = self.presence_penalty
    if self.frequency_penalty is not None:
      cfg["frequencyPenalty"] = self.frequency_penalty
    if self.seed is not None:
      cfg["seed"] = self.seed

    thinking: Dict[str, Any] = {}
    if self.thinking_budget is not None:
      thinking["thinkingBudget"] = self.thinking_budget
    if self.include_thoughts is not None:
      thinking["includeThoughts"] = self.include_thoughts
    if thinking:
      cfg["thinkingConfig"] = thinking

    if response_format is not None:
      cfg["responseMimeType"] = "application/json"
      if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        from definable.model.google.transform import _sanitize_schema

        cfg["responseSchema"] = _sanitize_schema(response_format.model_json_schema())

    if self.request_params:
      cfg.update(self.request_params)
    return cfg

  # --- token counting -------------------------------------------------------

  def count_tokens(
    self,
    messages: List[Message],
    tools: Optional[Sequence[Any]] = None,
    output_schema: Optional[Union[Dict, Type[BaseModel]]] = None,
  ) -> int:
    from definable.model.google.transform import _to_contents

    self._resolve_auth()
    contents, system_instruction = _to_contents(messages, compress=True)
    body: Dict[str, Any] = {"contents": contents}
    if system_instruction:
      body["systemInstruction"] = system_instruction
    raw = post_json(self._endpoint(count_tokens=True), self._build_headers(), body, self.timeout, self.name, self.id)
    return int(raw.get("totalTokens", 0))

  async def acount_tokens(
    self,
    messages: List[Message],
    tools: Optional[Sequence[Any]] = None,
    output_schema: Optional[Union[Dict, Type[BaseModel]]] = None,
  ) -> int:
    from definable.model.google.transform import _to_contents

    self._resolve_auth()
    contents, system_instruction = _to_contents(messages, compress=True)
    body: Dict[str, Any] = {"contents": contents}
    if system_instruction:
      body["systemInstruction"] = system_instruction
    raw = await apost_json(self._endpoint(count_tokens=True), self._build_headers(), body, self.timeout, self.name, self.id)
    return int(raw.get("totalTokens", 0))
