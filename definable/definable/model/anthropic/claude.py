from dataclasses import asdict, dataclass
from os import getenv
from typing import Any, Dict, List, Literal, Optional, Sequence, Type, Union

import httpx
from pydantic import BaseModel


class SystemPromptBlock(BaseModel):
  """A typed slice of the system prompt with per-block cache control."""

  text: str
  cache: bool = True
  ttl: Optional[Literal["5m", "1h"]] = None


from definable.exceptions import ModelAuthenticationError
from definable.model.anthropic.transform import ANTHROPIC_TRANSFORM
from definable.model.base import Model
from definable.model.http import apost_json, post_json
from definable.model.message import Message
from definable.tokens import count_schema_tokens
from definable.agent.toolkit.function import Function
from definable.utils.claude import MCPServerConfiguration, format_messages, format_tools_for_model
from definable.utils.log import log_debug

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"


@dataclass
class Claude(Model):
  """Anthropic Claude over raw HTTP. The dialect (build/parse/stream) lives in
  `transform.py`; this class is config + the hooks the transform calls."""

  NON_THINKING_MODELS = {
    "claude-3-haiku-20240307",
    "claude-3-5-haiku-20241022",
    "claude-3-5-haiku-latest",
  }

  id: str = "claude-sonnet-4-5-20250929"
  name: str = "Claude"
  provider: str = "Anthropic"

  transform = ANTHROPIC_TRANSFORM

  # Request parameters
  max_tokens: Optional[int] = 8192
  thinking: Optional[Dict[str, Any]] = None
  temperature: Optional[float] = None
  stop_sequences: Optional[List[str]] = None
  top_p: Optional[float] = None
  top_k: Optional[int] = None
  # Master prompt-caching switch (default on). Caches the system prompt, tool
  # schemas, and the rolling conversation prefix so each agentic turn re-reads
  # them at ~10% input cost instead of re-billing the full prefix. Set False to
  # disable caching entirely.
  cache_system_prompt: Optional[bool] = True
  extended_cache_time: Optional[bool] = False
  system_prompt_blocks: Optional[List[SystemPromptBlock]] = None
  request_params: Optional[Dict[str, Any]] = None

  # Anthropic beta + experimental features
  betas: Optional[List[str]] = None
  context_management: Optional[Dict[str, Any]] = None
  mcp_servers: Optional[List[MCPServerConfiguration]] = None
  skills: Optional[List[Dict[str, str]]] = None

  # Client parameters
  auth_token: Optional[str] = None
  default_headers: Optional[Dict[str, str]] = None
  http_client: Optional[Union[httpx.Client, httpx.AsyncClient]] = None
  client_params: Optional[Dict[str, Any]] = None
  async_client: Optional[Any] = None

  def __post_init__(self) -> None:
    super().__post_init__()
    if self.thinking:
      self._validate_thinking_support()
    # Advertised capability metadata (read by observability/contract tests) —
    # NOT a gate: output_format is sent whenever a schema is requested.
    s = self.spec()
    self.supports_native_structured_outputs = bool(s and s.supports("structured_output"))
    if self.id not in self.NON_THINKING_MODELS:
      self.supports_native_thinking = True
    if self.skills:
      self._setup_skills_configuration()

  # --- auth + endpoint ------------------------------------------------------

  def _resolve_auth(self) -> None:
    self._get_client_params()

  def _get_client_params(self) -> Dict[str, Any]:
    """Resolve + validate credentials. Bearer auth-token takes precedence over the API key."""
    self.auth_token = self.auth_token or getenv("ANTHROPIC_AUTH_TOKEN")
    if not self.auth_token:
      self.api_key = self.api_key or getenv("ANTHROPIC_API_KEY")
      if not self.api_key:
        raise ModelAuthenticationError(
          message="ANTHROPIC_API_KEY not set. Please set the ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) environment variable.",
          model_name=self.name,
        )
    return {}

  def _chat_endpoint(self, count_tokens: bool = False) -> str:
    base = str(self.base_url or getenv("ANTHROPIC_BASE_URL") or DEFAULT_ANTHROPIC_BASE_URL).rstrip("/")
    return f"{base}/v1/messages/count_tokens" if count_tokens else f"{base}/v1/messages"

  def _build_headers(self, betas: Optional[List[str]] = None) -> Dict[str, str]:
    """`x-api-key` and bearer `Authorization` are mutually exclusive; bearer needs the oauth beta."""
    headers: Dict[str, str] = {"content-type": "application/json", "anthropic-version": ANTHROPIC_VERSION}
    beta_flags = list(betas or [])
    if self.auth_token:
      headers["authorization"] = f"Bearer {self.auth_token}"
      if "oauth-2025-04-20" not in beta_flags:
        beta_flags.append("oauth-2025-04-20")
    else:
      headers["x-api-key"] = self.api_key or ""
    if beta_flags:
      headers["anthropic-beta"] = ",".join(beta_flags)
    if self.default_headers:
      headers.update(self.default_headers)
    return headers

  def _validate_thinking_support(self) -> None:
    if self.thinking and self.id in self.NON_THINKING_MODELS:
      models = "\n  - ".join(sorted(self.NON_THINKING_MODELS))
      raise ValueError(f"Model '{self.id}' does not support extended thinking.\n\nModels without thinking:\n  - {models}")

  def _setup_skills_configuration(self) -> None:
    required = ["code-execution-2025-08-25", "skills-2025-10-02"]
    if self.betas is None:
      self.betas = required
    else:
      for beta in required:
        if beta not in self.betas:
          self.betas.append(beta)

  def _ensure_additional_properties_false(self, schema: Dict[str, Any]) -> None:
    if isinstance(schema, dict):
      if schema.get("type") == "object":
        schema["additionalProperties"] = False
      for key, value in schema.items():
        if key in ["properties", "items", "allOf", "anyOf", "oneOf"]:
          if isinstance(value, dict):
            self._ensure_additional_properties_false(value)
          elif isinstance(value, list):
            for item in value:
              if isinstance(item, dict):
                self._ensure_additional_properties_false(item)

  def _build_output_format(self, response_format: Optional[Union[Dict, Type[BaseModel]]]) -> Optional[Dict[str, Any]]:
    # No capability gate: send output_format whenever a schema is requested. A
    # model that supports structured outputs honours it; one that doesn't returns
    # an API error — we don't second-guess it from a spec sheet.
    if response_format is None:
      return None
    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
      schema = response_format.model_json_schema()
      if isinstance(schema, dict):
        if "additionalProperties" not in schema:
          schema["additionalProperties"] = False
        self._ensure_additional_properties_false(schema)
      return {"type": "json_schema", "schema": schema}
    if response_format.get("type") == "json_object":
      return None
    return response_format

  # --- request body (called by the transform) -------------------------------

  def get_request_params(
    self,
    response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
  ) -> Dict[str, Any]:
    if self.thinking:
      self._validate_thinking_support()

    params: Dict[str, Any] = {}
    if self.max_tokens:
      params["max_tokens"] = self.max_tokens
    if self.thinking:
      params["thinking"] = self.thinking
    # `is not None` (not truthiness) so 0.0 isn't dropped → Anthropic default 1.0.
    if self.temperature is not None:
      params["temperature"] = self.temperature
    if self.stop_sequences:
      params["stop_sequences"] = self.stop_sequences
    if self.top_p is not None:
      params["top_p"] = self.top_p
    if self.top_k is not None:
      params["top_k"] = self.top_k

    betas = list(self.betas) if self.betas else []
    uses_structured = response_format is not None or any(
      t.get("type") == "function" and (t.get("function") or {}).get("strict") is True for t in tools or []
    )
    if uses_structured and "structured-outputs-2025-11-13" not in betas:
      betas.append("structured-outputs-2025-11-13")
    if betas:
      params["betas"] = betas  # routed to the anthropic-beta header by the transform

    if self.context_management:
      params["context_management"] = self.context_management
    if self.mcp_servers:
      params["mcp_servers"] = [{k: v for k, v in asdict(s).items() if v is not None} for s in self.mcp_servers]
    if self.skills:
      params["container"] = {"skills": self.skills}
    if self.request_params:
      params.update(self.request_params)
    return params

  def _prepare_request_kwargs(
    self,
    system_message: str,
    tools: Optional[List[Dict[str, Any]]] = None,
    response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    system_blocks: Optional[List[Dict[str, Any]]] = None,
  ) -> Dict[str, Any]:
    body = self.get_request_params(response_format=response_format, tools=tools).copy()

    if self.system_prompt_blocks:
      body["system"] = self._build_system_prompt_blocks(system_message)
    elif system_message:
      block: Dict[str, Any] = {"text": system_message, "type": "text"}
      if self.cache_system_prompt:
        block["cache_control"] = self._cache_control()
      body["system"] = [block]
    if system_blocks:
      body["system"] = system_blocks

    if self.skills:
      code_execution_tool = {"type": "code_execution_20250825", "name": "code_execution"}
      tools = (tools + [code_execution_tool]) if tools else [code_execution_tool]

    if tools:
      formatted_tools = format_tools_for_model(tools)
      # Cache the tool schemas (static across turns) on the last tool — Anthropic
      # caches the whole tools+system prefix up to the last breakpoint.
      if formatted_tools and self.cache_system_prompt:
        formatted_tools[-1] = {**formatted_tools[-1], "cache_control": self._cache_control()}
      body["tools"] = formatted_tools

    output_format = self._build_output_format(response_format)
    if output_format:
      body["output_format"] = output_format

    if body:
      log_debug(f"Calling {self.provider} with request parameters: {body}", log_level=2)
    return body

  def _cache_control(self) -> Dict[str, Any]:
    """Ephemeral cache breakpoint — 1h TTL when extended_cache_time is set, else Anthropic's 5m default."""
    return {"type": "ephemeral", "ttl": "1h"} if self.extended_cache_time else {"type": "ephemeral"}

  def _build_system_prompt_blocks(self, system_message: Optional[str]) -> List[Dict[str, Any]]:
    blocks = list(self.system_prompt_blocks or [])
    self._validate_cache_ttl_order(blocks)
    out: List[Dict[str, Any]] = []
    for block in blocks:
      entry: Dict[str, Any] = {"type": "text", "text": block.text}
      if block.cache:
        cache_control: Dict[str, Any] = {"type": "ephemeral"}
        ttl = block.ttl
        if ttl is None and self.extended_cache_time:
          ttl = "1h"
        if ttl is not None:
          cache_control["ttl"] = ttl
        entry["cache_control"] = cache_control
      out.append(entry)
    if system_message:
      out.append({"type": "text", "text": system_message})
    return out

  @staticmethod
  def _validate_cache_ttl_order(blocks: List[SystemPromptBlock]) -> None:
    seen_one_hour = False
    for idx, block in enumerate(blocks):
      if not block.cache:
        continue
      if block.ttl == "1h":
        seen_one_hour = True
      elif (block.ttl == "5m" or block.ttl is None) and seen_one_hour:
        raise ValueError(
          f"system_prompt_blocks[{idx}] uses ttl='5m' (or default) after a 1h block. "
          "Anthropic requires longer-TTL cache blocks to come after shorter-TTL ones — reorder so 5m blocks precede 1h blocks."
        )

  @staticmethod
  def _extract_cache_blocks(messages: List[Message]) -> Optional[List[Dict[str, Any]]]:
    for msg in messages:
      if msg.role == "system" and hasattr(msg, "_cache_blocks"):
        blocks = getattr(msg, "_cache_blocks", None)
        if blocks:
          return blocks  # type: ignore[return-value]
    return None

  # --- token counting -------------------------------------------------------

  def _count_tokens_body(
    self,
    messages: List[Message],
    tools: Optional[Sequence[Union[Function, Dict[str, Any]]]],
    output_schema: Optional[Union[Dict, Type[BaseModel]]],
  ) -> Dict[str, Any]:
    anthropic_messages, system_prompt = format_messages(messages, compress_tool_results=True, cite_documents=output_schema is None)
    body: Dict[str, Any] = {"model": self.id, "messages": anthropic_messages}
    if system_prompt:
      body["system"] = system_prompt
    if tools:
      anthropic_tools = format_tools_for_model(self._format_tools(list(tools)))
      if anthropic_tools:
        body["tools"] = anthropic_tools
    return body

  def count_tokens(
    self,
    messages: List[Message],
    tools: Optional[Sequence[Any]] = None,
    output_schema: Optional[Union[Dict, Type[BaseModel]]] = None,
  ) -> int:
    self._get_client_params()
    body = self._count_tokens_body(messages, tools, output_schema)
    raw = post_json(self._chat_endpoint(count_tokens=True), self._build_headers(self.betas), body, self.timeout, self.name, self.id)
    return int(raw.get("input_tokens", 0)) + count_schema_tokens(output_schema, self.id)

  async def acount_tokens(
    self,
    messages: List[Message],
    tools: Optional[Sequence[Any]] = None,
    output_schema: Optional[Union[Dict, Type[BaseModel]]] = None,
  ) -> int:
    self._get_client_params()
    body = self._count_tokens_body(messages, tools, output_schema)
    raw = await apost_json(self._chat_endpoint(count_tokens=True), self._build_headers(self.betas), body, self.timeout, self.name, self.id)
    return int(raw.get("input_tokens", 0)) + count_schema_tokens(output_schema, self.id)

  def get_system_message_for_model(self, tools: Optional[List[Any]] = None) -> Optional[str]:
    if tools is not None and len(tools) > 0:
      return "Do not reflect on the quality of the returned search results in your response\n\n"
    return None

  def to_dict(self) -> Dict[str, Any]:
    model_dict = super().to_dict()
    model_dict.update({
      "max_tokens": self.max_tokens,
      "thinking": self.thinking,
      "temperature": self.temperature,
      "stop_sequences": self.stop_sequences,
      "top_p": self.top_p,
      "top_k": self.top_k,
      "cache_system_prompt": self.cache_system_prompt,
      "extended_cache_time": self.extended_cache_time,
      "betas": self.betas,
    })
    return {k: v for k, v in model_dict.items() if v is not None}
