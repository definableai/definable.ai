"""Anthropic (Claude) dialect — the Messages API transform.

`request` builds the `/v1/messages` payload (system top-level, max_tokens
required, betas → header); `response` parses the content-block array; `decoder`
runs Anthropic's named-event SSE (message_start → content_block_* →
message_delta → message_stop, no `[DONE]`), accumulating tool-arg `partial_json`
per block and merging split usage into a single trailing delta.
"""

import json
from typing import Any, Dict, Iterator, List, Optional, Type, Union

from pydantic import BaseModel, ValidationError

from definable.exceptions import classify_model_error
from definable.model.base import Request
from definable.model.http.sse import SSEEvent
from definable.model.message import Citations, DocumentCitation, Message, UrlCitation
from definable.model.metrics import Metrics
from definable.model.response import ModelResponse
from definable.utils.claude import format_messages
from definable.utils.log import log_debug, log_warning


def _metrics(usage: Dict[str, Any]) -> Metrics:
  m = Metrics()
  m.input_tokens = usage.get("input_tokens") or 0
  m.output_tokens = usage.get("output_tokens") or 0
  m.total_tokens = m.input_tokens + m.output_tokens  # Anthropic has no total_tokens
  m.cache_read_tokens = usage.get("cache_read_input_tokens") or 0
  m.cache_write_tokens = usage.get("cache_creation_input_tokens") or 0
  if server_tool_use := usage.get("server_tool_use"):
    m.provider_metrics = {"server_tool_use": server_tool_use}
  if service_tier := usage.get("service_tier"):
    m.provider_metrics = m.provider_metrics or {}
    m.provider_metrics["service_tier"] = service_tier
  return m


def _append_citations(model_response: ModelResponse, citations: List[Dict[str, Any]]) -> None:
  if model_response.citations is None:
    model_response.citations = Citations(raw=[], urls=[], documents=[])
  for citation in citations:
    model_response.citations.raw.append(citation)  # type: ignore[union-attr]
    ctype = citation.get("type")
    if ctype == "web_search_result_location":
      model_response.citations.urls.append(UrlCitation(url=citation.get("url"), title=citation.get("cited_text")))  # type: ignore[union-attr]
    elif ctype == "page_location":
      model_response.citations.documents.append(  # type: ignore[union-attr]
        DocumentCitation(document_title=citation.get("document_title"), cited_text=citation.get("cited_text"))
      )


def _parse_structured_output(model_response: ModelResponse, response_format: Any, text: str) -> None:
  if not (response_format is not None and isinstance(response_format, type) and issubclass(response_format, BaseModel)) or not text:
    return
  try:
    model_response.parsed = response_format.model_validate(json.loads(text))
    log_debug(f"Successfully parsed structured output: {model_response.parsed}")
  except json.JSONDecodeError as e:
    log_warning(f"Failed to parse JSON from structured output: {e}")
  except ValidationError as e:
    log_warning(f"Failed to validate structured output against schema: {e}")
  except Exception as e:
    log_warning(f"Unexpected error parsing structured output: {e}")


def _parse_response(model: Any, response: Dict[str, Any], response_format: Optional[Union[Dict, Type[BaseModel]]]) -> ModelResponse:
  mr = ModelResponse()
  mr.role = response.get("role") or "assistant"

  file_ids: List[str] = []
  for block in response.get("content") or []:
    btype = block.get("type")
    if btype == "text":
      text_content = block.get("text") or ""
      mr.content = text_content if mr.content is None else mr.content + text_content
      if block.get("citations"):
        _append_citations(mr, block["citations"])
    elif btype == "thinking":
      mr.reasoning_content = block.get("thinking")
      mr.provider_data = {"signature": block.get("signature")}
    elif btype == "redacted_thinking":
      mr.redacted_reasoning_content = block.get("data")
    elif btype == "tool_use":
      function_def: Dict[str, Any] = {"name": block.get("name")}
      if block.get("input"):
        function_def["arguments"] = json.dumps(block["input"])
      mr.extra = mr.extra or {}
      mr.tool_calls.append({"id": block.get("id"), "type": "function", "function": function_def})
    elif btype == "bash_code_execution_tool_result" and model.skills:
      inner = block.get("content")
      inner_content = inner.get("content") if isinstance(inner, dict) else None
      if isinstance(inner_content, list):
        for output_block in inner_content:
          if isinstance(output_block, dict) and output_block.get("file_id"):
            file_ids.append(output_block["file_id"])

  _parse_structured_output(mr, response_format, mr.content or "")

  if response.get("usage") is not None:
    mr.response_usage = _metrics(response["usage"])

  context_mgmt = response.get("context_management")
  if model.context_management is not None and context_mgmt is not None:
    mr.provider_data = mr.provider_data or {}
    mr.provider_data["context_management"] = context_mgmt
  if file_ids:
    mr.provider_data = mr.provider_data or {}
    mr.provider_data["file_ids"] = file_ids

  return mr


class ClaudeStreamDecoder:
  """Anthropic named-event SSE → unified deltas.

  Tool args arrive as `input_json_delta.partial_json` fragments accumulated per
  block index and emitted whole at `content_block_stop`. Usage is split (input @
  message_start, cumulative output @ message_delta) so it is merged and emitted
  exactly once in `finish()` — the agent layer sums usage across deltas.
  """

  def __init__(self, model: Any, response_format: Optional[Union[Dict, Type[BaseModel]]]):
    self._model = model
    self._response_format = response_format
    self._blocks: Dict[int, Dict[str, Any]] = {}
    self._usage: Dict[str, Any] = {}
    self._text = ""
    self._citations: List[Dict[str, Any]] = []

  def feed(self, event: SSEEvent) -> Iterator[ModelResponse]:
    try:
      payload = json.loads(event.data)
    except json.JSONDecodeError:
      log_warning(f"Skipping malformed SSE chunk from {self._model.provider}")
      return
    etype = payload.get("type") or event.event

    if etype == "message_start":
      self._usage.update((payload.get("message") or {}).get("usage") or {})
    elif etype == "content_block_start":
      yield from self._on_block_start(payload)
    elif etype == "content_block_delta":
      yield from self._on_block_delta(payload)
    elif etype == "content_block_stop":
      yield from self._on_block_stop(payload)
    elif etype == "message_delta":
      self._usage.update(payload.get("usage") or {})
    elif etype == "error":
      err = payload.get("error") or {}
      raise classify_model_error(
        err.get("message") or "Anthropic stream error", status_code=502, model_name=self._model.name, model_id=self._model.id
      )

  def _on_block_start(self, payload: Dict[str, Any]) -> Iterator[ModelResponse]:
    index = payload.get("index", 0)
    block = payload.get("content_block") or {}
    self._blocks[index] = {"type": block.get("type"), "id": block.get("id"), "name": block.get("name"), "args": ""}
    if block.get("type") == "redacted_thinking" and block.get("data"):
      mr = ModelResponse()
      mr.redacted_reasoning_content = block["data"]
      yield mr

  def _on_block_delta(self, payload: Dict[str, Any]) -> Iterator[ModelResponse]:
    index = payload.get("index", 0)
    delta = payload.get("delta") or {}
    dtype = delta.get("type")
    if dtype == "text_delta":
      text = delta.get("text") or ""
      self._text += text
      mr = ModelResponse()
      mr.content = text
      yield mr
    elif dtype == "thinking_delta":
      mr = ModelResponse()
      mr.reasoning_content = delta.get("thinking")
      yield mr
    elif dtype == "signature_delta":
      mr = ModelResponse()
      mr.provider_data = {"signature": delta.get("signature")}
      yield mr
    elif dtype == "input_json_delta":
      blk = self._blocks.get(index)
      if blk is not None:
        blk["args"] += delta.get("partial_json") or ""
    elif dtype == "citations_delta":
      if delta.get("citation"):
        self._citations.append(delta["citation"])

  def _on_block_stop(self, payload: Dict[str, Any]) -> Iterator[ModelResponse]:
    blk = self._blocks.get(payload.get("index", 0))
    if blk and blk.get("type") == "tool_use":
      function_def: Dict[str, Any] = {"name": blk.get("name")}
      args = blk.get("args") or ""
      if args.strip():
        function_def["arguments"] = args
      mr = ModelResponse()
      mr.extra = {}
      mr.tool_calls = [{"id": blk.get("id"), "type": "function", "function": function_def}]
      yield mr

  def finish(self) -> Iterator[ModelResponse]:
    mr = ModelResponse()
    emitted = False
    if self._citations:
      _append_citations(mr, self._citations)
      emitted = True
    if self._response_format is not None and self._text:
      _parse_structured_output(mr, self._response_format, self._text)
      if mr.parsed is not None:
        emitted = True
    if self._usage:
      mr.response_usage = _metrics(self._usage)
      emitted = True
    if emitted:
      yield mr


def _cache_last_content_block(messages: List[Dict[str, Any]], cache_control: Dict[str, Any]) -> None:
  """Put a rolling cache breakpoint on the final content block — caches the whole
  conversation prefix so the next turn re-reads history instead of re-billing it.
  Safe to mutate in place: format_messages builds fresh blocks every call."""
  if not messages:
    return
  content = messages[-1].get("content")
  if isinstance(content, list) and content and isinstance(content[-1], dict):
    content[-1]["cache_control"] = cache_control


class AnthropicTransform:
  def request(
    self,
    model: Any,
    messages: List[Message],
    tools: Optional[List[Dict[str, Any]]],
    response_format: Optional[Union[Dict, Type[BaseModel]]],
    tool_choice: Optional[Union[str, Dict[str, Any]]],
    stream: bool,
    compress_tool_results: bool,
  ) -> Request:
    chat_messages, system_message = format_messages(messages, compress_tool_results=compress_tool_results, cite_documents=response_format is None)
    cache_blocks = model._extract_cache_blocks(messages)
    body = model._prepare_request_kwargs(system_message, tools=tools, response_format=response_format, system_blocks=cache_blocks)
    betas = body.pop("betas", None)
    body["model"] = model.id
    body["messages"] = chat_messages
    if getattr(model, "cache_system_prompt", False):
      _cache_last_content_block(chat_messages, model._cache_control())
    if stream:
      body["stream"] = True
    return Request(model._chat_endpoint(), model._build_headers(betas), body)

  def response(self, model: Any, raw: Dict[str, Any], response_format: Optional[Union[Dict, Type[BaseModel]]]) -> ModelResponse:
    return _parse_response(model, raw, response_format)

  def decoder(self, model: Any, response_format: Optional[Union[Dict, Type[BaseModel]]]) -> ClaudeStreamDecoder:
    return ClaudeStreamDecoder(model, response_format)


ANTHROPIC_TRANSFORM = AnthropicTransform()
