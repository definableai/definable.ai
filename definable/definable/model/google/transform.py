"""Gemini dialect — the generateContent transform (raw HTTP, no SDK).

Gemini's wire shape is the odd one: model id in the URL path, `contents`/`parts`
(not `messages`), assistant role is `"model"`, system prompt is a top-level
`systemInstruction`, every sampling param nests under `generationConfig`
(camelCase), function args/results are JSON objects (no call id), SSE has no
`[DONE]` and reports cumulative usage per chunk (take the last).
"""

import base64
import json
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union

from pydantic import BaseModel, ValidationError

from definable.exceptions import classify_model_error
from definable.media import File, Image
from definable.model.base import Request
from definable.model.http.sse import SSEEvent
from definable.model.message import Message
from definable.model.metrics import Metrics
from definable.model.response import ModelResponse
from definable.utils.log import log_debug, log_warning

# JSON-Schema keywords Gemini's OpenAPI-subset rejects — strip them recursively.
_UNSUPPORTED_SCHEMA_KEYS = ("additionalProperties", "$schema", "title", "default", "$defs", "definitions")


def _sanitize_schema(schema: Any) -> Any:
  if isinstance(schema, dict):
    return {k: _sanitize_schema(v) for k, v in schema.items() if k not in _UNSUPPORTED_SCHEMA_KEYS}
  if isinstance(schema, list):
    return [_sanitize_schema(v) for v in schema]
  return schema


def _media_bytes(media: Union[Image, File]) -> Optional[bytes]:
  """Read raw bytes from content / filepath / url (works for Image and File)."""
  content = getattr(media, "content", None)
  if isinstance(content, bytes):
    return content
  filepath = getattr(media, "filepath", None)
  if filepath:
    with open(filepath, "rb") as fh:
      return fh.read()
  url = getattr(media, "url", None)
  if url:
    import httpx

    return httpx.get(url).content
  return None


def _inline_media(media: Union[Image, File]) -> Optional[Dict[str, Any]]:
  try:
    raw = _media_bytes(media)
  except Exception as e:
    log_warning(f"Gemini: could not read media bytes: {e}")
    return None
  if not raw:
    return None
  mime = media.mime_type or ("image/png" if isinstance(media, Image) else "application/pdf")
  return {"inlineData": {"mimeType": mime, "data": base64.b64encode(raw).decode("utf-8")}}


def _to_contents(messages: List[Message], compress: bool) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
  contents: List[Dict[str, Any]] = []
  system_parts: List[Dict[str, Any]] = []

  for m in messages:
    if m.role in ("system", "developer"):
      if m.content:
        system_parts.append({"text": m.content if isinstance(m.content, str) else str(m.content)})
      continue

    if m.role == "tool":
      # functionResponse goes in a `user` content, matched by function name (no id in Gemini).
      name = m.tool_name or m.name or "tool"
      result = m.get_content(use_compressed_content=compress)
      response = result if isinstance(result, dict) else {"result": str(result)}
      contents.append({"role": "user", "parts": [{"functionResponse": {"name": name, "response": response}}]})
      continue

    role = "model" if m.role in ("assistant", "model") else "user"
    parts: List[Dict[str, Any]] = []
    if isinstance(m.content, str) and m.content.strip():
      parts.append({"text": m.content})
    if m.images:
      for img in m.images:
        if part := _inline_media(img):
          parts.append(part)
    if m.files:
      for f in m.files:
        if part := _inline_media(f):
          parts.append(part)
    if m.tool_calls:
      for tc in m.tool_calls:
        fn = tc.get("function", {})
        args = json.loads(fn["arguments"]) if fn.get("arguments", "").strip() else {}
        parts.append({"functionCall": {"name": fn.get("name"), "args": args}})
    if parts:
      contents.append({"role": role, "parts": parts})

  system_instruction = {"parts": system_parts} if system_parts else None
  return contents, system_instruction


def _to_declaration(tool: Dict[str, Any]) -> Dict[str, Any]:
  fn = tool.get("function", tool)
  decl: Dict[str, Any] = {"name": fn.get("name"), "description": fn.get("description") or ""}
  params = fn.get("parameters")
  if params:
    decl["parameters"] = _sanitize_schema(params)
  return decl


def _tool_mode(tool_choice: Optional[Union[str, Dict[str, Any]]]) -> Optional[str]:
  if tool_choice in ("required", "any"):
    return "ANY"
  if tool_choice == "none":
    return "NONE"
  return None  # AUTO is Gemini's default


def _metrics(usage: Dict[str, Any]) -> Metrics:
  m = Metrics()
  m.input_tokens = usage.get("promptTokenCount") or 0
  m.output_tokens = usage.get("candidatesTokenCount") or 0
  m.total_tokens = usage.get("totalTokenCount") or (m.input_tokens + m.output_tokens)
  m.reasoning_tokens = usage.get("thoughtsTokenCount") or 0
  m.cache_read_tokens = usage.get("cachedContentTokenCount") or 0
  return m


def _parse_structured_output(mr: ModelResponse, response_format: Any, text: str) -> None:
  if not (response_format is not None and isinstance(response_format, type) and issubclass(response_format, BaseModel)) or not text:
    return
  try:
    mr.parsed = response_format.model_validate(json.loads(text))
  except (json.JSONDecodeError, ValidationError) as e:
    log_warning(f"Gemini: failed to parse structured output: {e}")


def _parse_response(model: Any, raw: Dict[str, Any], response_format: Optional[Union[Dict, Type[BaseModel]]]) -> ModelResponse:
  if raw.get("error"):
    err = raw["error"]
    raise classify_model_error(
      err.get("message") or "Gemini error", status_code=int(err.get("code") or 502), model_name=model.name, model_id=model.id
    )

  mr = ModelResponse()
  candidates = raw.get("candidates") or []
  if not candidates:
    block = (raw.get("promptFeedback") or {}).get("blockReason")
    if block:
      raise classify_model_error(f"Gemini blocked the prompt: {block}", status_code=400, model_name=model.name, model_id=model.id)
  else:
    mr.role = "assistant"
    text_parts: List[str] = []
    reasoning_parts: List[str] = []
    for i, part in enumerate((candidates[0].get("content") or {}).get("parts") or []):
      if "functionCall" in part:
        fc = part["functionCall"]
        mr.tool_calls.append({
          "id": f"{fc.get('name')}-{i}",
          "type": "function",
          "function": {"name": fc.get("name"), "arguments": json.dumps(fc.get("args") or {})},
        })
      elif "text" in part:
        (reasoning_parts if part.get("thought") else text_parts).append(part["text"])
    if text_parts:
      mr.content = "".join(text_parts)
    if reasoning_parts:
      mr.reasoning_content = "".join(reasoning_parts)

  if raw.get("usageMetadata"):
    mr.response_usage = _metrics(raw["usageMetadata"])

  _parse_structured_output(mr, response_format, mr.content or "")
  return mr


class GeminiStreamDecoder:
  """Each SSE `data:` is a full GenerateContentResponse with a partial candidate.
  Text concatenates across chunks; functionCall parts arrive whole; usageMetadata
  is cumulative so we keep the last and emit it once at the end."""

  def __init__(self, model: Any, response_format: Optional[Union[Dict, Type[BaseModel]]]):
    self._model = model
    self._response_format = response_format
    self._usage: Dict[str, Any] = {}
    self._text = ""

  def feed(self, event: SSEEvent) -> Iterator[ModelResponse]:
    try:
      chunk = json.loads(event.data)
    except json.JSONDecodeError:
      log_warning(f"Skipping malformed SSE chunk from {self._model.provider}")
      return
    if chunk.get("error"):
      err = chunk["error"]
      raise classify_model_error(
        err.get("message") or "Gemini stream error", status_code=int(err.get("code") or 502), model_name=self._model.name, model_id=self._model.id
      )
    if chunk.get("usageMetadata"):
      self._usage = chunk["usageMetadata"]  # cumulative — keep latest
    for cand in chunk.get("candidates") or []:
      for i, part in enumerate((cand.get("content") or {}).get("parts") or []):
        if "functionCall" in part:
          fc = part["functionCall"]
          mr = ModelResponse()
          mr.tool_calls = [
            {"id": f"{fc.get('name')}-{i}", "type": "function", "function": {"name": fc.get("name"), "arguments": json.dumps(fc.get("args") or {})}}
          ]
          yield mr
        elif "text" in part:
          if part.get("thought"):
            mr = ModelResponse()
            mr.reasoning_content = part["text"]
            yield mr
          else:
            self._text += part["text"]
            mr = ModelResponse()
            mr.content = part["text"]
            yield mr

  def finish(self) -> Iterator[ModelResponse]:
    mr = ModelResponse()
    emitted = False
    if self._response_format is not None and self._text:
      _parse_structured_output(mr, self._response_format, self._text)
      if mr.parsed is not None:
        emitted = True
    if self._usage:
      mr.response_usage = _metrics(self._usage)
      emitted = True
    if emitted:
      yield mr


class GeminiTransform:
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
    contents, system_instruction = _to_contents(messages, compress_tool_results)
    body: Dict[str, Any] = {"contents": contents}
    if system_instruction:
      body["systemInstruction"] = system_instruction
    gen = model.get_request_params(response_format)
    if gen:
      body["generationConfig"] = gen
    if tools:
      body["tools"] = [{"functionDeclarations": [_to_declaration(t) for t in tools]}]
      mode = _tool_mode(tool_choice)
      if mode:
        body["toolConfig"] = {"functionCallingConfig": {"mode": mode}}
    if model.safety_settings:
      body["safetySettings"] = model.safety_settings
    log_debug(f"Calling {model.provider} :{('streamGenerateContent' if stream else 'generateContent')}", log_level=2)
    return Request(model._endpoint(stream=stream), model._build_headers(), body)

  def response(self, model: Any, raw: Dict[str, Any], response_format: Optional[Union[Dict, Type[BaseModel]]]) -> ModelResponse:
    return _parse_response(model, raw, response_format)

  def decoder(self, model: Any, response_format: Optional[Union[Dict, Type[BaseModel]]]) -> GeminiStreamDecoder:
    return GeminiStreamDecoder(model, response_format)


GEMINI_TRANSFORM = GeminiTransform()
