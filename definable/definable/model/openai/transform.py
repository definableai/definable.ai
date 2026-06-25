"""OpenAI dialect — the transform that rules every OpenAI-schema provider.

`request` builds the Chat Completions payload from unified messages; `response`
parses the wire JSON; `decoder` turns the SSE stream into deltas. The OpenAI
family (xAI/DeepSeek/Moonshot/OpenRouter/Perplexity/Ollama) all share this — they
only tweak config fields and a couple of small hooks on the model
(`get_request_params`, `_format_message`, `_augment_response`).
"""

import json
from typing import Any, Dict, Iterator, Optional, Type, Union
from uuid import uuid4

from pydantic import BaseModel

from definable.exceptions import ModelProviderError
from definable.media import Audio
from definable.model.base import Request
from definable.model.http.sse import SSEEvent
from definable.model.message import Message
from definable.model.metrics import Metrics
from definable.model.response import ModelResponse
from definable.utils.log import log_warning
from definable.utils.openai import _format_file_for_message, audio_to_message, images_to_message
from definable.utils.reasoning import extract_thinking_content

_DEFAULT_ROLE_MAP = {"system": "developer", "user": "user", "assistant": "assistant", "tool": "tool", "model": "assistant"}


def _format_message(model: Any, message: Message, compress_tool_results: bool) -> Dict[str, Any]:
  """Unified Message → OpenAI Chat Completions message dict. Owns the dialect's
  message conversion (role mapping, multimodal parts, tool calls)."""
  role_map = getattr(model, "role_map", None) or getattr(model, "default_role_map", None) or _DEFAULT_ROLE_MAP
  message_dict: Dict[str, Any] = {
    "role": role_map[message.role],
    "content": message.get_content(use_compressed_content=compress_tool_results),
    "name": message.name,
    "tool_call_id": message.tool_call_id,
    "tool_calls": message.tool_calls,
  }
  # Some OpenAI-compat providers (DeepSeek, Moonshot) echo assistant reasoning back.
  if getattr(model, "send_reasoning_content", False):
    message_dict["reasoning_content"] = message.reasoning_content
  if message.provider_data and message.provider_data.get("reasoning_details"):
    message_dict["reasoning_details"] = message.provider_data["reasoning_details"]
  message_dict = {k: v for k, v in message_dict.items() if v is not None}

  if (message.images and len(message.images) > 0) or (message.audio and len(message.audio) > 0):
    if isinstance(message.content, str):
      message_dict["content"] = [{"type": "text", "text": message.content}]
      if message.images:
        message_dict["content"].extend(images_to_message(images=message.images))
      if message.audio:
        message_dict["content"].extend(audio_to_message(audio=message.audio))

  if message.audio_output is not None:
    message_dict["content"] = ""
    message_dict["audio"] = {"id": message.audio_output.id}

  if message.videos and len(message.videos) > 0:
    log_warning("Video input is currently unsupported.")

  if message.tool_calls is not None and len(message.tool_calls) == 0:
    message_dict["tool_calls"] = None

  if message.files is not None:
    content = message_dict.get("content")
    if isinstance(content, str):
      message_dict["content"] = [{"type": "text", "text": content}]
    elif content is None:
      message_dict["content"] = []
    for file in message.files:
      if file_part := _format_file_for_message(file):
        message_dict["content"].insert(0, file_part)

  # content="" if None, except assistant turns that carry tool_calls (content is legitimately absent).
  if message.content is None and not (message_dict.get("role") == "assistant" and message_dict.get("tool_calls")):
    message_dict["content"] = ""
  return message_dict


def _metrics(usage: Dict[str, Any]) -> Metrics:
  m = Metrics()
  m.input_tokens = usage.get("prompt_tokens") or 0
  m.output_tokens = usage.get("completion_tokens") or 0
  m.total_tokens = usage.get("total_tokens") or 0
  if prompt_details := usage.get("prompt_tokens_details"):
    m.audio_input_tokens = prompt_details.get("audio_tokens") or 0
    m.cache_read_tokens = prompt_details.get("cached_tokens") or 0
  if completion_details := usage.get("completion_tokens_details"):
    m.audio_output_tokens = completion_details.get("audio_tokens") or 0
    m.reasoning_tokens = completion_details.get("reasoning_tokens") or 0
  m.cost = usage.get("cost")
  return m


def _parse_response(model: Any, response: Dict[str, Any], response_format: Optional[Union[Dict, Type[BaseModel]]]) -> ModelResponse:
  mr = ModelResponse()

  if response.get("error"):
    error = response["error"]
    message = error.get("message", "Unknown model error") if isinstance(error, dict) else str(error)
    raise ModelProviderError(message=message or "Unknown model error", model_name=model.name, model_id=model.id)

  choices = response.get("choices") or []
  if choices:
    msg: Dict[str, Any] = choices[0].get("message") or {}
    if msg.get("role") is not None:
      mr.role = msg["role"]
    if msg.get("content") is not None:
      mr.content = msg["content"]
      if mr.content:
        reasoning, output = extract_thinking_content(mr.content)
        if reasoning:
          mr.reasoning_content = reasoning
          mr.content = output
    if msg.get("tool_calls"):
      mr.tool_calls = msg["tool_calls"]
    response_audio = msg.get("audio")
    if response_audio:
      if response_audio.get("transcript") and not mr.content:
        mr.content = response_audio["transcript"]
      try:
        mr.audio = Audio(
          id=response_audio.get("id"),
          content=response_audio.get("data"),
          expires_at=response_audio.get("expires_at"),
          transcript=response_audio.get("transcript"),
        )
      except Exception as e:
        log_warning(f"Error processing audio: {e}")
    if msg.get("reasoning_content") is not None:
      mr.reasoning_content = msg["reasoning_content"]
    elif msg.get("reasoning") is not None:
      mr.reasoning_content = msg["reasoning"]

  if response.get("usage") is not None:
    mr.response_usage = _metrics(response["usage"])
    model._calculate_cost_if_needed(mr.response_usage)

  if mr.provider_data is None:
    mr.provider_data = {}
  if response.get("id"):
    mr.provider_data["id"] = response["id"]
  if response.get("system_fingerprint"):
    mr.provider_data["system_fingerprint"] = response["system_fingerprint"]

  if response_format is not None and isinstance(response_format, type) and issubclass(response_format, BaseModel) and mr.content:
    try:
      mr.parsed = response_format.model_validate(json.loads(mr.content))
    except json.JSONDecodeError as e:
      log_warning(f"Failed to parse JSON from structured output: {e}")
    except Exception as e:
      log_warning(f"Failed to validate structured output against schema: {e}")

  model._augment_response(response, mr)
  return mr


def _parse_delta(model: Any, response_delta: Dict[str, Any]) -> ModelResponse:
  mr = ModelResponse()
  choices = response_delta.get("choices") or []
  if choices:
    delta: Dict[str, Any] = choices[0].get("delta") or {}
    if delta:
      if delta.get("content") is not None:
        mr.content = delta["content"]
        if mr.provider_data is None:
          mr.provider_data = {}
        if response_delta.get("id"):
          mr.provider_data["id"] = response_delta["id"]
        if response_delta.get("system_fingerprint"):
          mr.provider_data["system_fingerprint"] = response_delta["system_fingerprint"]
      if delta.get("tool_calls") is not None:
        mr.tool_calls = delta["tool_calls"]
      if delta.get("reasoning_content") is not None:
        mr.reasoning_content = delta["reasoning_content"]
      elif delta.get("reasoning") is not None:
        mr.reasoning_content = delta["reasoning"]
      audio = delta.get("audio")
      if audio:
        try:
          audio_data = audio.get("data")
          if audio_data is not None:
            mr.audio = Audio(
              id=audio.get("id"),
              content=audio_data,
              expires_at=audio.get("expires_at"),
              transcript=audio.get("transcript"),
              sample_rate=24000,
              mime_type="pcm16",
            )
          elif audio.get("transcript") is not None or audio.get("id") is not None:
            mr.audio = Audio(
              id=audio.get("id") or str(uuid4()),
              content=b"",
              expires_at=audio.get("expires_at"),
              transcript=audio.get("transcript"),
              sample_rate=24000,
              mime_type="pcm16",
            )
        except Exception as e:
          log_warning(f"Error processing audio: {e}")

  if response_delta.get("usage") is not None:
    mr.response_usage = _metrics(response_delta["usage"])
    model._calculate_cost_if_needed(mr.response_usage)

  model._augment_delta(response_delta, mr)
  return mr


class OpenAIStreamDecoder:
  """Flat `data: {...}` chunks terminated by `data: [DONE]`. Stateless per event;
  tool-call fragments are assembled later by `model.parse_tool_calls`."""

  def __init__(self, model: Any, response_format: Optional[Union[Dict, Type[BaseModel]]]):
    self._model = model

  def feed(self, event: SSEEvent) -> Iterator[ModelResponse]:
    if event.data == "[DONE]":
      return
    try:
      chunk = json.loads(event.data)
    except json.JSONDecodeError:
      log_warning(f"Skipping malformed SSE chunk from {self._model.provider}")
      return
    yield _parse_delta(self._model, chunk)

  def finish(self) -> Iterator[ModelResponse]:
    return iter(())


class OpenAITransform:
  def request(
    self,
    model: Any,
    messages: list[Message],
    tools: Optional[list[Dict[str, Any]]],
    response_format: Optional[Union[Dict, Type[BaseModel]]],
    tool_choice: Optional[Union[str, Dict[str, Any]]],
    stream: bool,
    compress_tool_results: bool,
  ) -> Request:
    params = model.get_request_params(response_format=response_format, tools=tools, tool_choice=tool_choice)
    extra_headers = params.pop("extra_headers", None)
    extra_query = params.pop("extra_query", None)
    extra_body = params.pop("extra_body", None)

    body: Dict[str, Any] = {"model": model.id, "messages": [_format_message(model, m, compress_tool_results) for m in messages]}
    body.update(params)
    if extra_body:
      body.update(extra_body)
    if stream:
      body["stream"] = True
      body["stream_options"] = {"include_usage": True}

    return Request(model._chat_endpoint(), model._build_headers(extra_headers), body, extra_query)

  def response(self, model: Any, raw: Dict[str, Any], response_format: Optional[Union[Dict, Type[BaseModel]]]) -> ModelResponse:
    return _parse_response(model, raw, response_format)

  def decoder(self, model: Any, response_format: Optional[Union[Dict, Type[BaseModel]]]) -> OpenAIStreamDecoder:
    return OpenAIStreamDecoder(model, response_format)


OPENAI_TRANSFORM = OpenAITransform()
