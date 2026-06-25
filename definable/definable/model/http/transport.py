"""Raw-HTTP transport for model providers.

The single seam through which every dialect talks to a provider: a plain
``httpx`` POST (JSON in, JSON out) and a streaming POST (JSON in, SSE out).
There is no vendor SDK below this line — every payload is a plain ``dict`` and
every response is plain JSON, which is the whole point: it is fully ``pdb``-able
and the exact bytes can be logged.

Error mapping is centralized here so every provider funnels through
``classify_model_error`` and the resilience layer sees stable exception types.
"""

from collections.abc import AsyncIterator, Iterator
from typing import Any, Optional

import httpx

from definable.exceptions import ContextWindowExceededError, ModelAuthenticationError, ModelProviderError, classify_model_error
from definable.model.http.sse import SSEEvent, parse_sse_alines, parse_sse_lines
from definable.utils.http import get_default_async_client, get_default_sync_client
from definable.utils.log import log_debug, log_error

# Provider error codes that mean "context window exceeded" even when the
# human-readable message carries no recognizable pattern.
_CONTEXT_WINDOW_CODES = frozenset({"context_length_exceeded", "string_above_max_length"})


def _extract_error(response: httpx.Response) -> tuple[str, Optional[str]]:
  """Pull (message, code) out of a provider error body.

  OpenAI/Anthropic/Gemini all nest the error under ``error`` with a
  ``message`` (and OpenAI an additional ``code``); fall back to raw text.
  """
  try:
    body = response.json()
  except Exception:
    return response.text or "Unknown model error", None
  if isinstance(body, dict):
    err = body.get("error", body)
    if isinstance(err, dict):
      msg = err.get("message")
      code = err.get("code")
      if isinstance(msg, str) and msg:
        return msg, (code if isinstance(code, str) else None)
    if isinstance(err, str) and err:
      return err, None
  return response.text or "Unknown model error", None


def _raise_for_status(response: httpx.Response, model_name: str, model_id: str) -> None:
  """Translate a >=400 response into a Definable exception."""
  if response.status_code < 400:
    return
  message, code = _extract_error(response)
  log_error(f"HTTP {response.status_code} from {model_name} API: {message}")
  if response.status_code in (401, 403):
    raise ModelAuthenticationError(message=message, status_code=response.status_code, model_name=model_name)
  # A code can flag context exhaustion even when the message has no matchable pattern.
  if code in _CONTEXT_WINDOW_CODES:
    raise ContextWindowExceededError(message=message, status_code=response.status_code, model_name=model_name, model_id=model_id)
  # Otherwise classify by message pattern + status (rate limit, context window, generic).
  raise classify_model_error(message, status_code=response.status_code, model_name=model_name, model_id=model_id)


def _wrap_request_error(e: httpx.RequestError, model_name: str, model_id: str) -> ModelProviderError:
  log_error(f"HTTP request error to {model_name} API: {e}")
  return ModelProviderError(message=str(e), model_name=model_name, model_id=model_id)


def post_json(
  url: str,
  headers: dict[str, str],
  body: dict[str, Any],
  timeout: Optional[float],
  model_name: str,
  model_id: str,
  params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
  """Synchronous JSON POST."""
  log_debug(f"POST {url}", log_level=2)
  client = get_default_sync_client()
  try:
    response = client.post(url, headers=headers, json=body, params=params, timeout=timeout)
  except httpx.RequestError as e:
    raise _wrap_request_error(e, model_name, model_id) from e
  _raise_for_status(response, model_name, model_id)
  return response.json()


async def apost_json(
  url: str,
  headers: dict[str, str],
  body: dict[str, Any],
  timeout: Optional[float],
  model_name: str,
  model_id: str,
  params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
  """Asynchronous JSON POST."""
  log_debug(f"POST {url}", log_level=2)
  client = get_default_async_client()
  try:
    response = await client.post(url, headers=headers, json=body, params=params, timeout=timeout)
  except httpx.RequestError as e:
    raise _wrap_request_error(e, model_name, model_id) from e
  _raise_for_status(response, model_name, model_id)
  return response.json()


def stream_sse(
  url: str,
  headers: dict[str, str],
  body: dict[str, Any],
  timeout: Optional[float],
  model_name: str,
  model_id: str,
  params: Optional[dict[str, Any]] = None,
) -> Iterator[SSEEvent]:
  """Synchronous streaming POST yielding parsed SSE events."""
  log_debug(f"POST {url} (stream)", log_level=2)
  client = get_default_sync_client()
  try:
    with client.stream("POST", url, headers=headers, json=body, params=params, timeout=timeout) as response:
      if response.status_code >= 400:
        response.read()
        _raise_for_status(response, model_name, model_id)
      yield from parse_sse_lines(response.iter_lines())
  except httpx.RequestError as e:
    raise _wrap_request_error(e, model_name, model_id) from e


async def astream_sse(
  url: str,
  headers: dict[str, str],
  body: dict[str, Any],
  timeout: Optional[float],
  model_name: str,
  model_id: str,
  params: Optional[dict[str, Any]] = None,
) -> AsyncIterator[SSEEvent]:
  """Asynchronous streaming POST yielding parsed SSE events."""
  log_debug(f"POST {url} (stream)", log_level=2)
  client = get_default_async_client()
  try:
    async with client.stream("POST", url, headers=headers, json=body, params=params, timeout=timeout) as response:
      if response.status_code >= 400:
        await response.aread()
        _raise_for_status(response, model_name, model_id)
      async for event in parse_sse_alines(response.aiter_lines()):
        yield event
  except httpx.RequestError as e:
    raise _wrap_request_error(e, model_name, model_id) from e
