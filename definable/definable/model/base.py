"""The Model contract — one slim seam every provider sits behind.

A model call is just: **input → transform to the provider's wire shape → HTTP →
transform the wire response back to our unified `ModelResponse`.** That is the
whole job. This file owns the HTTP plumbing (the four `invoke*` methods) exactly
once; a provider supplies a `Transform` (build request / parse response / parse
stream) and a few config fields. Nothing else.

Adding a provider = write `transform.py` (3 functions) + a config `model.py`.
No `invoke` code, no tool loop, no orchestration. The agent layer
(`agent/core/loop.py`) owns the tool-call loop, retries, and HITL — the model
just does one request/response.

A provider that can't talk raw HTTP can instead override `invoke*` directly and
leave `transform = None`; every current provider supplies a `Transform`.
"""

import os
from abc import ABC
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any, AsyncIterator, ClassVar, Dict, Iterator, List, Optional, Protocol, Sequence, Type, Union, runtime_checkable

from pydantic import BaseModel

from definable.model.http import apost_json, astream_sse, post_json, stream_sse
from definable.model.http.sse import SSEEvent
from definable.model.message import Message
from definable.model.metrics import Metrics
from definable.model.response import ModelResponse
from definable.types import ToolCallDict

if TYPE_CHECKING:
  from definable.model.pricing import ModelSpec

# --- test guard: block accidental real API calls ---------------------------

_ALLOW_MODEL_REQUESTS: Optional[bool] = None


def _check_model_requests_allowed() -> None:
  """Raise if model requests are blocked via ALLOW_MODEL_REQUESTS=0 (test safety)."""
  global _ALLOW_MODEL_REQUESTS
  if _ALLOW_MODEL_REQUESTS is None:
    _ALLOW_MODEL_REQUESTS = os.environ.get("ALLOW_MODEL_REQUESTS", "1") != "0"
  if not _ALLOW_MODEL_REQUESTS:
    raise RuntimeError("Model requests are blocked (ALLOW_MODEL_REQUESTS=0). Set ALLOW_MODEL_REQUESTS=1 or use the allow_model_requests fixture.")


def override_allow_model_requests(allowed: bool) -> None:
  """Override the ALLOW_MODEL_REQUESTS guard at runtime (test fixtures)."""
  global _ALLOW_MODEL_REQUESTS
  _ALLOW_MODEL_REQUESTS = allowed


# --- timing: one Metrics object per call, owned by the response ------------
# The model's single source of usage is `ModelResponse.response_usage` — the
# provider fills its token counts; these helpers stamp wall-clock timing onto
# that same object so one Metrics carries everything. No caller-supplied
# container, no second metrics object.


def _stamp_timing(resp: ModelResponse, duration: float) -> None:
  """Non-streaming: stamp duration onto the response's usage metrics.

  The whole response arrives as one block, so first-token latency equals total
  duration. Creates the metrics if the provider reported no token usage, so
  timing is always available.
  """
  if resp.response_usage is None:
    resp.response_usage = Metrics()
  resp.response_usage.duration = duration
  resp.response_usage.time_to_first_token = duration


def _stamp_stream_timing(usage: Optional[Metrics], duration: float, ttft: Optional[float]) -> Optional[ModelResponse]:
  """Streaming: stamp timing onto the call's single Metrics object.

  Normal path — the provider emitted a usage delta (OpenAI `include_usage`,
  Anthropic `message_delta`, Gemini `usageMetadata`): mutate that same object
  by reference (the caller holds it), return None, no extra delta.

  Fallback — the stream produced content/tool calls but no usage delta: the
  caller has nothing to stamp onto, so return a final timing-only delta to
  yield. Keeps parity with the non-streaming path, which always records timing.
  A truly empty stream (no content, no usage) gets nothing.
  """
  if usage is not None:
    usage.duration = duration
    usage.time_to_first_token = ttft
    return None
  if ttft is None:
    return None
  return ModelResponse(response_usage=Metrics(duration=duration, time_to_first_token=ttft))


# --- the transform seam ----------------------------------------------------


@dataclass
class Request:
  """The provider's wire request: where to POST, with what headers/body."""

  url: str
  headers: Dict[str, str]
  body: Dict[str, Any]
  params: Optional[Dict[str, Any]] = None  # query string params


class StreamDecoder(Protocol):
  """Stateful per-stream decoder: feed SSE events, get unified deltas.

  One instance per stream so it can carry cross-event state (tool-arg
  accumulation, split usage). Drives both the sync and async invoke paths.
  """

  def feed(self, event: SSEEvent) -> Iterator[ModelResponse]: ...

  def finish(self) -> Iterator[ModelResponse]: ...


@runtime_checkable
class Transform(Protocol):
  """A provider dialect — the only thing a new provider must write.

  Over a `Model` (which carries the config/credentials):
    - `request`: unified messages/tools/params -> provider wire `Request`
    - `response`: provider wire JSON           -> unified `ModelResponse`
    - `decoder`: a fresh `StreamDecoder` for one streaming response
  """

  def request(
    self,
    model: "Model",
    messages: List[Message],
    tools: Optional[List[Dict[str, Any]]],
    response_format: Optional[Union[Dict, Type[BaseModel]]],
    tool_choice: Optional[Union[str, Dict[str, Any]]],
    stream: bool,
    compress_tool_results: bool,
  ) -> Request: ...

  def response(self, model: "Model", raw: Dict[str, Any], response_format: Optional[Union[Dict, Type[BaseModel]]]) -> ModelResponse: ...

  def decoder(self, model: "Model", response_format: Optional[Union[Dict, Type[BaseModel]]]) -> StreamDecoder: ...


# --- the model -------------------------------------------------------------


@dataclass
class Model(ABC):
  # Identity (not sent to the API).
  id: str
  name: Optional[str] = None
  provider: Optional[str] = None
  client: Any = None  # SDK providers keep their client handle here
  context_window: Optional[int] = None

  # Credentials / endpoint — common to every HTTP provider.
  api_key: Optional[str] = None
  base_url: Optional[str] = None
  timeout: Optional[float] = None

  # Capability flags (set by Agent, read by tests/observability).
  supports_native_structured_outputs: bool = False
  supports_native_thinking: bool = False
  supports_json_schema_outputs: bool = False
  # True when the provider re-reports cumulative token counts each streaming
  # chunk (Perplexity) — keep the latest instead of summing.
  collect_metrics_on_completion: bool = False

  # Optional system prompt / instructions the model contributes to the Agent.
  system_prompt: Optional[str] = None
  instructions: Optional[List[str]] = None

  # The dialect. Raw-HTTP providers set this to their Transform; SDK providers
  # leave it None and override the invoke* methods instead.
  transform: ClassVar[Optional[Transform]] = None

  def __post_init__(self) -> None:
    if self.provider is None and self.name is not None:
      self.provider = f"{self.name} ({self.id})"

  # --- the seam: input -> transform -> HTTP -> transform ---------------------

  def invoke(
    self,
    messages: List[Message],
    response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    compress_tool_results: bool = False,
  ) -> ModelResponse:
    _check_model_requests_allowed()
    self._resolve_auth()
    spec = self.spec()
    req = self._transform().request(self, messages, tools, response_format, tool_choice, False, compress_tool_results)
    t0 = perf_counter()
    raw = post_json(req.url, req.headers, req.body, self.timeout, self.name or self.id, self.id, params=req.params)
    resp = self._transform().response(self, raw, response_format)
    _stamp_timing(resp, perf_counter() - t0)
    self._apply_cost(resp.response_usage, spec)
    return self._ensure_role(resp)

  async def ainvoke(
    self,
    messages: List[Message],
    response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    compress_tool_results: bool = False,
  ) -> ModelResponse:
    _check_model_requests_allowed()
    self._resolve_auth()
    spec = self.spec()
    req = self._transform().request(self, messages, tools, response_format, tool_choice, False, compress_tool_results)
    t0 = perf_counter()
    raw = await apost_json(req.url, req.headers, req.body, self.timeout, self.name or self.id, self.id, params=req.params)
    resp = self._transform().response(self, raw, response_format)
    _stamp_timing(resp, perf_counter() - t0)
    self._apply_cost(resp.response_usage, spec)
    return self._ensure_role(resp)

  def invoke_stream(
    self,
    messages: List[Message],
    response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    compress_tool_results: bool = False,
  ) -> Iterator[ModelResponse]:
    _check_model_requests_allowed()
    self._resolve_auth()
    spec = self.spec()
    req = self._transform().request(self, messages, tools, response_format, tool_choice, True, compress_tool_results)
    decoder = self._transform().decoder(self, response_format)
    t0 = perf_counter()
    ttft: Optional[float] = None
    usage: Optional[Metrics] = None
    pending_usage = None
    for event in stream_sse(req.url, req.headers, req.body, self.timeout, self.name or self.id, self.id, params=req.params):
      for delta in decoder.feed(event):
        if ttft is None and (delta.content or delta.tool_calls):
          ttft = perf_counter() - t0
        if delta.response_usage is not None:
          usage = delta.response_usage
          # A pure-usage delta (no content/tool_calls) is held back so cost +
          # timing can be stamped onto it *before* the consumer sees it. Whether
          # usage rides a mid-stream feed() chunk (OpenAI) or a finish() delta
          # (Anthropic/Gemini), the terminal usage event reaches an eager reader
          # already stamped — not mutated a tick after it was consumed.
          if not (delta.content or delta.tool_calls):
            pending_usage = delta
            continue
        yield self._ensure_role(delta)
    finals = list(decoder.finish())
    for delta in finals:
      if delta.response_usage is not None:
        usage = delta.response_usage
        if pending_usage is None and not (delta.content or delta.tool_calls):
          pending_usage = delta
    if pending_usage is not None:
      usage = pending_usage.response_usage  # stamp the exact metrics that rides the emitted terminal delta
    self._apply_cost(usage, spec)
    extra = _stamp_stream_timing(usage, perf_counter() - t0, ttft)
    for delta in finals:
      if delta is not pending_usage:
        yield self._ensure_role(delta)
    if pending_usage is not None:
      yield self._ensure_role(pending_usage)
    if extra is not None:
      yield self._ensure_role(extra)

  async def ainvoke_stream(
    self,
    messages: List[Message],
    response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    compress_tool_results: bool = False,
  ) -> AsyncIterator[ModelResponse]:
    _check_model_requests_allowed()
    self._resolve_auth()
    spec = self.spec()
    req = self._transform().request(self, messages, tools, response_format, tool_choice, True, compress_tool_results)
    decoder = self._transform().decoder(self, response_format)
    t0 = perf_counter()
    ttft: Optional[float] = None
    usage: Optional[Metrics] = None
    pending_usage = None
    async for event in astream_sse(req.url, req.headers, req.body, self.timeout, self.name or self.id, self.id, params=req.params):
      for delta in decoder.feed(event):
        if ttft is None and (delta.content or delta.tool_calls):
          ttft = perf_counter() - t0
        if delta.response_usage is not None:
          usage = delta.response_usage
          # A pure-usage delta (no content/tool_calls) is held back so cost +
          # timing can be stamped onto it *before* the consumer sees it. Whether
          # usage rides a mid-stream feed() chunk (OpenAI) or a finish() delta
          # (Anthropic/Gemini), the terminal usage event reaches an eager reader
          # already stamped — not mutated a tick after it was consumed.
          if not (delta.content or delta.tool_calls):
            pending_usage = delta
            continue
        yield self._ensure_role(delta)
    finals = list(decoder.finish())
    for delta in finals:
      if delta.response_usage is not None:
        usage = delta.response_usage
        if pending_usage is None and not (delta.content or delta.tool_calls):
          pending_usage = delta
    if pending_usage is not None:
      usage = pending_usage.response_usage  # stamp the exact metrics that rides the emitted terminal delta
    self._apply_cost(usage, spec)
    extra = _stamp_stream_timing(usage, perf_counter() - t0, ttft)
    for delta in finals:
      if delta is not pending_usage:
        yield self._ensure_role(delta)
    if pending_usage is not None:
      yield self._ensure_role(pending_usage)
    if extra is not None:
      yield self._ensure_role(extra)

  # --- provider hooks (override as needed) ----------------------------------

  def _transform(self) -> Transform:
    if self.transform is None:
      raise NotImplementedError(f"{type(self).__name__} has no `transform` set and does not override invoke().")
    return self.transform

  def _resolve_auth(self) -> None:
    """Resolve + validate credentials from env. Default: no-op. Providers override."""
    return None

  def parse_tool_calls(self, tool_calls_data: list[ToolCallDict]) -> list[ToolCallDict]:
    """Assemble streamed tool-call fragments into complete tool calls.

    Default: the deltas are already complete (Anthropic). OpenAI overrides to
    accumulate fragments keyed by `index`.
    """
    return tool_calls_data

  # --- shared helpers --------------------------------------------------------

  def _format_tools(self, tools: Optional[List[Any]]) -> List[Dict[str, Any]]:
    """Wrap Function objects in the `{type:function, function:{...}}` envelope."""
    from definable.agent.toolkit.function import Function

    out: List[Dict[str, Any]] = []
    for tool in tools or []:
      out.append({"type": "function", "function": tool.to_dict()} if isinstance(tool, Function) else tool)
    return out

  def spec(self) -> Optional["ModelSpec"]:
    """This model's spec sheet — pricing, capabilities, modalities, limits — from
    the per-provider JSON in ``model/pricing/``, or None if the model isn't
    catalogued. The spec sheet is the single source of truth for capabilities."""
    from definable.model.pricing import get_spec

    return get_spec(self.get_provider(), self.id)

  def _apply_cost(self, metrics: Optional[Metrics], spec: Optional["ModelSpec"]) -> None:
    """Stamp cost onto the response metrics from the spec sheet's pricing."""
    if metrics is not None and metrics.cost is None and spec is not None:
      metrics.cost = spec.pricing.calculate_cost(metrics)

  @staticmethod
  def _ensure_role(resp: ModelResponse) -> ModelResponse:
    """The gateway's role invariant: every ModelResponse it emits carries the
    unified ``assistant`` role. Providers report role inconsistently — some only
    on the first stream chunk, Gemini as ``model``, none on streamed deltas — so
    the seam normalizes it once, here, for both the streamed and non-streamed
    paths. A model response is always the assistant turn."""
    if resp.role is None:
      resp.role = "assistant"
    return resp

  def count_tokens(
    self,
    messages: List[Message],
    tools: Optional[Sequence[Any]] = None,
    output_schema: Optional[Union[Dict, Type[BaseModel]]] = None,
  ) -> int:
    from definable.tokens import count_tokens

    return count_tokens(messages, tools=list(tools) if tools else None, model_id=self.id, output_schema=output_schema)

  async def acount_tokens(
    self,
    messages: List[Message],
    tools: Optional[Sequence[Any]] = None,
    output_schema: Optional[Union[Dict, Type[BaseModel]]] = None,
  ) -> int:
    return self.count_tokens(messages, tools, output_schema=output_schema)

  def get_provider(self) -> str:
    return self.provider or self.name or self.__class__.__name__

  def _remove_temporary_messages(self, messages: List[Message]) -> None:
    """Drop messages flagged temporary (in place)."""
    messages[:] = [m for m in messages if not m.temporary]

  def get_system_message_for_model(self, tools: Optional[List[Any]] = None) -> Optional[str]:
    return self.system_prompt

  def get_instructions_for_model(self, tools: Optional[List[Any]] = None) -> Optional[List[str]]:
    return self.instructions

  def to_dict(self) -> Dict[str, Any]:
    return {f: getattr(self, f) for f in ("name", "id", "provider") if getattr(self, f) is not None}

  def __deepcopy__(self, memo: Dict[int, Any]) -> "Model":
    from copy import copy, deepcopy

    cls = self.__class__
    new_model = cls.__new__(cls)
    memo[id(self)] = new_model
    for k, v in self.__dict__.items():
      if k in {"client", "async_client", "http_client"}:
        setattr(new_model, k, None)
        continue
      try:
        setattr(new_model, k, deepcopy(v, memo))
      except Exception:
        try:
          setattr(new_model, k, copy(v))
        except Exception:
          setattr(new_model, k, v)
    return new_model
