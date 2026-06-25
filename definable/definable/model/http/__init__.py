"""Raw-HTTP transport layer for model providers (no vendor SDKs).

See `wiki/adr-001-unified-http-model-layer.md`. The base Model subclasses
(OpenAIChat, Claude, Gemini) are the dialects — they build the wire payload and
parse the wire response. This package is the shared plumbing they sit on:
``transport`` (POST + SSE) and ``sse`` (event parsing).
"""

from definable.model.http.sse import SSEEvent, parse_sse_alines, parse_sse_lines
from definable.model.http.transport import apost_json, astream_sse, post_json, stream_sse

__all__ = [
  "SSEEvent",
  "parse_sse_lines",
  "parse_sse_alines",
  "post_json",
  "apost_json",
  "stream_sse",
  "astream_sse",
]
