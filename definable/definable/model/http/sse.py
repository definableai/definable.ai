"""Minimal Server-Sent Events parser.

Providers stream completions as SSE. The wire format is identical across
OpenAI / Anthropic / Gemini (the W3C SSE grammar); only the event payloads
differ. This parser turns a stream of raw lines (from ``httpx`` ``iter_lines``
/ ``aiter_lines``) into ``SSEEvent`` records — a dialect then interprets the
``data`` payload.

We deliberately do not depend on ``httpx-sse``: the grammar we need is ~20
lines and a dependency for that is not worth it (ponytail).
"""

from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass
from typing import Optional


@dataclass
class SSEEvent:
  """One dispatched SSE event.

  ``event`` is the ``event:`` field (``None`` for the common data-only framing
  used by OpenAI/Gemini; set by Anthropic's named events). ``data`` is the
  concatenation of all ``data:`` lines in the event, newline-joined per spec.
  """

  event: Optional[str]
  data: str


def _strip_value(line: str, field_len: int) -> str:
  # Per spec, a single leading space after the colon is removed.
  value = line[field_len + 1 :]
  if value.startswith(" "):
    value = value[1:]
  return value


def parse_sse_lines(lines: Iterable[str]) -> Iterator[SSEEvent]:
  """Parse decoded SSE lines (no trailing newlines) into events.

  A blank line dispatches the buffered event. Comment lines (starting with
  ``:``, e.g. Anthropic ``ping``) are ignored. A trailing event with no final
  blank line is flushed at end-of-stream.
  """
  event_type: Optional[str] = None
  data_lines: list[str] = []

  for line in lines:
    if line == "":
      if data_lines:
        yield SSEEvent(event_type, "\n".join(data_lines))
      event_type = None
      data_lines = []
      continue
    if line.startswith(":"):
      continue
    if line.startswith("data:"):
      data_lines.append(_strip_value(line, 4))
    elif line.startswith("event:"):
      event_type = _strip_value(line, 5)
    # other fields (id:, retry:) are not used by chat providers — ignore

  if data_lines:
    yield SSEEvent(event_type, "\n".join(data_lines))


async def parse_sse_alines(lines: AsyncIterator[str]) -> AsyncIterator[SSEEvent]:
  """Async counterpart of :func:`parse_sse_lines`."""
  event_type: Optional[str] = None
  data_lines: list[str] = []

  async for line in lines:
    if line == "":
      if data_lines:
        yield SSEEvent(event_type, "\n".join(data_lines))
      event_type = None
      data_lines = []
      continue
    if line.startswith(":"):
      continue
    if line.startswith("data:"):
      data_lines.append(_strip_value(line, 4))
    elif line.startswith("event:"):
      event_type = _strip_value(line, 5)

  if data_lines:
    yield SSEEvent(event_type, "\n".join(data_lines))
