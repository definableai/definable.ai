"""EventBus subscribers that pipe events to durable sinks.

Phase 15: minimal — JSONL only. Dashboard / OTel exporters land later if
needed; the EventBus is the single integration point so adding a new
subscriber never touches the harness.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
  from definable.agent.core.events import Event, EventBus


def attach_jsonl(events: EventBus, path: Path | str) -> Callable[[], None]:
  """Subscribe a JSONL writer to `events`. Returns a `close()` callable.

  Each emitted event is serialized as a single JSON line with `_type` set
  to the event class name. The file is opened in append mode and flushed
  after every line so durability survives a crash.
  """
  p = Path(path)
  p.parent.mkdir(parents=True, exist_ok=True)
  f = p.open("a", encoding="utf-8")

  def _write(e: Event) -> None:
    with contextlib.suppress(Exception):
      data = dataclasses.asdict(e)
      data["_type"] = e.__class__.__name__
      f.write(json.dumps(data, default=str) + "\n")
      f.flush()

  unsub = events.subscribe(_write)

  def close() -> None:
    unsub()
    with contextlib.suppress(Exception):
      f.close()

  return close


class Observability:
  """Convenience wrapper that attaches one or more subscribers to an EventBus.

  Today this only knows about JSONL persistence — but the shape (subscribe
  on construct, `.close()` on shutdown) is stable for adding more sinks.
  """

  def __init__(
    self,
    events: EventBus,
    *,
    jsonl_path: str | Path | None = None,
  ) -> None:
    self.events = events
    self._closes: list[Callable[[], None]] = []
    if jsonl_path is not None:
      self._closes.append(attach_jsonl(events, jsonl_path))

  def close(self) -> None:
    for c in self._closes:
      with contextlib.suppress(Exception):
        c()
    self._closes.clear()
