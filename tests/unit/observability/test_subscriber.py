"""Observability subscriber — JSONL persistence sanity tests."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from definable.agent.core.events import EventBus, RunCompleted
from definable.observability import Observability, attach_jsonl


def _now() -> float:
  return time.time()


@pytest.mark.unit
def test_attach_jsonl_writes_one_line_per_event(tmp_path: Path) -> None:
  bus = EventBus()
  path = tmp_path / "trace.jsonl"
  close = attach_jsonl(bus, path)
  try:
    bus.emit(RunCompleted(run_id="r1", timestamp=_now(), content="ok", turns=1))
    bus.emit(RunCompleted(run_id="r1", timestamp=_now(), content="bye", turns=2))
  finally:
    close()
  lines = path.read_text(encoding="utf-8").strip().splitlines()
  assert len(lines) == 2
  d0 = json.loads(lines[0])
  d1 = json.loads(lines[1])
  assert d0["_type"] == "RunCompleted"
  assert d0["content"] == "ok"
  assert d1["content"] == "bye"


@pytest.mark.unit
def test_observability_wraps_jsonl_attach(tmp_path: Path) -> None:
  bus = EventBus()
  path = tmp_path / "obs.jsonl"
  obs = Observability(bus, jsonl_path=path)
  try:
    bus.emit(RunCompleted(run_id="r1", timestamp=_now(), content="x", turns=1))
  finally:
    obs.close()
  data = json.loads(path.read_text(encoding="utf-8").strip())
  assert data["_type"] == "RunCompleted"


@pytest.mark.unit
def test_observability_close_unsubscribes(tmp_path: Path) -> None:
  bus = EventBus()
  path = tmp_path / "x.jsonl"
  obs = Observability(bus, jsonl_path=path)
  bus.emit(RunCompleted(run_id="r1", timestamp=_now(), content="before", turns=1))
  obs.close()
  bus.emit(RunCompleted(run_id="r1", timestamp=_now(), content="after", turns=2))
  lines = path.read_text(encoding="utf-8").strip().splitlines()
  assert len(lines) == 1
