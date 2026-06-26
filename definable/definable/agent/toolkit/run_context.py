"""RunContext — the per-run context object tools can receive.

A tool entrypoint (or its pre/post hook) that declares a ``run_context``
parameter gets this injected by ``FunctionCall.aexecute()``. It carries the
run identity plus the mutable session_state tools can read and update.

Relocated from the deleted ``definable.run`` package; the pipeline-era fields
(memory/research/readers context, active_layers) went with it — the harness-v2
loop never populated them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, Union

if TYPE_CHECKING:
  from pydantic import BaseModel


@dataclass
class RunContext:
  """Context passed to tools that declare a ``run_context`` parameter."""

  run_id: str
  session_id: str
  user_id: Optional[str] = None

  dependencies: Optional[Dict[str, Any]] = None
  metadata: Optional[Dict[str, Any]] = None
  session_state: Optional[Dict[str, Any]] = None
  output_schema: Optional[Union[Type["BaseModel"], Dict[str, Any]]] = None
