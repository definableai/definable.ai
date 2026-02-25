"""Pipeline — orchestrates phases with hooks."""

import asyncio
import time
from typing import (
  TYPE_CHECKING,
  Any,
  AsyncGenerator,
  Callable,
  Dict,
  List,
  Optional,
  Tuple,
  Union,
)

if TYPE_CHECKING:
  from definable.agent.loop import CancellationToken
  from definable.agent.pipeline.debug import DebugConfig
  from definable.agent.run.agent import PhaseCompletedEvent, PhaseStartedEvent

from definable.agent.pipeline.event_stream import EventStream
from definable.agent.pipeline.phase import Phase
from definable.agent.pipeline.state import LoopState, LoopStatus, PhaseMetric
from definable.agent.run.base import BaseRunOutputEvent
from definable.utils.log import log_debug, log_warning


class Pipeline:
  """Orchestrates pipeline phases with hooks and event emission.

  The Pipeline is built once at Agent.__init__ and reused for all runs.
  Hooks allow per-call customization without modifying the phase list.

  Example::

      pipeline = Pipeline(phases=[
          PreparePhase(),
          RecallPhase(),
          ThinkPhase(),
          GuardInputPhase(),
          ComposePhase(),
          InvokeLoopPhase(),
          GuardOutputPhase(),
          StorePhase(),
      ])

      # Register hooks
      @pipeline.hook("before:invoke_loop")
      async def log_messages(state):
          print(f"Messages: {len(state.invoke_messages)}")
          return state

      # Execute
      async for state, event in pipeline.execute(initial_state):
          if event:
              process(event)
  """

  def __init__(
    self,
    phases: Optional[List[Phase]] = None,
    *,
    debug: "Optional[DebugConfig]" = None,
  ) -> None:
    self._phases: List[Phase] = list(phases or [])
    self._hooks: Dict[str, List[Tuple[int, Callable]]] = {}  # spec → [(priority, callback)]
    self._event_stream: EventStream = EventStream()
    self._debug: "Optional[DebugConfig]" = debug

  # ── Phase manipulation ──────────────────────────────────────

  @property
  def phase_names(self) -> List[str]:
    """Ordered list of phase names."""
    return [p.name for p in self._phases]

  @property
  def phases(self) -> List[Phase]:
    """Read-only copy of phase list."""
    return list(self._phases)

  @property
  def event_stream(self) -> EventStream:
    """Access the unified event stream."""
    return self._event_stream

  def add_phase(
    self,
    phase: Phase,
    *,
    after: Optional[str] = None,
    before: Optional[str] = None,
  ) -> "Pipeline":
    """Insert a phase at a specific position.

    Args:
      phase: Phase to insert.
      after: Insert after the phase with this name.
      before: Insert before the phase with this name.
      If neither specified, appends to end.

    Returns:
      Self for chaining.

    Raises:
      ValueError: If the anchor phase name is not found.
    """
    if after and before:
      raise ValueError("Specify 'after' or 'before', not both.")

    if after:
      idx = self._find_phase_index(after)
      self._phases.insert(idx + 1, phase)
    elif before:
      idx = self._find_phase_index(before)
      self._phases.insert(idx, phase)
    else:
      self._phases.append(phase)

    self._validate_phase_order()
    return self

  def remove_phase(self, name: str) -> "Pipeline":
    """Remove a phase by name.

    Args:
      name: Phase name to remove.

    Returns:
      Self for chaining.

    Raises:
      ValueError: If no phase with that name exists.
    """
    idx = self._find_phase_index(name)
    self._phases.pop(idx)
    self._validate_phase_order()
    return self

  def replace_phase(self, name: str, new_phase: Phase) -> "Pipeline":
    """Replace a phase by name.

    Args:
      name: Phase name to replace.
      new_phase: Replacement phase.

    Returns:
      Self for chaining.

    Raises:
      ValueError: If no phase with that name exists.
    """
    idx = self._find_phase_index(name)
    self._phases[idx] = new_phase
    self._validate_phase_order()
    return self

  # ── Hook registration ───────────────────────────────────────

  def hook(
    self,
    spec: str,
    callback: Optional[Callable] = None,
    *,
    priority: int = 0,
  ) -> Union[Callable, Any]:
    """Register a hook on a phase.

    Hook spec format: ``"{timing}:{phase_name}"``
    - Timings: ``before``, ``after``, ``instead``
    - Phase: specific name or ``*`` (wildcard)

    Can be used as a decorator or called directly::

        @pipeline.hook("before:invoke_loop")
        async def my_hook(state):
            return state

        # Or directly:
        pipeline.hook("before:invoke_loop", my_callback)

    Args:
      spec: Hook specification string.
      callback: Hook function. If None, returns a decorator.
      priority: Lower runs first (default 0).

    Returns:
      Decorator if callback is None, else None.
    """
    self._validate_hook_spec(spec)

    if callback is not None:
      self._register_hook(spec, callback, priority)
      return callback

    def decorator(fn: Callable) -> Callable:
      self._register_hook(spec, fn, priority)
      return fn

    return decorator

  def remove_hook(self, spec: str, callback: Optional[Callable] = None) -> bool:
    """Remove a hook by spec (and optionally by callback).

    If callback is None, removes ALL hooks for that spec.
    If callback is provided, removes only that specific callback.

    Args:
      spec: Hook specification string (e.g. "before:invoke_loop").
      callback: Specific callback to remove. If None, removes all.

    Returns:
      True if any hooks were removed, False otherwise.
    """
    if spec not in self._hooks:
      return False

    if callback is None:
      # Remove all hooks for this spec
      del self._hooks[spec]
      return True

    # Remove specific callback
    original_len = len(self._hooks[spec])
    self._hooks[spec] = [(p, cb) for p, cb in self._hooks[spec] if cb is not callback]
    removed = len(self._hooks[spec]) < original_len

    if not self._hooks[spec]:
      del self._hooks[spec]

    return removed

  # ── Execution ───────────────────────────────────────────────

  async def execute(
    self,
    state: LoopState,
    *,
    cancellation_token: Optional["CancellationToken"] = None,
  ) -> AsyncGenerator[Tuple[LoopState, Optional[BaseRunOutputEvent]], None]:
    """Execute all phases in order, firing hooks around each.

    Yields (state, event) tuples. Events may be None.

    Phases with ``should_run(state) == False`` are skipped entirely —
    no hooks fire, no events emitted. A PhaseMetric with ``skipped=True``
    is recorded for observability.

    Args:
      state: Initial LoopState.
      cancellation_token: Optional CancellationToken for cooperative cancellation.
    """
    state.status = LoopStatus.running

    for phase in self._phases:
      # Check cancellation
      if cancellation_token:
        cancellation_token.raise_if_cancelled()

      # Check if phase should run
      if hasattr(phase, "should_run") and not phase.should_run(state):
        log_debug(f"Pipeline skipping phase: {phase.name} (should_run=False)")
        state.phase_metrics.append(PhaseMetric(phase_name=phase.name, duration_ms=0.0, skipped=True))
        continue

      state.phase = phase.name
      log_debug(f"Pipeline entering phase: {phase.name}")

      # Emit PhaseStartedEvent
      started_evt = _make_phase_started(state)
      await self._event_stream.emit(started_evt)
      yield state, started_evt

      phase_start = time.perf_counter()

      # Debug: breakpoint/inspector before phase
      if self._debug:
        if self._debug.step_mode or f"before:{phase.name}" in self._debug.breakpoints:
          if self._debug.inspector:
            result = self._debug.inspector(state, phase.name)
            if asyncio.iscoroutine(result):
              await result

        # Snapshot state for diff logging
        if self._debug.log_state_changes:
          pre_snapshot = _snapshot_state(state)

      # Fire before hooks
      state = await self._fire_hooks(f"before:{phase.name}", state)
      state = await self._fire_hooks("before:*", state)

      # Check for "instead" hooks (replaces phase entirely)
      instead_hooks = self._get_hooks(f"instead:{phase.name}")
      if instead_hooks:
        for _, hook_fn in instead_hooks:
          async for updated_state, event in hook_fn(state):
            state = updated_state
            if event is not None:
              await self._event_stream.emit(event)
            yield state, event
      else:
        # Execute the phase
        try:
          async for updated_state, event in phase.execute(state):
            state = updated_state
            if event is not None:
              await self._event_stream.emit(event)
            yield state, event
        except Exception as exc:
          state.status = LoopStatus.error
          log_warning(f"Phase {phase.name} raised: {exc!r}")
          raise

      duration_ms = (time.perf_counter() - phase_start) * 1000

      # Fire after hooks
      state = await self._fire_hooks(f"after:{phase.name}", state)
      state = await self._fire_hooks("after:*", state)

      # Record phase metric
      state.phase_metrics.append(PhaseMetric(phase_name=phase.name, duration_ms=duration_ms, skipped=False))

      # Emit PhaseCompletedEvent
      completed_evt = _make_phase_completed(state, phase.name, duration_ms, skipped=False)
      await self._event_stream.emit(completed_evt)
      yield state, completed_evt

      # Debug: log state changes
      if self._debug and self._debug.log_state_changes:
        _log_state_diff(pre_snapshot, state, phase.name)  # type: ignore[possibly-undefined]

      # Early exit on terminal states
      if state.status in (LoopStatus.blocked, LoopStatus.cancelled, LoopStatus.error):
        break

    if state.status == LoopStatus.running:
      state.status = LoopStatus.completed

  # ── Subscription ────────────────────────────────────────────

  def subscribe(self, handler: Callable) -> None:
    """Register a handler on the event stream."""
    self._event_stream.subscribe(handler)

  # ── Private helpers ─────────────────────────────────────────

  def _find_phase_index(self, name: str) -> int:
    """Find phase index by name or raise ValueError."""
    for i, p in enumerate(self._phases):
      if p.name == name:
        return i
    raise ValueError(f"Phase '{name}' not found. Available: {self.phase_names}")

  @staticmethod
  def _validate_hook_spec(spec: str) -> None:
    """Validate hook spec format."""
    parts = spec.split(":", 1)
    if len(parts) != 2:
      raise ValueError(f"Invalid hook spec '{spec}'. Expected format: 'timing:phase_name' (e.g., 'before:invoke_loop')")
    timing, _ = parts
    valid_timings = {"before", "after", "instead", "on"}
    if timing not in valid_timings:
      raise ValueError(f"Invalid hook timing '{timing}'. Valid: {valid_timings}")

  def _register_hook(self, spec: str, callback: Callable, priority: int) -> None:
    """Register a hook callback."""
    if spec not in self._hooks:
      self._hooks[spec] = []
    self._hooks[spec].append((priority, callback))
    # Keep sorted by priority (lower first)
    self._hooks[spec].sort(key=lambda x: x[0])

  def _get_hooks(self, spec: str) -> List[Tuple[int, Callable]]:
    """Get hooks for a spec, sorted by priority."""
    return self._hooks.get(spec, [])

  async def _fire_hooks(self, spec: str, state: LoopState) -> LoopState:
    """Fire all hooks for a spec, allowing state mutation.

    Each hook receives the current state and may return a modified
    state or None (no change).
    """
    hooks = self._get_hooks(spec)
    for _, hook_fn in hooks:
      try:
        result = hook_fn(state)
        if asyncio.iscoroutine(result):
          result = await result
        if result is not None and isinstance(result, LoopState):
          state = result
      except Exception as exc:
        log_warning(f"Hook {spec} raised: {exc!r}")
    return state

  def _validate_phase_order(self) -> None:
    """Validate that phase dependencies are satisfied by ordering.

    Emits warnings for any phase that requires fields not yet provided
    by earlier phases. Never errors — warnings only.
    """
    provided: set[str] = set()
    for phase in self._phases:
      if hasattr(phase, "requires"):
        missing = phase.requires - provided
        if missing:
          log_warning(f"Phase '{phase.name}' requires {missing} but no earlier phase provides them.")
      if hasattr(phase, "provides"):
        provided.update(phase.provides)


def _make_phase_started(state: LoopState) -> "PhaseStartedEvent":
  """Create a PhaseStartedEvent from current state."""
  from definable.agent.run.agent import PhaseStartedEvent

  return PhaseStartedEvent(
    run_id=state.run_id,
    session_id=state.session_id,
    agent_id=state.agent_id,
    agent_name=state.agent_name,
    phase_name=state.phase,
  )


def _make_phase_completed(state: LoopState, phase_name: str, duration_ms: float, *, skipped: bool) -> "PhaseCompletedEvent":
  """Create a PhaseCompletedEvent from current state."""
  from definable.agent.run.agent import PhaseCompletedEvent

  return PhaseCompletedEvent(
    run_id=state.run_id,
    session_id=state.session_id,
    agent_id=state.agent_id,
    agent_name=state.agent_name,
    phase_name=phase_name,
    duration_ms=duration_ms,
    skipped=skipped,
  )


def _snapshot_state(state: LoopState) -> dict[str, Any]:
  """Capture a shallow snapshot of state fields for diff logging."""
  import dataclasses

  snapshot: dict[str, Any] = {}
  for f in dataclasses.fields(state):
    val = getattr(state, f.name)
    # For mutable collections, snapshot length + identity
    if isinstance(val, (list, dict)):
      snapshot[f.name] = (type(val).__name__, len(val), id(val))
    else:
      snapshot[f.name] = val
  return snapshot


def _log_state_diff(pre: dict[str, Any], state: LoopState, phase_name: str) -> None:
  """Log fields that changed between pre-snapshot and current state."""
  import dataclasses
  import sys

  changes: list[str] = []
  for f in dataclasses.fields(state):
    val = getattr(state, f.name)
    pre_val = pre.get(f.name)
    if isinstance(val, (list, dict)):
      cur = (type(val).__name__, len(val), id(val))
      if cur != pre_val:
        changes.append(f"  {f.name}: {pre_val} → {cur}")
    elif val != pre_val:
      changes.append(f"  {f.name}: {pre_val!r} → {val!r}")

  if changes:
    msg = f"[Pipeline debug] State changes after '{phase_name}':\n" + "\n".join(changes)
    print(msg, file=sys.stderr)
