"""Phase protocol and base class for pipeline blocks."""

from typing import AsyncGenerator, Optional, Protocol, Set, Tuple, runtime_checkable

from definable.agent.pipeline.state import LoopState
from definable.run.base import BaseRunOutputEvent


@runtime_checkable
class Phase(Protocol):
  """Protocol for pipeline phases.

  Each phase is an async generator that receives mutable LoopState
  and yields (state, event) tuples. The event may be None if the
  phase has no event to emit for that step.

  Example::

      class MyPhase:
          @property
          def name(self) -> str:
              return "my_phase"

          async def execute(self, state):
              state.extra["my_data"] = "hello"
              yield state, None
  """

  @property
  def name(self) -> str: ...

  def execute(self, state: LoopState) -> AsyncGenerator[Tuple[LoopState, Optional[BaseRunOutputEvent]], None]: ...


class BasePhase:
  """Convenience base class for pipeline phases.

  Subclasses must implement ``_name`` and ``execute()``.
  Optionally override ``should_run()`` for conditional execution
  and declare ``_requires``/``_provides`` for dependency validation.
  """

  _name: str = ""
  _requires: Set[str] = set()
  _provides: Set[str] = set()

  @property
  def name(self) -> str:
    return self._name

  @property
  def requires(self) -> Set[str]:
    """LoopState fields this phase reads."""
    return self._requires

  @property
  def provides(self) -> Set[str]:
    """LoopState fields this phase writes."""
    return self._provides

  def should_run(self, state: LoopState) -> bool:
    """Return True if this phase should execute. Default: always run.

    Override in subclasses to skip phases dynamically based on state
    or agent configuration. When False, the phase is entirely skipped
    — no hooks fire, no events emitted.
    """
    return True

  async def execute(self, state: LoopState) -> AsyncGenerator[Tuple[LoopState, Optional[BaseRunOutputEvent]], None]:
    yield state, None
