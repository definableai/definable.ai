"""Renderer registry and base protocol for CLI event rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Protocol, runtime_checkable

if TYPE_CHECKING:
  from rich.console import Console

  from definable.agent.interface.cli.config import CLIConfig
  from definable.agent.run.base import BaseRunOutputEvent


@runtime_checkable
class BaseRenderer(Protocol):
  """Protocol for CLI event renderers.

  Each renderer handles a specific category of agent events
  (tools, reasoning, memory, etc.) and renders them to the
  Rich console.
  """

  def handles(self, event: "BaseRunOutputEvent") -> bool:
    """Return True if this renderer should handle the given event."""
    ...

  def render(self, event: "BaseRunOutputEvent", console: "Console", config: "CLIConfig") -> None:
    """Render the event to the console."""
    ...


class RendererRegistry:
  """Dispatches events to the appropriate renderer.

  Renderers are checked in registration order — the first renderer
  whose ``handles()`` returns True gets to render the event.

  Example::

      registry = RendererRegistry()
      registry.add(ToolCallRenderer())
      registry.add(RunRenderer())
      registry.dispatch(event, console, config)
  """

  def __init__(self) -> None:
    self._renderers: List[BaseRenderer] = []

  def add(self, renderer: BaseRenderer) -> "RendererRegistry":
    """Register a renderer. Returns self for chaining."""
    self._renderers.append(renderer)
    return self

  def dispatch(self, event: "BaseRunOutputEvent", console: "Console", config: "CLIConfig") -> None:
    """Find the first matching renderer and render the event."""
    for renderer in self._renderers:
      if renderer.handles(event):
        renderer.render(event, console, config)
        return

  @property
  def renderer_count(self) -> int:
    return len(self._renderers)
