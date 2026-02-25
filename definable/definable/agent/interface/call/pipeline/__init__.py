"""Voice pipeline strategies for the call interface."""

from definable.agent.interface.call.pipeline.base import CallPipeline

__all__ = [
  "CallPipeline",
]


def __getattr__(name: str):
  if name == "CascadingPipeline":
    from definable.agent.interface.call.pipeline.cascading import CascadingPipeline

    return CascadingPipeline
  if name == "ManagedPipeline":
    from definable.agent.interface.call.pipeline.managed import ManagedPipeline

    return ManagedPipeline
  if name == "RealtimePipeline":
    from definable.agent.interface.call.pipeline.realtime import RealtimePipeline

    return RealtimePipeline
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
