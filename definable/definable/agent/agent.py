"""Minimal Agent — the harness-v2 facade.

Phase 12 deliverable: the new Agent class. Lives alongside the existing
agent.py until Phase 13 swaps them. Hand-rolled, no inheritance, no
pipeline, no middleware, no hooks. The constructor stores config; arun()
delegates to `core.run()`. Read top to bottom.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator
from uuid import uuid4

from definable.agent.core import AgentEnd, AgentError, Event, EventBus, Hook, RunResult, ToolRegistry, run
from definable.agent.core.messages import build_messages
from definable.agent.memory import FileMemory, memory_tools
from definable.agent.toolkit.function import Function

if TYPE_CHECKING:
  from definable.agent.mcp.toolkit import MCPToolkit
  from definable.agent.skill.base import Skill
  from definable.agent.toolkit.base import Toolkit
  from definable.agent.toolkit.function import Function
  from definable.model.base import Model


class Agent:
  """Minimal agent harness. Read the constructor + arun() top to bottom.

  Twelve kwargs, no inheritance, no pipeline, no middleware, no hooks.
  Composition via the EventBus subscribers (`agent.events.on(EventType)`)
  and the ToolRegistry (toolkits / mcp / skills / memory tools all flatten).
  """

  def __init__(
    self,
    *,
    name: str,
    model: Model | str,
    instructions: str | None = None,
    tools: list[Function] | None = None,
    toolkits: list[Toolkit] | None = None,
    mcp: list[MCPToolkit] | None = None,
    skills: list[Skill] | None = None,
    memory: FileMemory | bool = False,
    session_id: str | None = None,
    max_turns: int = 50,
    hooks: list[Hook] | None = None,
    observability: Any | bool = False,
  ) -> None:
    self.name = name
    self.model = _resolve_model(model)
    self.instructions = instructions
    self.memory = _resolve_memory(memory, name)
    self.skills = skills or []
    self.session_id = session_id or str(uuid4())
    self.max_turns = max_turns
    self.hooks = hooks or []
    self.events = EventBus()

    # Memory exposes itself as four tools the agent can call.
    extra: list[Function] = list[Function](memory_tools(self.memory)) if self.memory is not None else []

    # Skills contribute their own tools.
    skill_tools: list[Function] = []
    for s in self.skills:
      tools_attr = getattr(s, "tools", None)
      if isinstance(tools_attr, list):
        skill_tools.extend(tools_attr)

    self.tools = ToolRegistry(
      tools=(tools or []) + extra + skill_tools,
      toolkits=toolkits,
      mcp=mcp,
      skills=None,  # already pulled into skill_tools above
    )

    self._async_toolkits: list[Any] = [tk for tk in [*(toolkits or []), *(mcp or [])] if hasattr(tk, "aopen") and hasattr(tk, "aclose")]
    self._opened = False

    # JSONL writer wires sync — sees every event from agent construction onward.
    self._obs: Any = _resolve_observability(observability, name, self.events)
    # Dashboard server boot is async — deferred to first aopen().
    self._obs_dashboard: bool = bool(observability)
    self._obs_attached: bool = False

  # ---- lifecycle ----------------------------------------------------------

  async def aopen(self) -> None:
    if self._opened:
      return
    for tk in self._async_toolkits:
      await tk.aopen()
    # Toolkits whose tools only become known post-aopen (MCP, browser,
    # anything that introspects a remote service) get re-extracted into
    # the registry now. Idempotent — already-registered names skip.
    for tk in self._async_toolkits:
      self.tools.add_from(tk)
    if self._obs_dashboard and not self._obs_attached:
      # Best-effort: dashboard requires fastapi+uvicorn (optional). If the
      # extras aren't installed the JSONL writer keeps working alone.
      try:
        from definable.observability.server import ObservabilityServer

        await ObservabilityServer.singleton().attach(self)
        self._obs_attached = True
      except ImportError:
        pass
    self._opened = True

  async def aclose(self) -> None:
    if not self._opened:
      return
    for tk in reversed(self._async_toolkits):
      # Best-effort teardown — log + continue
      with contextlib.suppress(Exception):
        await tk.aclose()
    self._opened = False

  async def __aenter__(self) -> Agent:
    await self.aopen()
    return self

  async def __aexit__(self, *args: Any) -> None:
    await self.aclose()

  # ---- run ---------------------------------------------------------------

  async def arun(
    self,
    input: str,  # noqa: A002 — `input` is the canonical agent prompt arg
    *,
    stream: bool = False,
    session_id: str | None = None,  # noqa: ARG002  # forward-compat: per-call session override
    output_schema: Any | None = None,
  ) -> RunResult | AsyncIterator[Event]:
    """Run a single turn or multi-turn loop until natural completion.

    `stream=False` returns a `RunResult`. `stream=True` returns an async
    iterator yielding `Event` objects in real time; the run completes
    after `AgentEnd` is yielded.
    """
    if not self._opened:
      await self.aopen()

    messages = build_messages(
      instructions=self.instructions,
      memory_index=self.memory.index() if self.memory is not None else None,
      skill_descriptions=[_skill_description(s) for s in self.skills],
      user_input=input,
    )
    run_id = str(uuid4())

    if stream:
      return self._arun_stream(messages, run_id=run_id, output_schema=output_schema)

    return await run(
      llm=self.model,
      messages=messages,
      tools=self.tools,
      events=self.events,
      hooks=self.hooks,
      memory=self.memory,
      stream=False,
      max_turns=self.max_turns,
      output_schema=output_schema,
      run_id=run_id,
    )

  async def _arun_stream(
    self,
    messages: list[Any],
    *,
    run_id: str,
    output_schema: Any | None,
  ) -> AsyncIterator[Event]:
    """Spawn the loop in a background task, yield events from the bus until AgentEnd."""
    task = asyncio.create_task(
      run(
        llm=self.model,
        messages=messages,
        tools=self.tools,
        events=self.events,
        hooks=self.hooks,
        memory=self.memory,
        stream=True,
        max_turns=self.max_turns,
        output_schema=output_schema,
        run_id=run_id,
      )
    )
    try:
      async for event in self.events.stream():
        if event.run_id != run_id:
          continue
        yield event
        if isinstance(event, (AgentEnd, AgentError)):
          break
    finally:
      if not task.done():
        task.cancel()
      with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


# ---- helpers -------------------------------------------------------------


def _skill_description(skill: Any) -> str:
  """Pull the system-prompt text from a Skill.

  Prefer get_instructions() when available; fall back to .instructions or
  .description (legacy attribute name some external code may use).
  """
  fn = getattr(skill, "get_instructions", None)
  if callable(fn):
    return fn() or ""
  return getattr(skill, "instructions", "") or getattr(skill, "description", "") or ""


def _resolve_model(model: Model | str) -> Model:
  if isinstance(model, str):
    from definable.model.utils import resolve_model_string

    return resolve_model_string(model)
  return model


def _resolve_memory(memory: FileMemory | bool, name: str) -> FileMemory | None:
  if memory is False:
    return None
  if memory is True:
    return FileMemory(Path(".definable/memory") / name)
  return memory


def _resolve_observability(observability: Any | bool, name: str, events: EventBus) -> Any:
  """observability=True wires a default JSONL writer; an instance gets attached as-is."""
  if observability is False or observability is None:
    return None
  from definable.observability import Observability

  if observability is True:
    return Observability(events, jsonl_path=Path(".definable/traces") / f"{name}.jsonl")
  if isinstance(observability, Observability):
    if observability.events is not events:
      # Re-attach to this agent's bus
      return Observability(events, jsonl_path=None)
    return observability
  return None
