"""Agent lifecycle management — startup, shutdown, toolkit init, audio transcription.

Functions take the agent instance as the first parameter. Called from Agent's
context managers and run methods.
"""

import asyncio
import contextlib
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.model.message import Message


async def drain_memory_tasks(agent: "Agent") -> None:
  """Await all pending memory background tasks (with timeout)."""
  if not agent._pending_memory_tasks:
    return
  done, _ = await asyncio.wait(agent._pending_memory_tasks, timeout=30.0)
  for task in done:
    with contextlib.suppress(Exception):
      task.result()
  agent._pending_memory_tasks = [t for t in agent._pending_memory_tasks if not t.done()]


def start(agent: "Agent") -> None:
  """Initialize resources."""
  if agent._started:
    return
  agent._started = True


def shutdown(agent: "Agent") -> None:
  """Cleanup resources (sync-safe).

  Closes memory, drains pending memory tasks, and shuts down
  agent-owned toolkits when called outside an async event loop.
  """
  for skill in agent.skills:
    with contextlib.suppress(Exception):
      skill.teardown()
  if agent._trace_writer:
    agent._trace_writer.shutdown()
  _sync_close_async_resources(agent)
  agent._started = False


def _sync_close_async_resources(agent: "Agent") -> None:
  """Close memory, toolkits, and drain tasks from a sync context."""

  async def _cleanup() -> None:
    await drain_memory_tasks(agent)
    for toolkit in agent._agent_owned_toolkits:
      with contextlib.suppress(Exception):
        await toolkit.shutdown()
    agent._agent_owned_toolkits.clear()
    if agent.memory:
      with contextlib.suppress(Exception):
        await agent.memory.close()

  try:
    asyncio.get_running_loop()
    from definable.utils.log import log_warning

    log_warning(
      "Agent._shutdown() cannot close async resources (memory, toolkits) "
      "from inside a running event loop. Use 'async with Agent(...)' or "
      "await agent._ashutdown() instead."
    )
  except RuntimeError:
    with contextlib.suppress(Exception):
      asyncio.run(_cleanup())


async def ashutdown(agent: "Agent") -> None:
  """Async cleanup."""
  await drain_memory_tasks(agent)
  for toolkit in agent._agent_owned_toolkits:
    with contextlib.suppress(Exception):
      await toolkit.shutdown()
  agent._agent_owned_toolkits.clear()
  if agent.memory:
    with contextlib.suppress(Exception):
      await agent.memory.close()
  for skill in agent.skills:
    with contextlib.suppress(Exception):
      skill.teardown()
  if agent._trace_writer:
    agent._trace_writer.shutdown()
  agent._started = False


async def ensure_toolkits_initialized(agent: "Agent") -> None:
  """Initialize any AsyncLifecycleToolkit instances that aren't yet initialized."""
  from definable.agent.agent import AsyncLifecycleToolkit
  from definable.agent.resolution import flatten_tools

  async with agent._toolkit_init_lock:
    needs_refresh = False
    for toolkit in agent.toolkits:
      if isinstance(toolkit, AsyncLifecycleToolkit) and not toolkit._initialized:
        try:
          await toolkit.initialize()
          agent._agent_owned_toolkits.append(toolkit)
          needs_refresh = True
        except Exception as e:
          from definable.utils.log import log_warning

          log_warning(f"Toolkit {toolkit!r} init failed (non-fatal): {e}")
    if needs_refresh:
      agent._tools_dict = flatten_tools(agent.skills, agent.toolkits, agent.tools)


async def transcribe_audio(agent: "Agent", messages: "List[Message]") -> None:
  """Transcribe audio in messages if a transcriber is configured.

  Mutates messages in-place: populates ``audio.transcript``, injects
  the transcript text into ``message.content``, and clears ``message.audio``.
  """
  if agent._audio_transcriber is None:
    return

  for msg in messages:
    if not msg.audio:
      continue
    transcripts: List[str] = []
    for audio_item in msg.audio:
      if audio_item.transcript:
        transcripts.append(audio_item.transcript)
        continue
      audio_bytes = audio_item.get_content_bytes()
      if audio_bytes is None:
        continue
      mime = audio_item.mime_type or "audio/ogg"
      try:
        text = await agent._audio_transcriber.atranscribe(audio_bytes, mime)
      except Exception as e:
        from definable.utils.log import log_warning

        log_warning(f"Audio transcription failed: {e}")
        continue
      audio_item.transcript = text
      transcripts.append(text)
    if transcripts:
      transcript_text = "\n".join(transcripts)
      if msg.content:
        msg.content = f"{msg.content}\n\n{transcript_text}"
      else:
        msg.content = transcript_text
      msg.audio = None
