"""Trigger executor — calls handlers and processes return values."""

import inspect
from typing import TYPE_CHECKING, Any

from definable.agent.trigger.base import BaseTrigger, TriggerEvent
from definable.utils.log import log_error

if TYPE_CHECKING:
  from definable.agent.agent import Agent


class TriggerExecutor:
  """Executes trigger handlers and processes their return values.

  Return value semantics:
    - ``None`` → no-op
    - ``str`` → ``agent.arun(str)``
    - ``dict`` → ``agent.arun(**dict)``
    - awaitable → await the result and recurse
  """

  def __init__(self, agent: "Agent") -> None:
    self.agent = agent

  async def execute(self, trigger: BaseTrigger, event: TriggerEvent) -> Any:
    """Execute a trigger's handler and process the return value.

    Args:
      trigger: The trigger whose handler to call.
      event: The event to pass to the handler.

    Returns:
      The final result (typically a RunOutput or None).
    """
    if trigger.handler is None:
      return None

    try:
      result = trigger.handler(event)

      # If the handler is async, await it
      if inspect.isawaitable(result):
        result = await result

      return await self._process_result(result)
    except Exception as e:
      log_error(f"Trigger {trigger.name} handler failed: {e}")
      return None

  async def _process_result(self, result: Any) -> Any:
    """Process the handler's return value.

    Args:
      result: The value returned by the handler.

    Returns:
      The final result after processing.
    """
    if result is None:
      return None

    if isinstance(result, str):
      return await self.agent.arun(result)

    if isinstance(result, dict):
      return await self.agent.arun(**result)

    if inspect.isawaitable(result):
      resolved = await result
      return await self._process_result(resolved)

    return result
