"""Event trigger — fired programmatically via agent.emit()."""

from definable.agent.trigger.base import BaseTrigger


class EventTrigger(BaseTrigger):
  """Programmatic event trigger.

  Fires when ``agent.emit(event_name, data)`` is called with a
  matching event name.

  Args:
    event_name: Name of the event to listen for.

  Example::

    @agent.on(EventTrigger("user_signup"))
    async def on_signup(event):
        return f"Welcome {event.body['name']}!"

    # Later:
    agent.emit("user_signup", {"name": "Alice"})
  """

  def __init__(self, event_name: str) -> None:
    self.event_name = event_name

  @property
  def name(self) -> str:
    return f"event({self.event_name})"
