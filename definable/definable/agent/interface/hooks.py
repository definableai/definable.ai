"""Hook system for interfaces.

Hooks allow developers to intercept and modify messages at various
points in the message pipeline. All methods are optional — implement
only what you need.
"""

from typing import TYPE_CHECKING, Optional, Protocol, Set, runtime_checkable

from definable.utils.log import log_error, log_info

if TYPE_CHECKING:
  from definable.agent.interface.message import InterfaceMessage, InterfaceResponse
  from definable.agent.interface.session import InterfaceSession


@runtime_checkable
class InterfaceHook(Protocol):
  """Protocol for interface hooks.

  All methods are optional. Implement only the ones you need.
  The base class checks for method existence via ``hasattr()``.

  Methods:
    on_message_received: Called when a message arrives from the platform.
      Return False to veto (skip) the message.
    on_before_respond: Called before the agent processes the message.
      Can modify the message by returning a new InterfaceMessage.
    on_after_respond: Called after the agent produces a response.
      Can modify the response by returning a new InterfaceResponse.
    on_error: Called when an error occurs during processing.
  """

  async def on_message_received(self, message: "InterfaceMessage") -> Optional[bool]: ...

  async def on_before_respond(self, message: "InterfaceMessage", session: "InterfaceSession") -> Optional["InterfaceMessage"]: ...

  async def on_after_respond(
    self,
    message: "InterfaceMessage",
    response: "InterfaceResponse",
    session: "InterfaceSession",
  ) -> Optional["InterfaceResponse"]: ...

  async def on_error(self, error: Exception, message: Optional["InterfaceMessage"]) -> None: ...


class LoggingHook:
  """Hook that logs received messages and errors.

  Example:
    interface.add_hook(LoggingHook())
  """

  async def on_message_received(self, message: "InterfaceMessage") -> Optional[bool]:
    log_info(f"[{message.platform}] Message from user={message.platform_user_id} chat={message.platform_chat_id}: {message.text!r}")
    return None

  async def on_error(self, error: Exception, message: Optional["InterfaceMessage"]) -> None:
    if message:
      log_error(f"[{message.platform}] Error processing message from user={message.platform_user_id}: {error}")
    else:
      log_error(f"Interface error: {error}")


class AllowlistHook:
  """Hook that restricts access to a set of allowed user IDs.

  Messages from users not in the allowlist are silently dropped.

  Args:
    allowed_user_ids: Set of platform user IDs that are allowed.

  Example:
    interface.add_hook(AllowlistHook(allowed_user_ids={"123456", "789012"}))
  """

  def __init__(self, allowed_user_ids: Set[str]) -> None:
    self.allowed_user_ids = allowed_user_ids

  async def on_message_received(self, message: "InterfaceMessage") -> Optional[bool]:
    if message.platform_user_id not in self.allowed_user_ids:
      log_info(f"[{message.platform}] Blocked message from unauthorized user={message.platform_user_id}")
      return False
    return None
