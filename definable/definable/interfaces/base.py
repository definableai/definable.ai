"""Base interface for connecting agents to messaging platforms."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from definable.agents.agent import Agent
from definable.interfaces.config import InterfaceConfig
from definable.interfaces.hooks import InterfaceHook
from definable.interfaces.message import InterfaceMessage, InterfaceResponse
from definable.interfaces.session import InterfaceSession, SessionManager
from definable.run.agent import RunOutput
from definable.utils.log import log_error, log_info


class BaseInterface(ABC):
  """Abstract base class for platform interfaces.

  Provides the full message pipeline from platform message reception
  through agent execution to response delivery, with hook support
  at every stage.

  Subclasses must implement the four abstract methods to integrate
  with a specific messaging platform.

  Args:
    agent: The Agent instance to connect.
    config: Interface configuration.
    session_manager: Optional session manager (created automatically if not provided).
    hooks: Optional list of hooks to register.

  Example:
    class MyPlatformInterface(BaseInterface):
      async def _start_receiver(self): ...
      async def _stop_receiver(self): ...
      async def _convert_inbound(self, raw_message): ...
      async def _send_response(self, original_msg, response, raw_message): ...
  """

  def __init__(
    self,
    *,
    agent: Agent,
    config: InterfaceConfig,
    session_manager: Optional[SessionManager] = None,
    hooks: Optional[List[InterfaceHook]] = None,
  ) -> None:
    self.agent = agent
    self.config = config
    self.session_manager = session_manager or SessionManager(
      session_ttl_seconds=config.session_ttl_seconds,
    )
    self._hooks: List[InterfaceHook] = list(hooks or [])
    self._running = False
    self._request_semaphore: Optional[asyncio.Semaphore] = None

  # --- Hook management ---

  def add_hook(self, hook: InterfaceHook) -> "BaseInterface":
    """Add a hook to the interface.

    Args:
      hook: Hook instance to add.

    Returns:
      Self for method chaining.
    """
    self._hooks.append(hook)
    return self

  # --- Lifecycle ---

  async def start(self) -> None:
    """Start the interface (begin receiving messages)."""
    if self._running:
      return
    self._request_semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
    self._running = True
    log_info(f"[{self.config.platform}] Interface starting")
    await self._start_receiver()
    log_info(f"[{self.config.platform}] Interface started")

  async def stop(self) -> None:
    """Stop the interface (stop receiving messages)."""
    if not self._running:
      return
    self._running = False
    log_info(f"[{self.config.platform}] Interface stopping")
    await self._stop_receiver()
    log_info(f"[{self.config.platform}] Interface stopped")

  async def serve_forever(self) -> None:
    """Block until the interface is stopped (e.g. via signal or stop())."""
    if not self._running:
      await self.start()
    try:
      while self._running:
        await asyncio.sleep(1)
    except asyncio.CancelledError:
      pass
    finally:
      await self.stop()

  async def __aenter__(self) -> "BaseInterface":
    """Async context manager entry."""
    await self.start()
    return self

  async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    """Async context manager exit."""
    await self.stop()

  # --- Abstract methods (platform implementations) ---

  @abstractmethod
  async def _start_receiver(self) -> None:
    """Start the platform-specific message receiver (polling, webhook, etc.)."""
    ...

  @abstractmethod
  async def _stop_receiver(self) -> None:
    """Stop the platform-specific message receiver. Must be idempotent."""
    ...

  @abstractmethod
  async def _convert_inbound(self, raw_message: Any) -> Optional[InterfaceMessage]:
    """Convert a platform-specific message to an InterfaceMessage.

    Args:
      raw_message: The raw message from the platform.

    Returns:
      InterfaceMessage, or None to skip this message.
    """
    ...

  @abstractmethod
  async def _send_response(
    self,
    original_msg: InterfaceMessage,
    response: InterfaceResponse,
    raw_message: Any,
  ) -> None:
    """Send a response back to the platform.

    Args:
      original_msg: The original InterfaceMessage that triggered this response.
      response: The InterfaceResponse to send.
      raw_message: The raw platform message (for reply context).
    """
    ...

  # --- Message pipeline ---

  async def handle_platform_message(self, raw_message: Any) -> None:
    """Process a single platform message through the full pipeline.

    Pipeline steps:
      1. Convert inbound message
      2. Run on_message_received hooks (can veto)
      3. Get/create session
      4. Run on_before_respond hooks (can modify message)
      5. Call agent.arun()
      6. Build InterfaceResponse from RunOutput
      7. Run on_after_respond hooks (can modify response)
      8. Send response to platform
      9. Update session history

    Args:
      raw_message: The raw message from the platform.
    """
    message: Optional[InterfaceMessage] = None
    try:
      # 1. Convert inbound
      message = await self._convert_inbound(raw_message)
      if message is None:
        return

      # 2. Run on_message_received hooks
      for hook in self._hooks:
        if hasattr(hook, "on_message_received"):
          result = await hook.on_message_received(message)
          if result is False:
            return

      # 3. Get/create session
      session = self.session_manager.get_or_create(
        platform=message.platform,
        user_id=message.platform_user_id,
        chat_id=message.platform_chat_id,
      )

      # 4. Run on_before_respond hooks
      for hook in self._hooks:
        if hasattr(hook, "on_before_respond"):
          modified = await hook.on_before_respond(message, session)
          if modified is not None:
            message = modified

      # 5. Call agent
      assert self._request_semaphore is not None
      async with self._request_semaphore:
        run_output = await self._run_agent(message, session)

      # 6. Build response
      response = self._build_response(run_output)

      # 7. Run on_after_respond hooks
      for hook in self._hooks:
        if hasattr(hook, "on_after_respond"):
          modified = await hook.on_after_respond(message, response, session)
          if modified is not None:
            response = modified

      # 8. Send response
      await self._send_response(message, response, raw_message)

      # 9. Update session
      self._update_session(session, run_output)

    except Exception as e:
      await self._handle_error(e, message, raw_message)

  async def _run_agent(self, message: InterfaceMessage, session: InterfaceSession) -> RunOutput:
    """Run the agent with the given message and session context.

    Args:
      message: The normalized InterfaceMessage.
      session: The session for this conversation.

    Returns:
      RunOutput from the agent.
    """
    return await self.agent.arun(
      instruction=message.text or "",
      messages=session.messages,
      session_id=session.session_id,
      images=message.images,
      audio=message.audio,
      videos=message.videos,
      files=message.files,
    )

  def _build_response(self, run_output: RunOutput) -> InterfaceResponse:
    """Build an InterfaceResponse from the agent's RunOutput.

    Args:
      run_output: The agent's output.

    Returns:
      InterfaceResponse with content and media.
    """
    content: Optional[str] = None
    if run_output.content is not None:
      content = str(run_output.content)

    return InterfaceResponse(
      content=content,
      images=list(run_output.images) if run_output.images else None,
      videos=list(run_output.videos) if run_output.videos else None,
      audio=list(run_output.audio) if run_output.audio else None,
      files=list(run_output.files) if run_output.files else None,
    )

  def _update_session(self, session: InterfaceSession, run_output: RunOutput) -> None:
    """Update session with the agent's output.

    Args:
      session: The session to update.
      run_output: The agent's output.
    """
    session.last_run_output = run_output
    if run_output.messages:
      session.messages = list(run_output.messages)
    session.truncate_history(self.config.max_session_history)
    session.touch()

  async def _handle_error(
    self,
    error: Exception,
    message: Optional[InterfaceMessage],
    raw_message: Any,
  ) -> None:
    """Handle errors during message processing.

    Runs on_error hooks and attempts to send the configured error
    message back to the user.

    Args:
      error: The exception that occurred.
      message: The InterfaceMessage if available.
      raw_message: The raw platform message.
    """
    log_error(f"[{self.config.platform}] Error processing message: {error}")

    # Run on_error hooks
    for hook in self._hooks:
      if hasattr(hook, "on_error"):
        try:
          await hook.on_error(error, message)
        except Exception as hook_error:
          log_error(f"[{self.config.platform}] Hook on_error failed: {hook_error}")

    # Try to send error message to user
    if message is not None:
      try:
        error_response = InterfaceResponse(content=self.config.error_message)
        await self._send_response(message, error_response, raw_message)
      except Exception as send_error:
        log_error(f"[{self.config.platform}] Failed to send error message: {send_error}")
