"""Session management for interfaces."""

import threading
from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from definable.models.message import Message
from definable.run.agent import RunOutput


@dataclass
class InterfaceSession:
  """Represents a conversation session between a platform user and the agent.

  Maps to the agent's session_id parameter and maintains conversation
  history for multi-turn interactions.

  Attributes:
    session_id: Unique session identifier, used as the agent's session_id.
    platform: Platform name (e.g. "telegram").
    platform_user_id: User ID on the platform.
    platform_chat_id: Chat/conversation ID on the platform.
    messages: Conversation history as agent Message objects.
    session_state: Arbitrary state dict carried across runs.
    last_run_output: The most recent RunOutput from the agent.
    created_at: Unix timestamp of when the session was created.
    last_activity_at: Unix timestamp of last activity.
  """

  session_id: str = field(default_factory=lambda: str(uuid4()))
  platform: str = ""
  platform_user_id: str = ""
  platform_chat_id: str = ""

  messages: Optional[List[Message]] = None
  session_state: Dict[str, Any] = field(default_factory=dict)
  last_run_output: Optional[RunOutput] = None

  created_at: float = field(default_factory=time)
  last_activity_at: float = field(default_factory=time)

  def touch(self) -> None:
    """Update the last activity timestamp."""
    self.last_activity_at = time()

  def truncate_history(self, max_messages: int) -> None:
    """Truncate message history to the most recent max_messages."""
    if self.messages and len(self.messages) > max_messages:
      self.messages = self.messages[-max_messages:]


class SessionManager:
  """Thread-safe session manager for interface sessions.

  Keys sessions by ``platform:user_id:chat_id`` and automatically
  expires sessions past the configured TTL on access.

  Args:
    session_ttl_seconds: Time-to-live for sessions in seconds.
  """

  def __init__(self, session_ttl_seconds: int = 3600) -> None:
    self._sessions: Dict[str, InterfaceSession] = {}
    self._lock = threading.Lock()
    self._ttl = session_ttl_seconds

  def _make_key(self, platform: str, user_id: str, chat_id: str) -> str:
    return f"{platform}:{user_id}:{chat_id}"

  def _is_expired(self, session: InterfaceSession) -> bool:
    return (time() - session.last_activity_at) > self._ttl

  def get_or_create(self, platform: str, user_id: str, chat_id: str) -> InterfaceSession:
    """Get an existing session or create a new one.

    Expired sessions are removed and a new session is created.

    Args:
      platform: Platform name.
      user_id: User ID on the platform.
      chat_id: Chat/conversation ID on the platform.

    Returns:
      The session for the given platform/user/chat combination.
    """
    key = self._make_key(platform, user_id, chat_id)
    with self._lock:
      session = self._sessions.get(key)
      if session is not None and not self._is_expired(session):
        session.touch()
        return session

      # Create new session (or replace expired one)
      session = InterfaceSession(
        platform=platform,
        platform_user_id=user_id,
        platform_chat_id=chat_id,
      )
      self._sessions[key] = session
      return session

  def get(self, platform: str, user_id: str, chat_id: str) -> Optional[InterfaceSession]:
    """Get an existing session, or None if not found or expired.

    Args:
      platform: Platform name.
      user_id: User ID on the platform.
      chat_id: Chat/conversation ID on the platform.

    Returns:
      The session, or None.
    """
    key = self._make_key(platform, user_id, chat_id)
    with self._lock:
      session = self._sessions.get(key)
      if session is None:
        return None
      if self._is_expired(session):
        del self._sessions[key]
        return None
      session.touch()
      return session

  def remove(self, platform: str, user_id: str, chat_id: str) -> bool:
    """Remove a session.

    Args:
      platform: Platform name.
      user_id: User ID on the platform.
      chat_id: Chat/conversation ID on the platform.

    Returns:
      True if a session was removed, False otherwise.
    """
    key = self._make_key(platform, user_id, chat_id)
    with self._lock:
      if key in self._sessions:
        del self._sessions[key]
        return True
      return False

  def cleanup_expired(self) -> int:
    """Remove all expired sessions.

    Returns:
      Number of sessions removed.
    """
    with self._lock:
      expired_keys = [k for k, s in self._sessions.items() if self._is_expired(s)]
      for key in expired_keys:
        del self._sessions[key]
      return len(expired_keys)

  @property
  def active_session_count(self) -> int:
    """Return the number of active (non-expired) sessions."""
    with self._lock:
      return sum(1 for s in self._sessions.values() if not self._is_expired(s))
