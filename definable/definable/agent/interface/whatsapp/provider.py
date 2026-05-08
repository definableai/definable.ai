"""WhatsApp provider protocol — transport abstraction for Twilio and Baileys."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

from definable.media import Audio, File, Image, Video


# --------------------------------------------------------------------------- #
# Data types                                                                   #
# --------------------------------------------------------------------------- #


@dataclass
class WhatsAppContact:
  """A resolved WhatsApp contact."""

  phone: str
  jid: str
  push_name: str = ""
  is_group: bool = False


@dataclass
class InboundMessage:
  """Raw inbound message from a WhatsApp provider."""

  id: str
  from_phone: str
  from_jid: str
  chat_jid: str
  body: str = ""
  push_name: str = ""
  is_group: bool = False
  is_from_me: bool = False
  timestamp: float = 0.0
  # Reply context
  reply_to_id: Optional[str] = None
  reply_to_body: Optional[str] = None
  reply_to_sender: Optional[str] = None
  # Group context
  group_subject: Optional[str] = None
  group_participants: Optional[List[str]] = None
  mentioned_jids: Optional[List[str]] = None
  was_mentioned: bool = False
  # Media
  images: Optional[List[Image]] = None
  audio: Optional[List[Audio]] = None
  videos: Optional[List[Video]] = None
  files: Optional[List[File]] = None
  # Location
  latitude: Optional[float] = None
  longitude: Optional[float] = None
  # Raw provider payload
  raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutboundMessage:
  """Message to send via a WhatsApp provider."""

  to: str
  body: str = ""
  image: Optional[Image] = None
  audio: Optional[Audio] = None
  video: Optional[Video] = None
  file: Optional[File] = None
  reply_to_id: Optional[str] = None


@dataclass
class PollMessage:
  """Poll to send via a WhatsApp provider."""

  to: str
  question: str
  options: List[str]
  allows_multiple: bool = False


@dataclass
class ReactionMessage:
  """Reaction to send on a message."""

  chat_jid: str
  message_id: str
  emoji: str
  from_me: bool = False
  participant: Optional[str] = None


@dataclass
class SendResult:
  """Result of a send operation."""

  success: bool
  message_id: Optional[str] = None
  error: Optional[str] = None


@dataclass
class ConnectionStatus:
  """Provider connection status snapshot."""

  connected: bool = False
  running: bool = False
  reconnect_attempts: int = 0
  last_connected_at: Optional[float] = None
  last_disconnect_at: Optional[float] = None
  last_message_at: Optional[float] = None
  last_error: Optional[str] = None
  linked: bool = False
  self_phone: Optional[str] = None
  self_jid: Optional[str] = None


@dataclass
class QRLoginResult:
  """Result of a QR login attempt."""

  qr_data: Optional[str] = None
  connected: bool = False
  message: str = ""


# --------------------------------------------------------------------------- #
# Callback type                                                                #
# --------------------------------------------------------------------------- #

MessageCallback = Callable[[InboundMessage], Coroutine[Any, Any, None]]


# --------------------------------------------------------------------------- #
# Provider ABC                                                                 #
# --------------------------------------------------------------------------- #


class WhatsAppProvider(ABC):
  """Abstract transport provider for WhatsApp.

  Providers handle the connection to WhatsApp and expose a uniform
  interface for sending/receiving messages. Two implementations:

  - **TwilioProvider**: Managed webhook + REST API. Paid. No QR login.
  - **BaileysProvider**: Self-hosted via Node.js sidecar. Free. Full protocol.

  Lifecycle::

    provider = BaileysProvider(auth_dir="./wa-auth")
    await provider.connect(on_message=my_handler)
    await provider.send_text("+15551234567", "Hello!")
    status = await provider.status()
    await provider.disconnect()
  """

  @abstractmethod
  async def connect(self, on_message: MessageCallback) -> None:
    """Connect to WhatsApp and start receiving messages.

    Args:
      on_message: Async callback for each inbound message.
        The provider MUST NOT block on this callback.
    """
    ...

  @abstractmethod
  async def disconnect(self) -> None:
    """Disconnect from WhatsApp. Must be idempotent."""
    ...

  @abstractmethod
  async def send_text(self, to: str, body: str) -> SendResult:
    """Send a text message.

    Args:
      to: Target JID or E.164 phone number.
      body: Message text.
    """
    ...

  @abstractmethod
  async def send_media(self, msg: OutboundMessage) -> SendResult:
    """Send a media message (image, audio, video, or file).

    Args:
      msg: OutboundMessage with exactly one media field set.
    """
    ...

  @abstractmethod
  async def send_poll(self, poll: PollMessage) -> SendResult:
    """Send a poll. Returns ``success=False`` if unsupported."""
    ...

  @abstractmethod
  async def send_reaction(self, reaction: ReactionMessage) -> SendResult:
    """Send a reaction. Returns ``success=False`` if unsupported."""
    ...

  @abstractmethod
  async def send_composing(self, to: str) -> None:
    """Send a typing indicator."""
    ...

  @abstractmethod
  async def status(self) -> ConnectionStatus:
    """Get current connection status."""
    ...

  # --- Optional capabilities (default no-op) ---

  async def login_qr_start(self, force: bool = False) -> QRLoginResult:
    """Start QR-based login. Only supported by Baileys."""
    return QRLoginResult(message="QR login not supported by this provider.")

  async def login_qr_wait(self, timeout_ms: int = 60_000) -> QRLoginResult:
    """Wait for QR scan to complete."""
    return QRLoginResult(message="QR login not supported by this provider.")

  async def logout(self) -> bool:
    """Logout and clear credentials. Returns True if cleared."""
    return False

  # --- Capability flags ---

  @property
  def supports_polls(self) -> bool:
    return False

  @property
  def supports_reactions(self) -> bool:
    return False

  @property
  def supports_groups(self) -> bool:
    return False

  @property
  def supports_media(self) -> bool:
    return False

  @property
  def supports_qr_login(self) -> bool:
    return False

  @property
  def provider_name(self) -> str:
    return "unknown"
