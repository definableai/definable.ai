"""Platform-agnostic message types for the interfaces module."""

from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Optional

from definable.media import Audio, File, Image, Video


@dataclass
class InterfaceMessage:
  """Inbound message from a platform, normalized to a common format.

  Bridges platform-specific messages (Telegram, Discord, etc.) to
  the agent's input format.

  Attributes:
    text: Message text content.
    platform: Platform name (e.g. "telegram").
    platform_user_id: User ID on the platform.
    platform_chat_id: Chat/conversation ID on the platform.
    platform_message_id: Message ID on the platform.
    username: Display name of the sender.
    images: Images attached to the message.
    audio: Audio files attached to the message.
    videos: Videos attached to the message.
    files: Files attached to the message.
    reply_to_message_id: ID of the message being replied to.
    metadata: Platform-specific data.
    created_at: Unix timestamp of when the message was created.
  """

  platform: str
  platform_user_id: str
  platform_chat_id: str
  platform_message_id: str

  text: Optional[str] = None
  username: Optional[str] = None

  images: Optional[List[Image]] = None
  audio: Optional[List[Audio]] = None
  videos: Optional[List[Video]] = None
  files: Optional[List[File]] = None

  reply_to_message_id: Optional[str] = None
  metadata: Dict[str, Any] = field(default_factory=dict)
  created_at: float = field(default_factory=time)


@dataclass
class InterfaceResponse:
  """Outbound response to send back to the platform.

  Built from an agent's RunOutput and sent via the platform API.

  Attributes:
    content: Text content of the response.
    images: Images to send back.
    videos: Videos to send back.
    audio: Audio files to send back.
    files: Files to send back.
    metadata: Platform-specific response data.
  """

  content: Optional[str] = None
  images: Optional[List[Image]] = None
  videos: Optional[List[Video]] = None
  audio: Optional[List[Audio]] = None
  files: Optional[List[File]] = None
  metadata: Dict[str, Any] = field(default_factory=dict)
