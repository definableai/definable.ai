"""Sticker description cache for Telegram Bot API."""

from collections import OrderedDict
from typing import Any, Dict, Optional


class StickerCache:
  """LRU cache for sticker set descriptions.

  Maps sticker ``file_unique_id`` → text description. Used to convert
  stickers into textual context for the agent.

  Args:
    max_size: Maximum number of entries to keep.
  """

  def __init__(self, max_size: int = 500) -> None:
    self._cache: OrderedDict[str, str] = OrderedDict()
    self._max_size = max_size

  def get(self, file_unique_id: str) -> Optional[str]:
    """Look up a cached sticker description.

    Args:
      file_unique_id: Telegram file_unique_id for the sticker.

    Returns:
      Cached description, or None if not found.
    """
    if file_unique_id in self._cache:
      self._cache.move_to_end(file_unique_id)
      return self._cache[file_unique_id]
    return None

  def put(self, file_unique_id: str, description: str) -> None:
    """Store a sticker description.

    Args:
      file_unique_id: Telegram file_unique_id for the sticker.
      description: Human-readable description of the sticker.
    """
    if file_unique_id in self._cache:
      self._cache.move_to_end(file_unique_id)
    self._cache[file_unique_id] = description
    while len(self._cache) > self._max_size:
      self._cache.popitem(last=False)

  def describe_sticker(self, sticker: Dict[str, Any]) -> str:
    """Build a text description for a sticker.

    Extracts emoji and sticker set name to create a readable
    description like ``[Sticker: emoji from 'SetName']``.

    Args:
      sticker: Telegram sticker dict from the update.

    Returns:
      Human-readable description string.
    """
    file_unique_id = sticker.get("file_unique_id", "")
    cached = self.get(file_unique_id)
    if cached:
      return cached

    emoji = sticker.get("emoji", "")
    set_name = sticker.get("set_name", "")

    if emoji and set_name:
      desc = f"[Sticker: {emoji} from '{set_name}']"
    elif emoji:
      desc = f"[Sticker: {emoji}]"
    elif set_name:
      desc = f"[Sticker from '{set_name}']"
    else:
      desc = "[Sticker]"

    if file_unique_id:
      self.put(file_unique_id, desc)
    return desc

  def __len__(self) -> int:
    return len(self._cache)
