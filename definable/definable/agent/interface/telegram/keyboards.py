"""Inline keyboard builder for Telegram Bot API."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class InlineButton:
  """A single inline keyboard button.

  Args:
    text: Button label shown to the user.
    callback_data: Data sent when button is pressed (max 64 bytes).
    url: URL to open when button is pressed (mutually exclusive with callback_data).
  """

  text: str
  callback_data: Optional[str] = None
  url: Optional[str] = None

  def __post_init__(self) -> None:
    if not self.callback_data and not self.url:
      raise ValueError("InlineButton requires either callback_data or url")
    if self.callback_data and self.url:
      raise ValueError("InlineButton cannot have both callback_data and url")
    if self.callback_data and len(self.callback_data.encode("utf-8")) > 64:
      raise ValueError(f"callback_data exceeds 64-byte limit: {len(self.callback_data.encode('utf-8'))} bytes")

  def to_dict(self) -> Dict[str, str]:
    """Convert to Telegram API dict."""
    result: Dict[str, str] = {"text": self.text}
    if self.callback_data:
      result["callback_data"] = self.callback_data
    if self.url:
      result["url"] = self.url
    return result


@dataclass
class InlineKeyboard:
  """Builder for Telegram inline keyboards.

  Example::

      kb = InlineKeyboard()
      kb.row(
        InlineButton("Yes", callback_data="confirm"),
        InlineButton("No", callback_data="cancel"),
      )
      kb.button("Help", url="https://example.com/help")
      markup = kb.to_dict()  # Pass as reply_markup to sendMessage
  """

  _rows: List[List[InlineButton]] = field(default_factory=list, repr=False)

  def row(self, *buttons: InlineButton) -> "InlineKeyboard":
    """Add a row of buttons.

    Args:
      *buttons: One or more InlineButton instances.

    Returns:
      Self for method chaining.
    """
    self._rows.append(list(buttons))
    return self

  def button(self, text: str, callback_data: Optional[str] = None, url: Optional[str] = None) -> "InlineKeyboard":
    """Add a single button as its own row.

    Args:
      text: Button label.
      callback_data: Data sent on press.
      url: URL to open.

    Returns:
      Self for method chaining.
    """
    self._rows.append([InlineButton(text=text, callback_data=callback_data, url=url)])
    return self

  def to_dict(self) -> Dict[str, Any]:
    """Convert to Telegram reply_markup dict.

    Returns:
      Dict suitable for the ``reply_markup`` parameter.
    """
    return {"inline_keyboard": [[btn.to_dict() for btn in row] for row in self._rows]}
