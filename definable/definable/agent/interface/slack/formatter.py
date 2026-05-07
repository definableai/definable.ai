"""Markdown to Slack mrkdwn converter, text utilities, and Block Kit builders."""

import re
from typing import Any, Dict, List, Optional


def markdown_to_mrkdwn(text: str) -> str:
  """Convert standard Markdown to Slack mrkdwn format.

  Handles the key differences between Markdown and Slack's mrkdwn:
    - ``**bold**`` → ``*bold*``
    - ``*italic*`` / ``_italic_`` → ``_italic_``
    - ``~~strike~~`` → ``~strike~``
    - ``[text](url)`` → ``<url|text>``
    - ``# Heading`` → ``*Heading*`` (bold, since mrkdwn has no headings)
    - Fenced code blocks: strip language hints

  Does NOT touch content inside inline code or code blocks.

  Args:
    text: Standard Markdown text.

  Returns:
    Slack mrkdwn formatted text.
  """
  # Protect code blocks and inline code from transformation
  code_blocks: List[str] = []
  inline_codes: List[str] = []

  def _stash_code_block(m: re.Match[str]) -> str:
    code_blocks.append(m.group(0))
    return f"\x00CODEBLOCK{len(code_blocks) - 1}\x00"

  def _stash_inline_code(m: re.Match[str]) -> str:
    inline_codes.append(m.group(0))
    return f"\x00INLINE{len(inline_codes) - 1}\x00"

  # Stash fenced code blocks (``` ... ```)
  text = re.sub(r"```[\s\S]*?```", _stash_code_block, text)
  # Stash inline code (` ... `)
  text = re.sub(r"`[^`]+`", _stash_inline_code, text)

  # Links: [text](url) → <url|text>
  text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", text)

  # Bold: **text** → *text*
  # Stash bold results to prevent italic regex from reconverting them
  bold_stash: List[str] = []

  def _stash_bold(m: re.Match[str]) -> str:
    bold_stash.append(f"*{m.group(1)}*")
    return f"\x00BOLD{len(bold_stash) - 1}\x00"

  text = re.sub(r"\*\*(.+?)\*\*", _stash_bold, text)

  # Italic: *text* → _text_ (single asterisks only, bold already stashed)
  text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"_\1_", text)

  # Restore bold
  for i, bold in enumerate(bold_stash):
    text = text.replace(f"\x00BOLD{i}\x00", bold)

  # Strikethrough: ~~text~~ → ~text~
  text = re.sub(r"~~(.+?)~~", r"~\1~", text)

  # Headings: # Heading → *Heading* (bold line)
  text = re.sub(r"^#{1,6}\s+(.+)$", r"*\1*", text, flags=re.MULTILINE)

  # Restore inline code
  for i, code in enumerate(inline_codes):
    text = text.replace(f"\x00INLINE{i}\x00", code)

  # Restore code blocks (strip language hint from fenced blocks)
  for i, block in enumerate(code_blocks):
    # Remove language identifier: ```python → ```
    cleaned = re.sub(r"^```\w+", "```", block)
    text = text.replace(f"\x00CODEBLOCK{i}\x00", cleaned)

  return text


def split_text(text: str, max_length: int) -> List[str]:
  """Split text into chunks respecting max_length.

  Tries to split at newlines, then at spaces, falling back to
  hard splits if necessary.

  Args:
    text: Text to split.
    max_length: Maximum length per chunk.

  Returns:
    List of text chunks.
  """
  if len(text) <= max_length:
    return [text]

  chunks: List[str] = []
  remaining = text
  while remaining:
    if len(remaining) <= max_length:
      chunks.append(remaining)
      break

    # Try to split at a newline
    split_pos = remaining.rfind("\n", 0, max_length)
    if split_pos == -1:
      # Try to split at a space
      split_pos = remaining.rfind(" ", 0, max_length)
    if split_pos == -1:
      # Hard split
      split_pos = max_length

    chunks.append(remaining[:split_pos])
    remaining = remaining[split_pos:].lstrip("\n ")  # Strip newlines and spaces at split point

  return chunks


# ============================================================================
# Block Kit builders — pure dict constructors, no SDK dependency
# ============================================================================


def plain_text(text: str) -> Dict[str, Any]:
  """Create a plain_text text object."""
  return {"type": "plain_text", "text": text}


def mrkdwn_text(text: str) -> Dict[str, Any]:
  """Create a mrkdwn text object."""
  return {"type": "mrkdwn", "text": text}


def divider_block() -> Dict[str, Any]:
  """Create a divider block."""
  return {"type": "divider"}


def header_block(text: str) -> Dict[str, Any]:
  """Create a header block.

  Args:
    text: Header text (max 150 chars, rendered as plain_text).
  """
  return {"type": "header", "text": plain_text(text)}


def section_block(text: str, *, accessory: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
  """Create a section block with mrkdwn text and optional accessory.

  Args:
    text: Section text in mrkdwn format.
    accessory: Optional Block Kit element (button, image, select, etc.).
  """
  block: Dict[str, Any] = {"type": "section", "text": mrkdwn_text(text)}
  if accessory:
    block["accessory"] = accessory
  return block


def actions_block(elements: List[Dict[str, Any]], *, block_id: Optional[str] = None) -> Dict[str, Any]:
  """Create an actions block containing interactive elements.

  Args:
    elements: List of interactive elements (buttons, selects, etc.).
    block_id: Optional block identifier for referencing in callbacks.
  """
  block: Dict[str, Any] = {"type": "actions", "elements": elements}
  if block_id:
    block["block_id"] = block_id
  return block


def context_block(elements: List[Dict[str, Any]]) -> Dict[str, Any]:
  """Create a context block for secondary information.

  Args:
    elements: List of text objects (plain_text or mrkdwn) and image elements.
  """
  return {"type": "context", "elements": elements}


def image_block(image_url: str, alt_text: str, *, title: Optional[str] = None) -> Dict[str, Any]:
  """Create an image block.

  Args:
    image_url: URL of the image.
    alt_text: Accessibility text describing the image.
    title: Optional title displayed above the image.
  """
  block: Dict[str, Any] = {"type": "image", "image_url": image_url, "alt_text": alt_text}
  if title:
    block["title"] = plain_text(title)
  return block


def input_block(
  label: str,
  element: Dict[str, Any],
  *,
  block_id: Optional[str] = None,
  optional: bool = False,
) -> Dict[str, Any]:
  """Create an input block for forms and modals.

  Args:
    label: Label text displayed above the input.
    element: Input element (plain_text_input, static_select, etc.).
    block_id: Optional block identifier for referencing submitted values.
    optional: Whether this input is optional (default False).
  """
  block: Dict[str, Any] = {"type": "input", "label": plain_text(label), "element": element}
  if block_id:
    block["block_id"] = block_id
  if optional:
    block["optional"] = True
  return block


# ============================================================================
# Block Kit elements — interactive components within blocks
# ============================================================================


def button_element(
  text: str,
  action_id: str,
  *,
  value: Optional[str] = None,
  style: Optional[str] = None,
) -> Dict[str, Any]:
  """Create a button element.

  Args:
    text: Button label.
    action_id: Unique identifier for this action (received in callbacks).
    value: Optional value sent with the action payload.
    style: Optional ``"primary"`` (green) or ``"danger"`` (red).
  """
  btn: Dict[str, Any] = {"type": "button", "text": plain_text(text), "action_id": action_id}
  if value is not None:
    btn["value"] = value
  if style:
    btn["style"] = style
  return btn


def static_select_element(
  placeholder: str,
  action_id: str,
  options: List[Dict[str, Any]],
) -> Dict[str, Any]:
  """Create a static select menu element.

  Args:
    placeholder: Placeholder text shown when nothing is selected.
    action_id: Unique identifier for this action.
    options: List of option objects (use ``option_object()``).
  """
  return {
    "type": "static_select",
    "placeholder": plain_text(placeholder),
    "action_id": action_id,
    "options": options,
  }


def plain_text_input(
  action_id: str,
  *,
  placeholder: Optional[str] = None,
  multiline: bool = False,
  initial_value: Optional[str] = None,
) -> Dict[str, Any]:
  """Create a plain text input element for modals.

  Args:
    action_id: Unique identifier for this input.
    placeholder: Optional placeholder text.
    multiline: Whether to render as a textarea (default False).
    initial_value: Optional pre-filled value.
  """
  elem: Dict[str, Any] = {"type": "plain_text_input", "action_id": action_id}
  if placeholder:
    elem["placeholder"] = plain_text(placeholder)
  if multiline:
    elem["multiline"] = True
  if initial_value is not None:
    elem["initial_value"] = initial_value
  return elem


def option_object(text: str, value: str) -> Dict[str, Any]:
  """Create an option object for select menus.

  Args:
    text: Display text for the option.
    value: Value sent in the action payload when selected.
  """
  return {"text": plain_text(text), "value": value}


# ============================================================================
# Modal view builder
# ============================================================================


def modal_view(
  title: str,
  blocks: List[Dict[str, Any]],
  *,
  callback_id: Optional[str] = None,
  submit: Optional[str] = None,
  close: Optional[str] = None,
) -> Dict[str, Any]:
  """Build a modal view definition for ``views.open``.

  Args:
    title: Modal title (max 24 chars).
    blocks: List of Block Kit blocks forming the modal body.
    callback_id: Identifier for the view submission callback
      (use with ``interface.on_view()``).
    submit: Label for the submit button (default: no submit button).
    close: Label for the close button (default: "Cancel").
  """
  view: Dict[str, Any] = {"type": "modal", "title": plain_text(title), "blocks": blocks}
  if callback_id:
    view["callback_id"] = callback_id
  if submit:
    view["submit"] = plain_text(submit)
  if close:
    view["close"] = plain_text(close)
  return view


# ============================================================================
# App Home tab view builder
# ============================================================================


def home_tab_view(
  blocks: List[Dict[str, Any]],
  *,
  external_id: Optional[str] = None,
) -> Dict[str, Any]:
  """Build an App Home tab view definition for ``views.publish``.

  Args:
    blocks: List of Block Kit blocks forming the home tab body.
    external_id: Optional external ID for caching/deduplication.
  """
  view: Dict[str, Any] = {"type": "home", "blocks": blocks}
  if external_id:
    view["external_id"] = external_id
  return view
