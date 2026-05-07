"""Markdown to WhatsApp formatting conversion."""

from __future__ import annotations

import re


def markdown_to_whatsapp(text: str) -> str:
  """Convert common Markdown formatting to WhatsApp-compatible formatting.

  Conversions:

  - ``**bold**`` or ``__bold__`` → ``*bold*``
  - ``*italic*`` or ``_italic_`` → ``_italic_``
  - ``~~strikethrough~~`` → ``~strikethrough~``
  - ````` `inline code` ````` → ````` ```inline code``` `````
  - Fenced code blocks (````` ```lang ... ``` `````) → ````` ``` ... ``` `````
  - ``# Heading`` → ``*Heading*`` (bolded)
  - ``[text](url)`` → ``text (url)``
  - ``![alt](url)`` → ``alt: url``
  - Bullet ``- item`` / ``* item`` → preserved as-is (WhatsApp renders these)
  - Numbered lists → preserved as-is

  Note:
    WhatsApp formatting is limited. This function does best-effort
    conversion. Nested formatting may not render perfectly.

  Args:
    text: Markdown-formatted text.

  Returns:
    WhatsApp-compatible text.
  """
  if not text:
    return text

  # Preserve fenced code blocks first (replace with placeholders)
  code_blocks: list[str] = []

  def _save_code_block(m: re.Match[str]) -> str:
    code_blocks.append(m.group(2))
    return f"\x00CODEBLOCK{len(code_blocks) - 1}\x00"

  result = re.sub(r"```(\w*)\n(.*?)```", _save_code_block, text, flags=re.DOTALL)

  # Inline code: `code` → ```code```
  result = re.sub(r"`([^`\n]+)`", r"```\1```", result)

  # Italic FIRST: single *text* → _text_ (must run before bold conversion)
  # Only match single * not preceded/followed by another *
  result = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"_\1_", result)

  # Bold: **text** or __text__ → *text* (WhatsApp bold)
  result = re.sub(r"\*\*(.+?)\*\*", r"*\1*", result)
  result = re.sub(r"__(.+?)__", r"*\1*", result)

  # Strikethrough: ~~text~~ → ~text~
  result = re.sub(r"~~(.+?)~~", r"~\1~", result)

  # Headings: # Heading → *Heading*
  result = re.sub(r"^#{1,6}\s+(.+)$", r"*\1*", result, flags=re.MULTILINE)

  # Images: ![alt](url) → alt: url
  result = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"\1: \2", result)

  # Links: [text](url) → text (url)
  result = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", result)

  # Horizontal rules: --- or *** or ___ → ───
  result = re.sub(r"^[-*_]{3,}$", "───", result, flags=re.MULTILINE)

  # Blockquotes: > text → > text (WhatsApp doesn't have native quotes, keep as-is)

  # Restore code blocks
  for i, block in enumerate(code_blocks):
    result = result.replace(f"\x00CODEBLOCK{i}\x00", f"```\n{block}```")

  return result
