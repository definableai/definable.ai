"""Markdown→Telegram HTML conversion and tag-aware chunking."""

import re
from typing import List, Tuple


# Pre-compiled patterns for performance
_CODE_BLOCK_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_STRIKETHROUGH_RE = re.compile(r"~~(.+?)~~")
_SPOILER_RE = re.compile(r"\|\|(.+?)\|\|")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BLOCKQUOTE_RE = re.compile(r"^> ?(.+)$", re.MULTILINE)

# Characters that must be escaped in non-formatted text
_HTML_ESCAPE = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}

# Tags we track for split_html
_OPEN_TAG_RE = re.compile(r"<(b|i|s|u|code|pre|blockquote|tg-spoiler|a)(?:\s[^>]*)?>")
_CLOSE_TAG_RE = re.compile(r"</(b|i|s|u|code|pre|blockquote|tg-spoiler|a)>")


def _html_escape(text: str) -> str:
  """Escape HTML special characters."""
  for char, escape in _HTML_ESCAPE.items():
    text = text.replace(char, escape)
  return text


def markdown_to_telegram_html(text: str) -> str:
  """Convert common Markdown to Telegram-compatible HTML.

  Supported conversions:
    - ``**bold**`` → ``<b>bold</b>``
    - ``*italic*`` → ``<i>italic</i>``
    - `` `code` `` → ``<code>code</code>``
    - Fenced code blocks → ``<pre><code class="language-X">...</code></pre>``
    - ``[text](url)`` → ``<a href="url">text</a>``
    - ``> quote`` → ``<blockquote>quote</blockquote>``
    - ``~~strike~~`` → ``<s>strike</s>``
    - ``||spoiler||`` → ``<tg-spoiler>spoiler</tg-spoiler>``

  Text outside formatting is HTML-escaped. Content inside code blocks
  is preserved verbatim (no formatting applied).

  Args:
    text: Markdown text to convert.

  Returns:
    Telegram-compatible HTML string.
  """
  if not text:
    return text

  # Pass 1: Extract code blocks and inline code to protect them
  code_blocks: List[Tuple[str, str]] = []  # (placeholder, replacement_html)
  counter = [0]

  def _extract_code_block(m: re.Match[str]) -> str:
    lang = m.group(1)
    code = m.group(2)
    key = f"\x00CB{counter[0]}\x00"
    counter[0] += 1
    escaped_code = _html_escape(code)
    if lang:
      html = f'<pre><code class="language-{lang}">{escaped_code}</code></pre>'
    else:
      html = f"<pre><code>{escaped_code}</code></pre>"
    code_blocks.append((key, html))
    return key

  def _extract_inline_code(m: re.Match[str]) -> str:
    code = m.group(1)
    key = f"\x00IC{counter[0]}\x00"
    counter[0] += 1
    html = f"<code>{_html_escape(code)}</code>"
    code_blocks.append((key, html))
    return key

  result = _CODE_BLOCK_RE.sub(_extract_code_block, text)
  result = _INLINE_CODE_RE.sub(_extract_inline_code, result)

  # Pass 2: HTML-escape the remaining text (outside code)
  result = _html_escape(result)

  # Pass 3: Apply formatting conversions
  # Links first (before bold/italic to avoid conflicts)
  result = _LINK_RE.sub(r'<a href="\2">\1</a>', result)
  # Bold before italic
  result = _BOLD_RE.sub(r"<b>\1</b>", result)
  result = _ITALIC_RE.sub(r"<i>\1</i>", result)
  result = _STRIKETHROUGH_RE.sub(r"<s>\1</s>", result)
  result = _SPOILER_RE.sub(r"<tg-spoiler>\1</tg-spoiler>", result)

  # Blockquotes: merge consecutive > lines into a single blockquote
  lines = result.split("\n")
  merged_lines: List[str] = []
  in_blockquote = False
  bq_lines: List[str] = []

  for line in lines:
    # Check for &gt; (escaped >) at start of line
    stripped = line.lstrip()
    if stripped.startswith("&gt; ") or stripped == "&gt;":
      content = stripped[5:] if stripped.startswith("&gt; ") else ""
      bq_lines.append(content)
      in_blockquote = True
    else:
      if in_blockquote:
        merged_lines.append(f"<blockquote>{chr(10).join(bq_lines)}</blockquote>")
        bq_lines = []
        in_blockquote = False
      merged_lines.append(line)

  if in_blockquote:
    merged_lines.append(f"<blockquote>{chr(10).join(bq_lines)}</blockquote>")

  result = "\n".join(merged_lines)

  # Pass 4: Restore code blocks
  for key, html in code_blocks:
    result = result.replace(key, html)

  return result


def split_html(text: str, max_length: int = 4096) -> List[str]:
  """Split HTML text into chunks that respect tag boundaries.

  Uses a tag-stack tracker to ensure all open tags are properly
  closed before a split point and reopened after. Tries to split
  at paragraph (``\\n\\n``) or newline boundaries.

  Falls back to plain-text splitting if the text contains no HTML tags.

  Args:
    text: HTML text to split.
    max_length: Maximum length per chunk (Telegram limit: 4096).

  Returns:
    List of HTML chunks, each within max_length.
  """
  if len(text) <= max_length:
    return [text]

  # Quick check: if no HTML tags at all, fall back to plain split
  if "<" not in text:
    return _split_plain(text, max_length)

  chunks: List[str] = []
  remaining = text

  while remaining:
    if len(remaining) <= max_length:
      chunks.append(remaining)
      break

    # Find best split point within max_length
    split_pos = _find_html_split_point(remaining, max_length)

    # Get the chunk and determine open tags
    chunk = remaining[:split_pos]
    open_tags = _get_open_tags(chunk)

    # Close any open tags at the end of this chunk
    close_suffix = "".join(f"</{tag}>" for tag in reversed(open_tags))
    chunk_with_close = chunk + close_suffix

    # If closing tags push us over, back up the split point
    if len(chunk_with_close) > max_length:
      # Recalculate with reduced budget
      overhead = len(close_suffix)
      reopen_prefix = "".join(f"<{tag}>" for tag in open_tags)
      overhead += len(reopen_prefix)
      split_pos = _find_html_split_point(remaining, max_length - overhead)
      chunk = remaining[:split_pos]
      open_tags = _get_open_tags(chunk)
      close_suffix = "".join(f"</{tag}>" for tag in reversed(open_tags))
      chunk_with_close = chunk + close_suffix

    chunks.append(chunk_with_close)

    # Reopen tags for the next chunk
    reopen_prefix = "".join(f"<{tag}>" for tag in open_tags)
    remaining = reopen_prefix + remaining[split_pos:].lstrip("\n")

  return chunks


def _find_html_split_point(text: str, max_length: int) -> int:
  """Find the best split point that doesn't break HTML tags."""
  # Never split inside an HTML tag
  safe_max = max_length

  # Check if we're inside a tag at max_length
  last_open = text.rfind("<", 0, safe_max)
  last_close = text.rfind(">", 0, safe_max)
  if last_open > last_close:
    # We're inside a tag — move split before the tag
    safe_max = last_open

  # Try paragraph boundary
  pos = text.rfind("\n\n", 0, safe_max)
  if pos > 0:
    return pos

  # Try newline
  pos = text.rfind("\n", 0, safe_max)
  if pos > 0:
    return pos

  # Try space
  pos = text.rfind(" ", 0, safe_max)
  if pos > 0:
    return pos

  # Hard split (but not inside a tag)
  return safe_max


def _get_open_tags(html: str) -> List[str]:
  """Return the stack of tags that are still open at the end of the HTML fragment."""
  stack: List[str] = []
  for m in _OPEN_TAG_RE.finditer(html):
    stack.append(m.group(1))
  for m in _CLOSE_TAG_RE.finditer(html):
    tag = m.group(1)
    # Pop the most recent matching open tag
    for i in range(len(stack) - 1, -1, -1):
      if stack[i] == tag:
        stack.pop(i)
        break
  return stack


def _split_plain(text: str, max_length: int) -> List[str]:
  """Split plain text at natural boundaries."""
  if len(text) <= max_length:
    return [text]

  chunks: List[str] = []
  remaining = text
  while remaining:
    if len(remaining) <= max_length:
      chunks.append(remaining)
      break

    pos = remaining.rfind("\n", 0, max_length)
    if pos == -1:
      pos = remaining.rfind(" ", 0, max_length)
    if pos == -1:
      pos = max_length

    chunks.append(remaining[:pos])
    remaining = remaining[pos:].lstrip("\n")

  return chunks
