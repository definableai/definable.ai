"""Text processing skill — regex, string manipulation, and text analysis.

Gives the agent precise text processing tools that don't rely on the
LLM's imprecise string handling (counting, regex, encoding, etc.).

Example:
    from definable.skills.builtin import TextProcessing

    agent = Agent(
        model=model,
        skills=[TextProcessing()],
    )
    output = agent.run("Count the words in this paragraph and find all email addresses.")
"""

import re
from typing import Optional

from definable.skills.base import Skill
from definable.tools.decorator import tool


@tool
def regex_search(text: str, pattern: str, flags: Optional[str] = None) -> str:
  """Search text using a regular expression pattern.

  Args:
    text: The text to search.
    pattern: A Python regular expression pattern.
    flags: Optional regex flags as a string: "i" (ignorecase),
        "m" (multiline), "s" (dotall). Combine like "im".

  Returns:
    All matches found, one per line, or "No matches found."
  """
  try:
    re_flags = 0
    for f in flags or "":
      if f == "i":
        re_flags |= re.IGNORECASE
      elif f == "m":
        re_flags |= re.MULTILINE
      elif f == "s":
        re_flags |= re.DOTALL

    matches = re.findall(pattern, text, re_flags)
    if not matches:
      return "No matches found."

    # Handle groups: findall returns tuples for grouped patterns
    results = []
    for match in matches:
      if isinstance(match, tuple):
        results.append(" | ".join(str(g) for g in match))
      else:
        results.append(str(match))

    count = len(results)
    result_text = "\n".join(results[:100])  # Cap at 100 matches
    if count > 100:
      result_text += f"\n... and {count - 100} more matches"
    return f"Found {count} match(es):\n{result_text}"

  except re.error as e:
    return f"Regex error: {e}"


@tool
def regex_replace(text: str, pattern: str, replacement: str, flags: Optional[str] = None) -> str:
  """Replace text matching a regular expression pattern.

  Args:
    text: The text to process.
    pattern: A Python regular expression pattern.
    replacement: The replacement string. Use \\1, \\2, etc. for group backreferences.
    flags: Optional regex flags: "i" (ignorecase), "m" (multiline), "s" (dotall).

  Returns:
    The text with all matches replaced.
  """
  try:
    re_flags = 0
    for f in flags or "":
      if f == "i":
        re_flags |= re.IGNORECASE
      elif f == "m":
        re_flags |= re.MULTILINE
      elif f == "s":
        re_flags |= re.DOTALL

    result = re.sub(pattern, replacement, text, flags=re_flags)
    return result
  except re.error as e:
    return f"Regex error: {e}"


@tool
def text_stats(text: str) -> str:
  """Get statistics about a text: character count, word count, line count, etc.

  Args:
    text: The text to analyze.

  Returns:
    A summary of text statistics.
  """
  chars = len(text)
  chars_no_spaces = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
  words = len(text.split())
  lines = text.count("\n") + (1 if text else 0)
  sentences = len(re.split(r"[.!?]+", text.strip())) - 1 if text.strip() else 0
  paragraphs = len([p for p in text.split("\n\n") if p.strip()])

  # Word frequency (top 10)
  word_list = re.findall(r"\b[a-zA-Z]+\b", text.lower())
  freq: dict = {}
  for w in word_list:
    freq[w] = freq.get(w, 0) + 1
  top_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]

  stats = [
    f"Characters: {chars:,} ({chars_no_spaces:,} without spaces)",
    f"Words: {words:,}",
    f"Lines: {lines:,}",
    f"Sentences: ~{sentences}",
    f"Paragraphs: {paragraphs}",
  ]

  if top_words:
    stats.append("Top words: " + ", ".join(f"'{w}' ({c})" for w, c in top_words))

  return "\n".join(stats)


@tool
def text_transform(text: str, operation: str) -> str:
  """Apply a transformation to text.

  Args:
    text: The text to transform.
    operation: The transformation to apply. Options:
        - "upper" — convert to uppercase
        - "lower" — convert to lowercase
        - "title" — convert to title case
        - "strip" — remove leading/trailing whitespace
        - "dedent" — remove common leading whitespace
        - "collapse_whitespace" — replace multiple spaces/newlines with single space
        - "sort_lines" — sort lines alphabetically
        - "unique_lines" — remove duplicate lines (preserves order)
        - "reverse_lines" — reverse the order of lines
        - "number_lines" — add line numbers

  Returns:
    The transformed text.
  """
  op = operation.strip().lower()

  if op == "upper":
    return text.upper()
  elif op == "lower":
    return text.lower()
  elif op == "title":
    return text.title()
  elif op == "strip":
    return text.strip()
  elif op == "dedent":
    import textwrap

    return textwrap.dedent(text)
  elif op == "collapse_whitespace":
    return re.sub(r"\s+", " ", text).strip()
  elif op == "sort_lines":
    lines = text.splitlines()
    return "\n".join(sorted(lines))
  elif op == "unique_lines":
    seen: set = set()
    result = []
    for line in text.splitlines():
      if line not in seen:
        seen.add(line)
        result.append(line)
    return "\n".join(result)
  elif op == "reverse_lines":
    return "\n".join(reversed(text.splitlines()))
  elif op == "number_lines":
    lines = text.splitlines()
    width = len(str(len(lines)))
    return "\n".join(f"{i + 1:>{width}} | {line}" for i, line in enumerate(lines))
  else:
    return (
      f"Unknown operation: '{operation}'. Options: upper, lower, title, strip, "
      "dedent, collapse_whitespace, sort_lines, unique_lines, reverse_lines, number_lines"
    )


@tool
def extract_patterns(text: str, pattern_type: str) -> str:
  """Extract common patterns from text (emails, URLs, numbers, etc.).

  Args:
    text: The text to search.
    pattern_type: What to extract. Options:
        - "emails" — email addresses
        - "urls" — URLs (http/https)
        - "numbers" — all numbers (integers and decimals)
        - "phone_numbers" — phone number patterns
        - "ip_addresses" — IPv4 addresses
        - "hashtags" — hashtag patterns (#word)
        - "mentions" — mention patterns (@user)

  Returns:
    A list of all extracted items, or "None found."
  """
  patterns = {
    "emails": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "urls": r"https?://[^\s<>\"']+",
    "numbers": r"-?\d+\.?\d*",
    "phone_numbers": r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}",
    "ip_addresses": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    "hashtags": r"#\w+",
    "mentions": r"@\w+",
  }

  pt = pattern_type.strip().lower()
  if pt not in patterns:
    return f"Unknown pattern type: '{pattern_type}'. Options: {', '.join(sorted(patterns.keys()))}"

  matches = re.findall(patterns[pt], text)
  if not matches:
    return "None found."

  # Deduplicate while preserving order
  seen: set = set()
  unique = []
  for m in matches:
    if m not in seen:
      seen.add(m)
      unique.append(m)

  return f"Found {len(unique)} unique match(es):\n" + "\n".join(unique)


class TextProcessing(Skill):
  """Skill for precise text processing and analysis.

  Adds tools for regex search/replace, text statistics, transformations,
  and pattern extraction. Prevents the common LLM problem of imprecise
  string operations (counting, pattern matching, etc.).

  Example:
    agent = Agent(model=model, skills=[TextProcessing()])
    output = agent.run("Find all email addresses in this text and count the words.")
  """

  name = "text_processing"
  instructions = (
    "You have access to text processing tools for precise string operations. "
    "Use regex_search and regex_replace for pattern matching — never try to "
    "count characters or match patterns yourself. Use text_stats for accurate "
    "word/character/line counts. Use extract_patterns for common patterns "
    "like emails, URLs, and phone numbers. Use text_transform for case "
    "conversion, deduplication, sorting, and other transformations."
  )

  def __init__(self):
    super().__init__()

  @property
  def tools(self) -> list:
    return [regex_search, regex_replace, text_stats, text_transform, extract_patterns]
