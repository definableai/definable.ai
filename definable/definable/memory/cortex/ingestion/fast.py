"""Fast-path ingestion processor — zero LLM calls, runs in <10ms.

Extracts entities via regex, timestamps via dateutil, and builds binary signatures.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
  from definable.memory.cortex.index.signature import SignatureBuilder

# Regex patterns for entity extraction
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_URL_RE = re.compile(r"https?://[^\s<>\"']+")
_MENTION_RE = re.compile(r"@[\w.-]+")
_HASHTAG_RE = re.compile(r"#[\w]+")
_CAMEL_CASE_RE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b")
_ALL_CAPS_RE = re.compile(r"\b[A-Z]{2,}\b")


class FastPathProcessor:
  """Extracts structured data from text without LLM calls.

  Performs:
    - Entity extraction (emails, URLs, mentions, CamelCase names)
    - Timestamp detection
    - Binary signature generation
  """

  def __init__(self, signature_builder: Optional["SignatureBuilder"] = None):
    self._sig_builder = signature_builder

  def process(self, text: str) -> FastPathResult:
    """Process text and return extracted data."""
    entities = self._extract_entities(text)
    persons = self._extract_persons(text)
    timestamp = self._detect_timestamp(text)
    signature = self._sig_builder.build(text) if self._sig_builder else None
    return FastPathResult(
      entities=entities,
      persons=persons,
      detected_timestamp=timestamp,
      signature=signature,
    )

  def _extract_entities(self, text: str) -> List[str]:
    """Extract named entities using regex patterns."""
    entities: List[str] = []
    entities.extend(_EMAIL_RE.findall(text))
    entities.extend(_URL_RE.findall(text))
    entities.extend(_MENTION_RE.findall(text))
    entities.extend(_HASHTAG_RE.findall(text))
    entities.extend(_CAMEL_CASE_RE.findall(text))
    # Deduplicate preserving order
    seen: set[str] = set()
    result: List[str] = []
    for e in entities:
      if e not in seen:
        seen.add(e)
        result.append(e)
    return result

  def _extract_persons(self, text: str) -> List[str]:
    """Extract potential person names (capitalized word pairs)."""
    name_pattern = re.compile(r"\b([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){1,2})\b")
    candidates = name_pattern.findall(text)
    stop_phrases = {"The", "This", "That", "These", "Those", "What", "When", "Where", "Which", "How"}
    return [c for c in candidates if c.split()[0] not in stop_phrases]

  def _detect_timestamp(self, text: str) -> Optional[float]:
    """Try to extract a timestamp/date from text."""
    try:
      from dateutil import parser as dateutil_parser  # type: ignore[import-untyped]

      # Look for date-like patterns in text
      date_patterns = re.findall(
        r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s*\d{4}\b",
        text,
      )
      if date_patterns:
        parsed = dateutil_parser.parse(date_patterns[0], fuzzy=True)
        return parsed.timestamp()
    except (ImportError, ValueError):
      pass
    return None


class FastPathResult:
  """Result of fast-path processing."""

  __slots__ = ("entities", "persons", "detected_timestamp", "signature")

  def __init__(
    self,
    entities: List[str],
    persons: List[str],
    detected_timestamp: Optional[float],
    signature: Optional[bytes],
  ):
    self.entities = entities
    self.persons = persons
    self.detected_timestamp = detected_timestamp
    self.signature = signature
