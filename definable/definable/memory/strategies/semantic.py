"""SemanticStrategy — extract atomic, self-contained memory units from conversation.

Instead of summarizing the middle section into a single paragraph, this strategy
extracts structured memory atoms — self-contained facts with metadata (keywords,
entities, persons, topic). Each atom is independently understandable with all
pronouns resolved and relative time expressions disambiguated.

Follows the same pin + middle + recent split and tool-call boundary awareness
as SummarizeStrategy. The difference is what replaces the middle section:
atoms instead of a summary.

Inspired by the Semantic Structured Compression approach: sliding windows over
the conversation, LLM-driven extraction with force disambiguation and lossless
restatement constraints.
"""

import json
import re
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
  from definable.model.base import Model

from definable.memory.strategies.base import MemoryStrategy
from definable.memory.types import MemoryEntry
from definable.utils.log import log_debug, log_warning


_EXTRACTION_PROMPT = """\
You are a memory extraction engine. Given a conversation segment, extract atomic \
memory units — self-contained facts that can be understood in complete isolation.

Rules:
1. FORCE DISAMBIGUATION: Replace ALL pronouns with explicit names/references. \
Replace ALL relative time expressions ("yesterday", "last week") with absolute \
references when inferable from context.
2. LOSSLESS RESTATEMENT: Each fact must be complete, independent, and unambiguous. \
A reader with no other context must fully understand each fact.
3. COMPLETE COVERAGE: Extract enough facts to capture ALL meaningful information \
— decisions, preferences, facts, action items, technical details. Omit greetings, \
filler, and pleasantries.
4. NO DUPLICATES: Do not repeat information already captured in the previous context.

{prior_context}Conversation to extract from:
{conversation}

Respond with a JSON array. Each element must have:
- "lossless_content": string — the self-contained fact
- "keywords": string[] — 3-7 search terms for this fact
- "entities": string[] — companies, products, technologies, places mentioned
- "persons": string[] — people referenced by name
- "topic": string — a short topic phrase (2-5 words)

Output ONLY a valid JSON array, no other text."""


class SemanticStrategy(MemoryStrategy):
  """Extract atomic memory units from conversation via LLM.

  Hybrid strategy: pin first N + extract atoms from middle + keep recent M.
  Tool-call-aware: tool result entries at boundaries are pulled into adjacent
  sections to avoid orphaned tool calls.

  Args:
    pin_count: How many initial messages to preserve.
    recent_count: How many recent messages to preserve.
    window_size: Maximum conversation entries per extraction window.
    overlap_size: Entries retained from previous window for context.
  """

  def __init__(
    self,
    pin_count: int = 2,
    recent_count: int = 5,
    window_size: int = 20,
    overlap_size: int = 5,
  ) -> None:
    self.pin_count = pin_count
    self.recent_count = recent_count
    self.window_size = window_size
    self.overlap_size = overlap_size

  async def optimize(self, entries: List[MemoryEntry], model: "Model") -> List[MemoryEntry]:
    """Extract atomic memory units from the middle section.

    Args:
      entries: All session entries, ordered by created_at.
      model: LLM model for extraction.

    Returns:
      Optimized list: pinned + atom entries + recent.
    """
    if len(entries) <= self.pin_count + self.recent_count:
      return entries

    pinned = list(entries[: self.pin_count])
    recent = list(entries[-self.recent_count :])
    middle = list(entries[self.pin_count : -self.recent_count])

    # Tool-call boundary: pull tool entries from start of middle into pinned.
    while middle and middle[0].role == "tool":
      pinned.append(middle.pop(0))

    # Tool-call boundary: pull preceding entries when recent starts with tool.
    while recent and recent[0].role == "tool" and middle:
      recent.insert(0, middle.pop(-1))

    if not middle:
      return entries  # Nothing to extract after adjustments

    # Extract atoms from middle using sliding windows.
    atoms = await self._extract_from_windows(middle, model, pinned)

    log_debug(
      f"SemanticStrategy: {len(entries)} entries -> {len(pinned) + len(atoms) + len(recent)} "
      f"(pin={len(pinned)}, atoms={len(atoms)}, recent={len(recent)})"
    )
    return pinned + atoms + recent

  async def _extract_from_windows(
    self,
    middle: List[MemoryEntry],
    model: "Model",
    pinned: List[MemoryEntry],
  ) -> List[MemoryEntry]:
    """Split middle into sliding windows and extract atoms from each."""
    windows = self._split_into_windows(middle)
    all_atoms: List[MemoryEntry] = []
    previous_atoms: List[MemoryEntry] = []

    # Gather prior context from any existing summaries/atoms in pinned.
    prior_summaries = [e.content for e in pinned if e.role in ("summary", "atom")]

    for window in windows:
      atoms = await self._extract_window(window, model, prior_summaries, previous_atoms)
      all_atoms.extend(atoms)
      previous_atoms = atoms[-3:] if atoms else []  # Keep last 3 for dedup context

    return all_atoms

  def _split_into_windows(self, entries: List[MemoryEntry]) -> List[List[MemoryEntry]]:
    """Split entries into overlapping windows."""
    if len(entries) <= self.window_size:
      return [entries]

    windows: List[List[MemoryEntry]] = []
    step = max(1, self.window_size - self.overlap_size)
    for i in range(0, len(entries), step):
      window = entries[i : i + self.window_size]
      if window:
        windows.append(window)
    return windows

  async def _extract_window(
    self,
    window: List[MemoryEntry],
    model: "Model",
    prior_summaries: List[str],
    previous_atoms: List[MemoryEntry],
  ) -> List[MemoryEntry]:
    """Extract atoms from a single conversation window."""
    # Build prior context.
    prior_parts: List[str] = []
    if prior_summaries:
      prior_parts.append("Previous context:\n" + "\n".join(prior_summaries))
    if previous_atoms:
      dedup_context = "\n".join(f"- {a.lossless_content or a.content}" for a in previous_atoms)
      prior_parts.append(f"Already captured (do NOT repeat):\n{dedup_context}")

    prior_context = "\n\n".join(prior_parts)
    if prior_context:
      prior_context += "\n\n"

    # Format conversation.
    conv_lines = [f"{e.role}: {e.content}" for e in window]
    conversation = "\n".join(conv_lines)

    prompt = _EXTRACTION_PROMPT.format(prior_context=prior_context, conversation=conversation)

    # Call LLM.
    try:
      from definable.model.message import Message

      messages = [Message(role="user", content=prompt)]
      assistant_message = Message(role="assistant", content="")
      response = await model.ainvoke(messages=messages, assistant_message=assistant_message)
      raw_text = response.content or "[]"
    except Exception as exc:
      log_warning(f"SemanticStrategy: LLM extraction failed: {exc}")
      return self._fallback_atoms(window)

    # Parse JSON response into atom entries.
    atoms_data = _extract_json_array(raw_text)
    if not atoms_data:
      log_warning("SemanticStrategy: no atoms extracted from LLM response, using fallback")
      return self._fallback_atoms(window)

    # Build atom MemoryEntries.
    first = window[0]
    atoms: List[MemoryEntry] = []
    for item in atoms_data:
      if not isinstance(item, dict):
        continue
      lossless = item.get("lossless_content", "")
      if not lossless:
        continue
      atoms.append(
        MemoryEntry(
          session_id=first.session_id,
          user_id=first.user_id,
          role="atom",
          content=lossless,
          created_at=first.created_at,
          updated_at=first.updated_at,
          entry_type="atom",
          lossless_content=lossless,
          keywords=item.get("keywords", []),
          entities=item.get("entities", []),
          persons=item.get("persons", []),
          topic=item.get("topic"),
          importance=0.5,
        )
      )

    return atoms or self._fallback_atoms(window)

  @staticmethod
  def _fallback_atoms(window: List[MemoryEntry]) -> List[MemoryEntry]:
    """Produce a single summary-style atom as fallback when extraction fails."""
    first = window[0]
    content = f"Conversation segment of {len(window)} messages."
    return [
      MemoryEntry(
        session_id=first.session_id,
        user_id=first.user_id,
        role="atom",
        content=content,
        created_at=first.created_at,
        updated_at=first.updated_at,
        entry_type="atom",
        lossless_content=content,
        keywords=[],
        entities=[],
        persons=[],
        topic="conversation segment",
        importance=0.3,
      )
    ]


def _extract_json_array(text: str) -> List[Any]:
  """Robustly extract a JSON array from LLM response text.

  Returns a list of items (expected to be dicts but not guaranteed — callers
  must validate). Handles: raw JSON, ```json fences, extra text around the array.
  """
  text = text.strip()

  # 1. Direct parse.
  try:
    parsed = json.loads(text)
    if isinstance(parsed, list):
      return parsed
  except (json.JSONDecodeError, ValueError):
    pass

  # 2. Extract from ```json ... ``` fences.
  fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
  if fence_match:
    try:
      parsed = json.loads(fence_match.group(1).strip())
      if isinstance(parsed, list):
        return parsed
    except (json.JSONDecodeError, ValueError):
      pass

  # 3. Find the outermost [...] bracket pair.
  start = text.find("[")
  if start != -1:
    depth = 0
    for i in range(start, len(text)):
      if text[i] == "[":
        depth += 1
      elif text[i] == "]":
        depth -= 1
        if depth == 0:
          try:
            parsed = json.loads(text[start : i + 1])
            if isinstance(parsed, list):
              return parsed
          except (json.JSONDecodeError, ValueError):
            pass
          break

  return []
