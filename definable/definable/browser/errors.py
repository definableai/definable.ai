"""AI-friendly error enrichment for Playwright browser operations.

Converts cryptic Playwright errors into actionable messages that agents
can understand and act upon. Ported from openclaw pw-tools-core.shared.ts.
"""

from __future__ import annotations

import re


def to_ai_friendly_error(error: Exception, ref_or_selector: str = "") -> str:
  """Transform a Playwright error into a message an AI agent can act on."""
  msg = str(error)

  # Pattern 1: strict mode violation (multiple matches)
  if "strict mode violation" in msg:
    count_match = re.search(r"resolved to (\d+) elements", msg)
    count = count_match.group(1) if count_match else "multiple"
    return f'Selector "{ref_or_selector}" matched {count} elements. Run browser_snapshot() to get updated refs, or use a more specific selector.'

  # Pattern 2: timeout / not visible / not attached
  if ("Timeout" in msg or "waiting for" in msg) and ("visible" in msg.lower() or "attached" in msg.lower()):
    return f'Element "{ref_or_selector}" not found or not visible. Run browser_snapshot() to see current page elements.'

  # Pattern 3: pointer events intercepted (element covered by overlay)
  if "intercepts pointer events" in msg or "not receive pointer events" in msg:
    return (
      f'Element "{ref_or_selector}" is covered by another element. Try scrolling into view, closing overlays, or using browser_remove_elements().'
    )

  # Pattern 4: context destroyed / detached (navigation happened)
  if "detached" in msg.lower() or "context was destroyed" in msg.lower():
    return "Page navigated during operation. Wait for page load and run browser_snapshot()."

  # Pattern 5: unknown ref
  if "Unknown ref" in msg:
    return f"{msg} Run browser_snapshot() to get current refs."

  # Passthrough
  return str(error)


def normalize_timeout_ms(timeout_ms: int | float | None, fallback: int) -> int:
  """Clamp timeout to [500, 120_000] ms."""
  raw = timeout_ms if timeout_ms is not None else fallback
  return max(500, min(120_000, int(raw)))
