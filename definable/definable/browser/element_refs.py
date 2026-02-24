"""Role-based element ref system from Playwright's ariaSnapshot.

The core DX differentiator: agents interact with semantic refs (e1, e2, e3)
instead of brittle CSS selectors. Ported from openclaw pw-role-snapshot.ts.

Usage:
    - Playwright's ariaSnapshot() returns accessibility tree text
    - build_role_snapshot() parses it and assigns e1/e2/e3 refs
    - RefResolver resolves refs to Playwright Locators (or falls back to CSS)
    - Agents use browser_click("e1") or browser_click("button.submit") — both work
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from playwright.async_api import FrameLocator, Locator, Page


# ---------------------------------------------------------------------------
# Role classification (from pw-role-snapshot.ts)
# ---------------------------------------------------------------------------

INTERACTIVE_ROLES = frozenset({
  "button",
  "link",
  "textbox",
  "checkbox",
  "radio",
  "combobox",
  "listbox",
  "menuitem",
  "menuitemcheckbox",
  "menuitemradio",
  "option",
  "searchbox",
  "slider",
  "spinbutton",
  "switch",
  "tab",
  "treeitem",
})

CONTENT_ROLES = frozenset({
  "heading",
  "cell",
  "gridcell",
  "columnheader",
  "rowheader",
  "listitem",
  "article",
  "region",
  "main",
  "navigation",
})

STRUCTURAL_ROLES = frozenset({
  "generic",
  "group",
  "list",
  "table",
  "row",
  "rowgroup",
  "grid",
  "treegrid",
  "menu",
  "menubar",
  "toolbar",
  "tablist",
  "tree",
  "directory",
  "document",
  "application",
  "presentation",
  "none",
})


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RoleRef:
  """A single role-based element reference."""

  role: str
  name: str | None = None
  nth: int | None = None


RoleRefMap = dict[str, RoleRef]


@dataclass
class RoleSnapshotOptions:
  """Options for building a role snapshot."""

  interactive: bool = False
  compact: bool = False
  max_depth: int | None = None


@dataclass
class RoleSnapshotStats:
  """Statistics about a role snapshot."""

  lines: int
  chars: int
  refs: int
  interactive: int


@dataclass
class RoleSnapshotResult:
  """Result of building a role snapshot."""

  snapshot: str
  refs: RoleRefMap
  stats: RoleSnapshotStats


# ---------------------------------------------------------------------------
# Snapshot building
# ---------------------------------------------------------------------------


def _get_indent_level(line: str) -> int:
  """Get the indentation level (2 spaces = 1 level)."""
  match = re.match(r"^(\s*)", line)
  return len(match.group(1)) // 2 if match else 0


def _parse_aria_line(line: str) -> tuple[str, str, str | None, str] | None:
  """Parse an aria snapshot line into (prefix, role, name, suffix).

  Returns None if the line doesn't match the expected format.
  """
  match = re.match(r'^(\s*-\s*)(\w+)(?:\s+"([^"]*)")?(.*)$', line)
  if not match:
    return None
  prefix, role_raw, name, suffix = match.group(1), match.group(2), match.group(3), match.group(4)
  if role_raw.startswith("/"):
    return None
  return prefix, role_raw, name, suffix


def _compact_tree(tree: str) -> str:
  """Remove unnamed structural nodes without ref children."""
  lines = tree.split("\n")
  result: list[str] = []

  for i, line in enumerate(lines):
    # Always keep lines with refs
    if "[ref=" in line:
      result.append(line)
      continue

    # Keep text content lines (role: "text")
    if ":" in line and not line.rstrip().endswith(":"):
      result.append(line)
      continue

    # Check if this node has relevant children (with refs)
    current_indent = _get_indent_level(line)
    has_relevant_children = False
    for j in range(i + 1, len(lines)):
      child_indent = _get_indent_level(lines[j])
      if child_indent <= current_indent:
        break
      if "[ref=" in lines[j]:
        has_relevant_children = True
        break

    if has_relevant_children:
      result.append(line)

  return "\n".join(result)


def build_role_snapshot(
  aria_snapshot: str,
  options: RoleSnapshotOptions | None = None,
) -> RoleSnapshotResult:
  """Parse Playwright ariaSnapshot output and assign e1/e2/e3 refs.

  Algorithm (from pw-role-snapshot.ts buildRoleSnapshotFromAriaSnapshot):
  1. Split into lines, parse each: indent, role, name, suffix
  2. Assign refs to interactive elements + named content elements
  3. Track role+name duplicates via counter map
  4. Post-pass: remove nth from non-duplicates
  5. If compact: remove unnamed structural nodes without ref children
  6. If max_depth: truncate beyond depth

  Input (from Playwright ariaSnapshot()):
      - heading "Welcome" [level=1]
      - button "Sign In"
      - textbox "Email"

  Output:
      - heading "Welcome" [ref=e1] [level=1]
      - button "Sign In" [ref=e2]
      - textbox "Email" [ref=e3]
  """
  opts = options or RoleSnapshotOptions()
  lines = aria_snapshot.split("\n")
  refs: RoleRefMap = {}

  # Role+name tracker for deduplication
  counts: dict[str, int] = {}  # "role:name" -> count
  refs_by_key: dict[str, list[str]] = {}  # "role:name" -> [ref_ids]

  counter = 0

  def next_ref() -> str:
    nonlocal counter
    counter += 1
    return f"e{counter}"

  def get_key(role: str, name: str | None) -> str:
    return f"{role}:{name or ''}"

  result_lines: list[str] = []

  for line in lines:
    depth = _get_indent_level(line)

    # Max depth filter
    if opts.max_depth is not None and depth > opts.max_depth:
      continue

    parsed = _parse_aria_line(line)
    if not parsed:
      if not opts.interactive:
        result_lines.append(line)
      continue

    prefix, role_raw, name, suffix = parsed
    role = role_raw.lower()

    is_interactive = role in INTERACTIVE_ROLES
    is_content = role in CONTENT_ROLES
    is_structural = role in STRUCTURAL_ROLES

    # Interactive-only mode
    if opts.interactive and not is_interactive:
      continue

    # Compact mode: skip unnamed structural
    if opts.compact and is_structural and not name:
      continue

    # Determine if this element should get a ref
    should_have_ref = is_interactive or (is_content and name is not None)

    if not should_have_ref:
      result_lines.append(line)
      continue

    # Assign ref
    ref = next_ref()
    key = get_key(role, name)
    nth = counts.get(key, 0)
    counts[key] = nth + 1

    if key not in refs_by_key:
      refs_by_key[key] = []
    refs_by_key[key].append(ref)

    refs[ref] = RoleRef(role=role, name=name, nth=nth)

    # Build enhanced line
    enhanced = f"{prefix}{role_raw}"
    if name is not None:
      enhanced += f' "{name}"'
    enhanced += f" [ref={ref}]"
    if nth > 0:
      enhanced += f" [nth={nth}]"
    if suffix:
      enhanced += suffix

    result_lines.append(enhanced)

  # Post-pass: remove nth from non-duplicates
  duplicate_keys = {key for key, ref_list in refs_by_key.items() if len(ref_list) > 1}
  for ref_id, ref_data in refs.items():
    key = get_key(ref_data.role, ref_data.name)
    if key not in duplicate_keys:
      ref_data.nth = None

  tree = "\n".join(result_lines) or "(empty)"

  if opts.compact:
    tree = _compact_tree(tree)

  # Compute stats
  interactive_count = sum(1 for r in refs.values() if r.role in INTERACTIVE_ROLES)
  stats = RoleSnapshotStats(
    lines=len(tree.split("\n")),
    chars=len(tree),
    refs=len(refs),
    interactive=interactive_count,
  )

  return RoleSnapshotResult(snapshot=tree, refs=refs, stats=stats)


# ---------------------------------------------------------------------------
# Ref parsing utilities
# ---------------------------------------------------------------------------

_REF_PATTERN = re.compile(r"^e\d+$")


def parse_role_ref(raw: str) -> str | None:
  """Normalize: 'e1', '@e1', 'ref=e1' -> 'e1'. None if not a ref."""
  trimmed = raw.strip()
  if not trimmed:
    return None
  normalized = trimmed
  if normalized.startswith("@"):
    normalized = normalized[1:]
  elif normalized.startswith("ref="):
    normalized = normalized[4:]
  return normalized if _REF_PATTERN.match(normalized) else None


def is_ref(value: str) -> bool:
  """True if value matches /^e\\d+$/."""
  return bool(_REF_PATTERN.match(value))


# ---------------------------------------------------------------------------
# Ref resolver — connects refs to Playwright locators
# ---------------------------------------------------------------------------


class RefResolver:
  """Resolves refs to Playwright locators. Maintains ref map from last snapshot.

  The key DX innovation: ``browser_click("e1")`` and ``browser_click("button.submit")``
  both work. Auto-detects ref (starts with ``e`` + digits) vs CSS selector.
  """

  def __init__(self) -> None:
    self._refs: RoleRefMap = {}
    self._frame_selector: str | None = None

  def store(self, refs: RoleRefMap, frame_selector: str | None = None) -> None:
    """Store a new ref map (replaces previous)."""
    self._refs = refs
    self._frame_selector = frame_selector

  @property
  def refs(self) -> RoleRefMap:
    """Current ref map."""
    return self._refs

  def resolve(self, page: "Page", ref_or_selector: str) -> "Locator":
    """Unified resolution: ref (e1) -> getByRole, CSS selector -> page.locator.

    Args:
        page: Playwright Page object.
        ref_or_selector: Either an element ref (e.g. "e1") or a CSS selector.

    Returns:
        A Playwright Locator.

    Raises:
        ValueError: If the ref is unknown (not in current snapshot).
    """
    normalized = ref_or_selector.strip()
    if normalized.startswith("@"):
      normalized = normalized[1:]
    if normalized.startswith("ref="):
      normalized = normalized[4:]

    if _REF_PATTERN.match(normalized):
      info = self._refs.get(normalized)
      if not info:
        raise ValueError(f'Unknown ref "{normalized}". Run browser_snapshot() first.')

      scope: Page | FrameLocator = page
      if self._frame_selector:
        scope = page.frame_locator(self._frame_selector)

      if info.name:
        loc = scope.get_by_role(info.role, name=info.name, exact=True)  # type: ignore[arg-type]
      else:
        loc = scope.get_by_role(info.role)  # type: ignore[arg-type]

      return loc.nth(info.nth) if info.nth is not None else loc

    # Fall back to CSS selector
    return page.locator(ref_or_selector)
