"""Compare two replays side-by-side."""

import difflib
from typing import Any, Union

from definable.agent.replay.replay import Replay
from definable.agent.replay.types import ReplayComparison, ToolCallsDiff


def compare_runs(
  a: Union[Replay, Any],
  b: Union[Replay, Any],
) -> ReplayComparison:
  """Compare two runs and produce a structured diff.

  Args:
    a: First run (Replay or RunOutput).
    b: Second run (Replay or RunOutput).

  Returns:
    ReplayComparison with diffs for content, cost, tokens, and tool calls.
  """
  # Convert RunOutput to Replay if needed
  if not isinstance(a, Replay):
    a = Replay.from_run_output(a)
  if not isinstance(b, Replay):
    b = Replay.from_run_output(b)

  # Content diff
  content_diff = None
  a_content = str(a.content) if a.content is not None else ""
  b_content = str(b.content) if b.content is not None else ""
  if a_content != b_content:
    diff_lines = difflib.unified_diff(
      a_content.splitlines(keepends=True),
      b_content.splitlines(keepends=True),
      fromfile="original",
      tofile="replayed",
    )
    content_diff = "".join(diff_lines)

  # Cost diff
  cost_diff = None
  if a.cost is not None and b.cost is not None:
    cost_diff = b.cost - a.cost

  # Token diff
  token_diff = b.tokens.total_tokens - a.tokens.total_tokens

  # Duration diff
  duration_diff = None
  if a.duration is not None and b.duration is not None:
    duration_diff = b.duration - a.duration

  # Tool calls diff
  a_tool_names = [tc.tool_name for tc in a.tool_calls]
  b_tool_names = [tc.tool_name for tc in b.tool_calls]

  # Find common (by position & name match)
  common = 0
  for a_name, b_name in zip(a_tool_names, b_tool_names):
    if a_name == b_name:
      common += 1

  # Build added/removed based on name presence
  a_name_set = set(a_tool_names)
  b_name_set = set(b_tool_names)

  added = [tc for tc in b.tool_calls if tc.tool_name not in a_name_set]
  removed = [tc for tc in a.tool_calls if tc.tool_name not in b_name_set]

  tool_calls_diff = ToolCallsDiff(
    added=added,
    removed=removed,
    common=common,
  )

  return ReplayComparison(
    original=a,
    replayed=b,
    content_diff=content_diff,
    cost_diff=cost_diff,
    token_diff=token_diff,
    duration_diff=duration_diff,
    tool_calls_diff=tool_calls_diff,
  )
