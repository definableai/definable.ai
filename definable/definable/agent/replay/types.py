"""Replay module types — structured representations of past agent runs."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCallRecord:
  """Record of a single tool call from a past run."""

  tool_name: str = ""
  tool_args: Optional[Dict[str, Any]] = None
  result: Optional[str] = None
  error: Optional[bool] = None
  started_at: int = 0
  completed_at: Optional[int] = None
  duration_ms: Optional[float] = None


@dataclass
class ReplayTokens:
  """Aggregated token usage for a replay."""

  input_tokens: int = 0
  output_tokens: int = 0
  total_tokens: int = 0
  reasoning_tokens: int = 0
  cache_read_tokens: int = 0
  cache_write_tokens: int = 0


@dataclass
class ReplayStep:
  """A step in a replay — either a model call, tool call, or retrieval."""

  step_type: str = ""  # "model_call", "tool_call", "knowledge_retrieval", "memory_recall"
  name: Optional[str] = None
  started_at: int = 0
  completed_at: Optional[int] = None
  duration_ms: Optional[float] = None


@dataclass
class KnowledgeRetrievalRecord:
  """Record of a knowledge retrieval during a run."""

  query: Optional[str] = None
  documents_found: int = 0
  documents_used: int = 0
  duration_ms: Optional[float] = None


@dataclass
class MemoryRecallRecord:
  """Record of a memory recall during a run."""

  query: Optional[str] = None
  tokens_used: int = 0
  chunks_included: int = 0
  chunks_available: int = 0
  duration_ms: Optional[float] = None


@dataclass
class ToolCallsDiff:
  """Diff of tool calls between two runs."""

  added: List[ToolCallRecord] = field(default_factory=list)
  removed: List[ToolCallRecord] = field(default_factory=list)
  common: int = 0


@dataclass
class ReplayComparison:
  """Side-by-side comparison of two runs."""

  original: Any = None  # Replay (forward ref to avoid circular import)
  replayed: Any = None  # Replay
  content_diff: Optional[str] = None
  cost_diff: Optional[float] = None
  token_diff: int = 0
  duration_diff: Optional[float] = None
  tool_calls_diff: ToolCallsDiff = field(default_factory=ToolCallsDiff)
