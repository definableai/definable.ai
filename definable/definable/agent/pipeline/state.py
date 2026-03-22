"""Pipeline state — mutable state that flows through all phases."""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from definable.agent.run.base import RunContext
from definable.model.message import Message
from definable.model.response import ToolExecution
from definable.tool.function import Function

if TYPE_CHECKING:
  from definable.agent.config import AgentConfig
  from definable.agent.loop import CancellationToken
  from definable.agent.pipeline.sub_agent import ThreadControlBlock
  from definable.agent.reasoning.step import ReasoningStep, ThinkingOutput
  from definable.agent.run.agent import RunInput
  from definable.agent.run.requirement import RunRequirement
  from definable.model.base import Model
  from definable.model.metrics import Metrics


class LoopStatus(str, Enum):
  """Status of the pipeline execution."""

  pending = "pending"
  running = "running"
  completed = "completed"
  paused = "paused"
  cancelled = "cancelled"
  blocked = "blocked"
  error = "error"


@dataclass
class PhaseMetric:
  """Timing and execution info for a single pipeline phase."""

  phase_name: str
  duration_ms: float
  skipped: bool = False


@dataclass
class LoopState:
  """Mutable state that flows through all pipeline phases.

  Each phase reads/writes fields relevant to its concern.
  The state is the single source of truth for the entire run.
  """

  # ── Identity ──────────────────────────────────────────────
  run_id: str
  session_id: str
  user_id: Optional[str] = None
  agent_id: str = ""
  agent_name: str = ""

  # ── Input ─────────────────────────────────────────────────
  raw_instruction: Union[str, Message, List[Message], None] = None
  new_messages: List[Message] = field(default_factory=list)
  all_messages: List[Message] = field(default_factory=list)
  invoke_messages: List[Message] = field(default_factory=list)

  # ── System prompt ─────────────────────────────────────────
  system_content: str = ""

  # ── Context ───────────────────────────────────────────────
  context: Optional[RunContext] = None

  # ── Tools ─────────────────────────────────────────────────
  tools: Dict[str, Function] = field(default_factory=dict)

  # ── Output ────────────────────────────────────────────────
  content: Optional[str] = None
  parsed: Optional[Any] = None
  metrics: Optional["Metrics"] = None
  status: LoopStatus = LoopStatus.pending
  phase: str = ""
  turn: int = 0

  # ── Config ────────────────────────────────────────────────
  config: Optional["AgentConfig"] = None
  model: Optional["Model"] = None

  # ── Recall results ────────────────────────────────────────
  knowledge_context: Optional[str] = None
  memory_context: Optional[str] = None
  research_context: Optional[str] = None
  readers_context: Optional[str] = None

  # ── Thinking results ──────────────────────────────────────
  thinking_output: Optional["ThinkingOutput"] = None
  thinking_text: Optional[str] = None
  reasoning_steps: Optional[List["ReasoningStep"]] = None
  reasoning_messages: Optional[List[Message]] = None
  native_reasoning_content: Optional[str] = None

  # ── Tool executions ───────────────────────────────────────
  tool_executions: List[ToolExecution] = field(default_factory=list)

  # ── Sub-agent tracking ────────────────────────────────────
  thread_control_blocks: List["ThreadControlBlock"] = field(default_factory=list)

  # ── Run input (for RunOutput construction) ────────────────
  run_input: Optional["RunInput"] = None

  # ── HITL requirements ─────────────────────────────────────
  requirements: Optional[List["RunRequirement"]] = None

  # ── Per-run settings ───────────────────────────────────────
  streaming: bool = False
  cancellation_token: Optional["CancellationToken"] = None

  # ── Phase metrics ────────────────────────────────────────
  phase_metrics: List[PhaseMetric] = field(default_factory=list)

  # ── Extra state for phases ────────────────────────────────
  context_stats: Optional[Dict[str, Any]] = None

  extra: Dict[str, Any] = field(default_factory=dict)

  # ── Output messages (post-loop, excludes system) ──────────
  output_messages: Optional[List[Message]] = None
