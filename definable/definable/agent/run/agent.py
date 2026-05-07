from dataclasses import asdict, dataclass, field
from enum import Enum
from time import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

from definable.media import Audio, File, Image, Video
from definable.types import ToolCallDict
from definable.model.message import Citations, Message
from definable.model.metrics import Metrics
from definable.model.response import ToolExecution
from definable.agent.reasoning.step import ReasoningStep
from definable.agent.run.base import BaseRunOutputEvent, MessageReferences, RunStatus
from definable.agent.run.requirement import RunRequirement
from definable.utils.log import logger
from definable.utils.media import reconstruct_audio_list, reconstruct_files, reconstruct_images, reconstruct_response_audio, reconstruct_videos
from pydantic import BaseModel

if TYPE_CHECKING:
  from definable.agent.guardrail.events import GuardrailBlockedEvent, GuardrailCheckedEvent
  from definable.agent.interface.desktop.events import BridgeCallEvent, DesktopActionEvent
  from definable.agent.interface.gateway import (
    InterfaceErrorEvent,
    InterfaceRestartedEvent,
    InterfaceStartedEvent,
    InterfaceStoppedEvent,
  )
  from definable.agent.pipeline.state import PhaseMetric


@dataclass
class RunInput:
  """Container for the raw input data passed to Agent.run().

  This captures the original input exactly as provided by the user,
  separate from the processed messages that go to the model.

  Attributes:
      input_content: The literal input message/content passed to run()
      images: Images directly passed to run()
      videos: Videos directly passed to run()
      audios: Audio files directly passed to run()
      files: Files directly passed to run()
  """

  input_content: Union[str, List, Dict, Message, BaseModel, List[Message]]
  images: Optional[Sequence[Image]] = None
  videos: Optional[Sequence[Video]] = None
  audios: Optional[Sequence[Audio]] = None
  files: Optional[Sequence[File]] = None

  def input_content_string(self) -> str:
    import json

    if isinstance(self.input_content, (str)):
      return self.input_content
    elif isinstance(self.input_content, BaseModel):
      return self.input_content.model_dump_json(exclude_none=True)
    elif isinstance(self.input_content, Message):  # type: ignore[unreachable]
      return json.dumps(self.input_content.to_dict())  # type: ignore[unreachable]
    elif isinstance(self.input_content, list):
      try:
        return json.dumps(self.to_dict().get("input_content"))
      except Exception:
        return str(self.input_content)
    else:
      return str(self.input_content)

  def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary representation"""
    result: Dict[str, Any] = {}

    if self.input_content is not None:
      if isinstance(self.input_content, (str)):
        result["input_content"] = self.input_content
      elif isinstance(self.input_content, BaseModel):
        result["input_content"] = self.input_content.model_dump(exclude_none=True)
      elif isinstance(self.input_content, Message):  # type: ignore[unreachable]
        result["input_content"] = self.input_content.to_dict()  # type: ignore[unreachable]
      elif isinstance(self.input_content, list):
        serialized_items: List[Any] = []
        for item in self.input_content:
          if isinstance(item, Message):
            serialized_items.append(item.to_dict())
          elif isinstance(item, BaseModel):
            serialized_items.append(item.model_dump(exclude_none=True))
          elif isinstance(item, dict):
            content = dict(item)
            if content.get("images"):
              content["images"] = [img.to_dict() if isinstance(img, Image) else img for img in content["images"]]
            if content.get("videos"):
              content["videos"] = [vid.to_dict() if isinstance(vid, Video) else vid for vid in content["videos"]]
            if content.get("audios"):
              content["audios"] = [aud.to_dict() if isinstance(aud, Audio) else aud for aud in content["audios"]]
            if content.get("files"):
              content["files"] = [file.to_dict() if isinstance(file, File) else file for file in content["files"]]
            serialized_items.append(content)
          else:
            serialized_items.append(item)

        result["input_content"] = serialized_items
      else:
        result["input_content"] = self.input_content

    if self.images:
      result["images"] = [img.to_dict() for img in self.images]
    if self.videos:
      result["videos"] = [vid.to_dict() for vid in self.videos]
    if self.audios:
      result["audios"] = [aud.to_dict() for aud in self.audios]
    if self.files:
      result["files"] = [file.to_dict() for file in self.files]

    return result

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "RunInput":
    """Create RunInput from dictionary"""
    images = reconstruct_images(data.get("images"))
    videos = reconstruct_videos(data.get("videos"))
    audios = reconstruct_audio_list(data.get("audios"))
    files = reconstruct_files(data.get("files"))

    return cls(input_content=data.get("input_content", ""), images=images, videos=videos, audios=audios, files=files)


class RunEvent(str, Enum):
  """Events that can be sent by the run() functions"""

  run_started = "RunStarted"
  run_content = "RunContent"
  run_content_completed = "RunContentCompleted"
  run_intermediate_content = "RunIntermediateContent"
  run_completed = "RunCompleted"
  run_error = "RunError"
  run_cancelled = "RunCancelled"

  run_paused = "RunPaused"
  run_continued = "RunContinued"

  pre_hook_started = "PreHookStarted"
  pre_hook_completed = "PreHookCompleted"

  post_hook_started = "PostHookStarted"
  post_hook_completed = "PostHookCompleted"

  tool_call_started = "ToolCallStarted"
  tool_call_completed = "ToolCallCompleted"
  tool_call_error = "ToolCallError"

  reasoning_started = "ReasoningStarted"
  reasoning_step = "ReasoningStep"
  reasoning_content_delta = "ReasoningContentDelta"
  reasoning_completed = "ReasoningCompleted"

  knowledge_retrieval_started = "KnowledgeRetrievalStarted"
  knowledge_retrieval_completed = "KnowledgeRetrievalCompleted"

  memory_recall_started = "MemoryRecallStarted"
  memory_recall_completed = "MemoryRecallCompleted"

  memory_update_started = "MemoryUpdateStarted"
  memory_update_completed = "MemoryUpdateCompleted"

  file_read_started = "FileReadStarted"
  file_read_completed = "FileReadCompleted"

  session_summary_started = "SessionSummaryStarted"
  session_summary_completed = "SessionSummaryCompleted"

  parser_model_response_started = "ParserModelResponseStarted"
  parser_model_response_completed = "ParserModelResponseCompleted"

  output_model_response_started = "OutputModelResponseStarted"
  output_model_response_completed = "OutputModelResponseCompleted"

  deep_research_started = "DeepResearchStarted"
  deep_research_progress = "DeepResearchProgress"
  deep_research_completed = "DeepResearchCompleted"

  guardrail_checked = "GuardrailChecked"
  guardrail_blocked = "GuardrailBlocked"

  model_call_started = "ModelCallStarted"
  model_call_completed = "ModelCallCompleted"

  custom_event = "CustomEvent"

  # Sub-agent events
  sub_agent_spawned = "SubAgentSpawned"
  sub_agent_completed = "SubAgentCompleted"
  sub_agent_failed = "SubAgentFailed"
  sub_agent_killed = "SubAgentKilled"

  # Tool content streaming
  tool_content = "ToolContent"

  # Compression events
  compression_started = "CompressionStarted"
  compression_completed = "CompressionCompleted"

  # Pipeline phase events
  phase_started = "PhaseStarted"
  phase_completed = "PhaseCompleted"

  # Interface gateway events
  interface_started = "InterfaceStarted"
  interface_stopped = "InterfaceStopped"
  interface_restarted = "InterfaceRestarted"
  interface_error = "InterfaceError"

  # Desktop bridge events
  bridge_call = "BridgeCall"
  desktop_action = "DesktopAction"


@dataclass
class BaseAgentRunEvent(BaseRunOutputEvent):
  created_at: int = field(default_factory=lambda: int(time()))
  event: str = ""
  agent_id: str = ""
  agent_name: str = ""
  run_id: Optional[str] = None
  parent_run_id: Optional[str] = None
  session_id: Optional[str] = None

  tools: Optional[List[ToolExecution]] = None

  # For backwards compatibility
  content: Optional[Any] = None

  @property
  def tools_requiring_confirmation(self):
    return [t for t in self.tools if t.requires_confirmation] if self.tools else []

  @property
  def tools_requiring_user_input(self):
    return [t for t in self.tools if t.requires_user_input] if self.tools else []

  @property
  def tools_awaiting_external_execution(self):
    return [t for t in self.tools if t.external_execution_required] if self.tools else []


@dataclass
class RunStartedEvent(BaseAgentRunEvent):
  """Event sent when the run starts"""

  event: str = RunEvent.run_started.value
  model: str = ""
  model_provider: str = ""
  run_input: Optional[RunInput] = None


@dataclass
class RunContentEvent(BaseAgentRunEvent):
  """Main event for each delta of the RunOutput"""

  event: str = RunEvent.run_content.value
  content: Optional[Any] = None
  content_type: str = "str"
  reasoning_content: Optional[str] = None
  model_provider_data: Optional[Dict[str, Any]] = None
  citations: Optional[Citations] = None
  response_audio: Optional[Audio] = None  # Model audio response
  image: Optional[Image] = None  # Image attached to the response
  references: Optional[List[MessageReferences]] = None
  additional_input: Optional[List[Message]] = None
  reasoning_steps: Optional[List[ReasoningStep]] = None
  reasoning_messages: Optional[List[Message]] = None


@dataclass
class RunContentCompletedEvent(BaseAgentRunEvent):
  event: str = RunEvent.run_content_completed.value


@dataclass
class IntermediateRunContentEvent(BaseAgentRunEvent):
  event: str = RunEvent.run_intermediate_content.value
  content: Optional[Any] = None
  content_type: str = "str"


@dataclass
class RunCompletedEvent(BaseAgentRunEvent):
  event: str = RunEvent.run_completed.value
  content: Optional[Any] = None
  parsed: Optional[Any] = None
  content_type: str = "str"
  reasoning_content: Optional[str] = None
  citations: Optional[Citations] = None
  model_provider_data: Optional[Dict[str, Any]] = None
  images: Optional[List[Image]] = None  # Images attached to the response
  videos: Optional[List[Video]] = None  # Videos attached to the response
  audio: Optional[List[Audio]] = None  # Audio attached to the response
  response_audio: Optional[Audio] = None  # Model audio response
  references: Optional[List[MessageReferences]] = None
  additional_input: Optional[List[Message]] = None
  reasoning_steps: Optional[List[ReasoningStep]] = None
  reasoning_messages: Optional[List[Message]] = None
  metadata: Optional[Dict[str, Any]] = None
  metrics: Optional[Metrics] = None
  session_state: Optional[Dict[str, Any]] = None


@dataclass
class RunPausedEvent(BaseAgentRunEvent):
  event: str = RunEvent.run_paused.value
  tools: Optional[List[ToolExecution]] = None
  requirements: Optional[List[RunRequirement]] = None

  @property
  def is_paused(self):
    return True

  @property
  def active_requirements(self) -> List[RunRequirement]:
    if not self.requirements:
      return []
    return [requirement for requirement in self.requirements if not requirement.is_resolved()]


@dataclass
class RunContinuedEvent(BaseAgentRunEvent):
  event: str = RunEvent.run_continued.value


@dataclass
class RunErrorEvent(BaseAgentRunEvent):
  event: str = RunEvent.run_error.value
  content: Optional[str] = None

  # From exceptions
  error_type: Optional[str] = None
  error_id: Optional[str] = None
  additional_data: Optional[Dict[str, Any]] = None


@dataclass
class RunCancelledEvent(BaseAgentRunEvent):
  event: str = RunEvent.run_cancelled.value
  reason: Optional[str] = None

  @property
  def is_cancelled(self):
    return True


@dataclass
class PreHookStartedEvent(BaseAgentRunEvent):
  event: str = RunEvent.pre_hook_started.value
  pre_hook_name: Optional[str] = None
  run_input: Optional[RunInput] = None


@dataclass
class PreHookCompletedEvent(BaseAgentRunEvent):
  event: str = RunEvent.pre_hook_completed.value
  pre_hook_name: Optional[str] = None
  run_input: Optional[RunInput] = None


@dataclass
class PostHookStartedEvent(BaseAgentRunEvent):
  event: str = RunEvent.post_hook_started.value
  post_hook_name: Optional[str] = None


@dataclass
class PostHookCompletedEvent(BaseAgentRunEvent):
  event: str = RunEvent.post_hook_completed.value
  post_hook_name: Optional[str] = None


@dataclass
class MemoryUpdateStartedEvent(BaseAgentRunEvent):
  event: str = RunEvent.memory_update_started.value
  message_count: int = 0


@dataclass
class MemoryUpdateCompletedEvent(BaseAgentRunEvent):
  event: str = RunEvent.memory_update_completed.value
  message_count: int = 0
  duration_ms: Optional[float] = None


@dataclass
class SessionSummaryStartedEvent(BaseAgentRunEvent):
  event: str = RunEvent.session_summary_started.value


@dataclass
class SessionSummaryCompletedEvent(BaseAgentRunEvent):
  event: str = RunEvent.session_summary_completed.value
  session_summary: Optional[Any] = None  # SessionSummary not available in this fork


@dataclass
class ReasoningStartedEvent(BaseAgentRunEvent):
  event: str = RunEvent.reasoning_started.value


@dataclass
class ReasoningStepEvent(BaseAgentRunEvent):
  event: str = RunEvent.reasoning_step.value
  content: Optional[Any] = None
  content_type: str = "str"
  reasoning_content: str = ""


@dataclass
class ReasoningContentDeltaEvent(BaseAgentRunEvent):
  """Event for streaming reasoning content chunks as they arrive."""

  event: str = RunEvent.reasoning_content_delta.value
  reasoning_content: str = ""  # The delta/chunk of reasoning content


@dataclass
class ReasoningCompletedEvent(BaseAgentRunEvent):
  event: str = RunEvent.reasoning_completed.value
  content: Optional[Any] = None
  content_type: str = "str"


@dataclass
class KnowledgeRetrievalStartedEvent(BaseAgentRunEvent):
  event: str = RunEvent.knowledge_retrieval_started.value
  query: Optional[str] = None


@dataclass
class KnowledgeRetrievalCompletedEvent(BaseAgentRunEvent):
  event: str = RunEvent.knowledge_retrieval_completed.value
  query: Optional[str] = None
  documents_found: int = 0
  documents_used: int = 0
  duration_ms: Optional[float] = None


@dataclass
class MemoryRecallStartedEvent(BaseAgentRunEvent):
  event: str = RunEvent.memory_recall_started.value
  query: Optional[str] = None


@dataclass
class MemoryRecallCompletedEvent(BaseAgentRunEvent):
  event: str = RunEvent.memory_recall_completed.value
  query: Optional[str] = None
  tokens_used: int = 0
  chunks_included: int = 0
  chunks_available: int = 0
  duration_ms: Optional[float] = None


@dataclass
class FileReadStartedEvent(BaseAgentRunEvent):
  event: str = RunEvent.file_read_started.value
  file_count: int = 0


@dataclass
class FileReadCompletedEvent(BaseAgentRunEvent):
  event: str = RunEvent.file_read_completed.value
  file_count: int = 0
  files_read: int = 0
  files_failed: int = 0
  duration_ms: Optional[float] = None


@dataclass
class DeepResearchStartedEvent(BaseAgentRunEvent):
  event: str = RunEvent.deep_research_started.value
  query: Optional[str] = None
  depth: Optional[str] = None


@dataclass
class DeepResearchProgressEvent(BaseAgentRunEvent):
  event: str = RunEvent.deep_research_progress.value
  wave: int = 0
  sources_read: int = 0
  facts_extracted: int = 0
  gaps_remaining: int = 0
  message: str = ""


@dataclass
class DeepResearchCompletedEvent(BaseAgentRunEvent):
  event: str = RunEvent.deep_research_completed.value
  query: Optional[str] = None
  sources_used: int = 0
  facts_extracted: int = 0
  contradictions_found: int = 0
  waves_executed: int = 0
  duration_ms: Optional[float] = None
  compression_ratio: float = 0.0


@dataclass
class ToolCallStartedEvent(BaseAgentRunEvent):
  event: str = RunEvent.tool_call_started.value
  tool: Optional[ToolExecution] = None


@dataclass
class ToolCallCompletedEvent(BaseAgentRunEvent):
  event: str = RunEvent.tool_call_completed.value
  tool: Optional[ToolExecution] = None
  content: Optional[Any] = None
  images: Optional[List[Image]] = None  # Images produced by the tool call
  videos: Optional[List[Video]] = None  # Videos produced by the tool call
  audio: Optional[List[Audio]] = None  # Audio produced by the tool call


@dataclass
class ToolCallErrorEvent(BaseAgentRunEvent):
  event: str = RunEvent.tool_call_error.value
  tool: Optional[ToolExecution] = None
  error: Optional[str] = None


@dataclass
class ToolContentEvent(BaseAgentRunEvent):
  """Streaming chunk emitted as a tool yields incremental results."""

  event: str = RunEvent.tool_content.value
  tool_name: str = ""
  tool_call_id: Optional[str] = None
  chunk: str = ""
  chunk_index: int = 0
  is_final: bool = False


@dataclass
class ParserModelResponseStartedEvent(BaseAgentRunEvent):
  event: str = RunEvent.parser_model_response_started.value


@dataclass
class ParserModelResponseCompletedEvent(BaseAgentRunEvent):
  event: str = RunEvent.parser_model_response_completed.value


@dataclass
class OutputModelResponseStartedEvent(BaseAgentRunEvent):
  event: str = RunEvent.output_model_response_started.value


@dataclass
class OutputModelResponseCompletedEvent(BaseAgentRunEvent):
  event: str = RunEvent.output_model_response_completed.value


@dataclass
class ModelCallStartedEvent(BaseAgentRunEvent):
  """Emitted immediately before each model.ainvoke() call."""

  event: str = RunEvent.model_call_started.value
  turn: int = 0
  messages: Optional[List[Message]] = None
  tool_definitions: Optional[List[Dict[str, Any]]] = None
  response_format: Optional[Any] = None
  model_id: str = ""
  model_provider: str = ""


@dataclass
class ModelCallCompletedEvent(BaseAgentRunEvent):
  """Emitted immediately after each model.ainvoke() returns."""

  event: str = RunEvent.model_call_completed.value
  turn: int = 0
  content: Optional[str] = None
  tool_calls: Optional[list[ToolCallDict]] = None
  metrics: Optional[Metrics] = None
  model_id: str = ""


@dataclass
class CompressionStartedEvent(BaseAgentRunEvent):
  """Emitted when tool result compression begins."""

  event: str = RunEvent.compression_started.value
  tool_results_count: int = 0


@dataclass
class CompressionCompletedEvent(BaseAgentRunEvent):
  """Emitted when tool result compression finishes."""

  event: str = RunEvent.compression_completed.value
  tool_results_compressed: int = 0
  original_size: int = 0
  compressed_size: int = 0
  duration_ms: float = 0.0


@dataclass
class SubAgentSpawnedEvent(BaseAgentRunEvent):
  """Emitted when a sub-agent is spawned by the parent."""

  event: str = RunEvent.sub_agent_spawned.value
  thread_id: Optional[str] = None
  goal: Optional[str] = None
  instructions: Optional[str] = None


@dataclass
class SubAgentCompletedEvent(BaseAgentRunEvent):
  """Emitted when a sub-agent completes successfully."""

  event: str = RunEvent.sub_agent_completed.value
  thread_id: Optional[str] = None
  result: Optional[str] = None
  metrics: Optional[Metrics] = None


@dataclass
class SubAgentFailedEvent(BaseAgentRunEvent):
  """Emitted when a sub-agent fails."""

  event: str = RunEvent.sub_agent_failed.value
  thread_id: Optional[str] = None
  error: Optional[str] = None


@dataclass
class SubAgentKilledEvent(BaseAgentRunEvent):
  """Emitted when a sub-agent is killed (e.g., timeout or parent cancellation)."""

  event: str = RunEvent.sub_agent_killed.value
  thread_id: Optional[str] = None
  reason: Optional[str] = None


@dataclass
class PhaseStartedEvent(BaseAgentRunEvent):
  """Emitted when a pipeline phase begins execution."""

  event: str = RunEvent.phase_started.value
  phase_name: str = ""


@dataclass
class PhaseCompletedEvent(BaseAgentRunEvent):
  """Emitted when a pipeline phase finishes execution."""

  event: str = RunEvent.phase_completed.value
  phase_name: str = ""
  duration_ms: float = 0.0
  skipped: bool = False


@dataclass
class CustomEvent(BaseAgentRunEvent):
  event: str = RunEvent.custom_event.value

  def __init__(self, **kwargs):
    # Store arbitrary attributes directly on the instance
    for key, value in kwargs.items():
      setattr(self, key, value)


RunOutputEvent = Union[
  RunStartedEvent,
  RunContentEvent,
  IntermediateRunContentEvent,
  RunContentCompletedEvent,
  RunCompletedEvent,
  RunErrorEvent,
  RunCancelledEvent,
  RunPausedEvent,
  RunContinuedEvent,
  PreHookStartedEvent,
  PreHookCompletedEvent,
  PostHookStartedEvent,
  PostHookCompletedEvent,
  ReasoningStartedEvent,
  ReasoningStepEvent,
  ReasoningContentDeltaEvent,
  ReasoningCompletedEvent,
  KnowledgeRetrievalStartedEvent,
  KnowledgeRetrievalCompletedEvent,
  MemoryRecallStartedEvent,
  MemoryRecallCompletedEvent,
  MemoryUpdateStartedEvent,
  MemoryUpdateCompletedEvent,
  FileReadStartedEvent,
  FileReadCompletedEvent,
  DeepResearchStartedEvent,
  DeepResearchProgressEvent,
  DeepResearchCompletedEvent,
  SessionSummaryStartedEvent,
  SessionSummaryCompletedEvent,
  ToolCallStartedEvent,
  ToolCallCompletedEvent,
  ToolCallErrorEvent,
  ToolContentEvent,
  ParserModelResponseStartedEvent,
  ParserModelResponseCompletedEvent,
  OutputModelResponseStartedEvent,
  OutputModelResponseCompletedEvent,
  ModelCallStartedEvent,
  ModelCallCompletedEvent,
  CompressionStartedEvent,
  CompressionCompletedEvent,
  SubAgentSpawnedEvent,
  SubAgentCompletedEvent,
  SubAgentFailedEvent,
  SubAgentKilledEvent,
  PhaseStartedEvent,
  PhaseCompletedEvent,
  "InterfaceStartedEvent",
  "InterfaceStoppedEvent",
  "InterfaceRestartedEvent",
  "InterfaceErrorEvent",
  CustomEvent,
  "GuardrailCheckedEvent",
  "GuardrailBlockedEvent",
  "BridgeCallEvent",
  "DesktopActionEvent",
]

# Map event string to dataclass
RUN_EVENT_TYPE_REGISTRY = {
  RunEvent.run_started.value: RunStartedEvent,
  RunEvent.run_content.value: RunContentEvent,
  RunEvent.run_content_completed.value: RunContentCompletedEvent,
  RunEvent.run_intermediate_content.value: IntermediateRunContentEvent,
  RunEvent.run_completed.value: RunCompletedEvent,
  RunEvent.run_error.value: RunErrorEvent,
  RunEvent.run_cancelled.value: RunCancelledEvent,
  RunEvent.run_paused.value: RunPausedEvent,
  RunEvent.run_continued.value: RunContinuedEvent,
  RunEvent.pre_hook_started.value: PreHookStartedEvent,
  RunEvent.pre_hook_completed.value: PreHookCompletedEvent,
  RunEvent.post_hook_started.value: PostHookStartedEvent,
  RunEvent.post_hook_completed.value: PostHookCompletedEvent,
  RunEvent.reasoning_started.value: ReasoningStartedEvent,
  RunEvent.reasoning_step.value: ReasoningStepEvent,
  RunEvent.reasoning_content_delta.value: ReasoningContentDeltaEvent,
  RunEvent.reasoning_completed.value: ReasoningCompletedEvent,
  RunEvent.knowledge_retrieval_started.value: KnowledgeRetrievalStartedEvent,
  RunEvent.knowledge_retrieval_completed.value: KnowledgeRetrievalCompletedEvent,
  RunEvent.memory_recall_started.value: MemoryRecallStartedEvent,
  RunEvent.memory_recall_completed.value: MemoryRecallCompletedEvent,
  RunEvent.memory_update_started.value: MemoryUpdateStartedEvent,
  RunEvent.memory_update_completed.value: MemoryUpdateCompletedEvent,
  RunEvent.file_read_started.value: FileReadStartedEvent,
  RunEvent.file_read_completed.value: FileReadCompletedEvent,
  RunEvent.deep_research_started.value: DeepResearchStartedEvent,
  RunEvent.deep_research_progress.value: DeepResearchProgressEvent,
  RunEvent.deep_research_completed.value: DeepResearchCompletedEvent,
  RunEvent.session_summary_started.value: SessionSummaryStartedEvent,
  RunEvent.session_summary_completed.value: SessionSummaryCompletedEvent,
  RunEvent.tool_call_started.value: ToolCallStartedEvent,
  RunEvent.tool_call_completed.value: ToolCallCompletedEvent,
  RunEvent.tool_call_error.value: ToolCallErrorEvent,
  RunEvent.tool_content.value: ToolContentEvent,
  RunEvent.parser_model_response_started.value: ParserModelResponseStartedEvent,
  RunEvent.parser_model_response_completed.value: ParserModelResponseCompletedEvent,
  RunEvent.output_model_response_started.value: OutputModelResponseStartedEvent,
  RunEvent.output_model_response_completed.value: OutputModelResponseCompletedEvent,
  RunEvent.model_call_started.value: ModelCallStartedEvent,
  RunEvent.model_call_completed.value: ModelCallCompletedEvent,
  RunEvent.compression_started.value: CompressionStartedEvent,
  RunEvent.compression_completed.value: CompressionCompletedEvent,
  RunEvent.custom_event.value: CustomEvent,
  # Sub-agent events
  RunEvent.sub_agent_spawned.value: SubAgentSpawnedEvent,
  RunEvent.sub_agent_completed.value: SubAgentCompletedEvent,
  RunEvent.sub_agent_failed.value: SubAgentFailedEvent,
  RunEvent.sub_agent_killed.value: SubAgentKilledEvent,
  # Pipeline phase events
  RunEvent.phase_started.value: PhaseStartedEvent,
  RunEvent.phase_completed.value: PhaseCompletedEvent,
  # Interface gateway events are registered lazily to avoid circular imports.
  # See _ensure_interface_events_registered() below.
  # Guardrail events are registered lazily to avoid circular imports.
  # See _ensure_guardrail_events_registered() below.
}


def _ensure_guardrail_events_registered() -> None:
  """Lazily register guardrail event types the first time they're needed."""
  if RunEvent.guardrail_checked.value not in RUN_EVENT_TYPE_REGISTRY:
    from definable.agent.guardrail.events import GuardrailBlockedEvent, GuardrailCheckedEvent

    RUN_EVENT_TYPE_REGISTRY[RunEvent.guardrail_checked.value] = GuardrailCheckedEvent
    RUN_EVENT_TYPE_REGISTRY[RunEvent.guardrail_blocked.value] = GuardrailBlockedEvent


def _ensure_interface_events_registered() -> None:
  """Lazily register interface gateway event types."""
  if RunEvent.interface_started.value not in RUN_EVENT_TYPE_REGISTRY:
    from definable.agent.interface.gateway import (
      InterfaceErrorEvent,
      InterfaceRestartedEvent,
      InterfaceStartedEvent,
      InterfaceStoppedEvent,
    )

    RUN_EVENT_TYPE_REGISTRY[RunEvent.interface_started.value] = InterfaceStartedEvent
    RUN_EVENT_TYPE_REGISTRY[RunEvent.interface_stopped.value] = InterfaceStoppedEvent
    RUN_EVENT_TYPE_REGISTRY[RunEvent.interface_restarted.value] = InterfaceRestartedEvent
    RUN_EVENT_TYPE_REGISTRY[RunEvent.interface_error.value] = InterfaceErrorEvent


def run_output_event_from_dict(data: dict) -> BaseRunOutputEvent:
  event_type = data.get("event", "")
  cls = RUN_EVENT_TYPE_REGISTRY.get(event_type)
  if not cls:
    # Try lazy registration for guardrail events
    _ensure_guardrail_events_registered()
    _ensure_interface_events_registered()
    cls = RUN_EVENT_TYPE_REGISTRY.get(event_type)
  if not cls:
    raise ValueError(f"Unknown event type: {event_type}")
  return cls.from_dict(data)  # type: ignore


@dataclass
class RunOutput:
  """Response returned by Agent.run()"""

  run_id: Optional[str] = None
  agent_id: Optional[str] = None
  agent_name: Optional[str] = None
  session_id: Optional[str] = None
  parent_run_id: Optional[str] = None
  user_id: Optional[str] = None

  # Input media and messages from user
  input: Optional[RunInput] = None

  content: Optional[Any] = None
  parsed: Optional[Any] = None
  content_type: str = "str"

  reasoning_content: Optional[str] = None
  reasoning_steps: Optional[List[ReasoningStep]] = None
  reasoning_messages: Optional[List[Message]] = None

  model_provider_data: Optional[Dict[str, Any]] = None

  model: Optional[str] = None
  model_provider: Optional[str] = None
  messages: Optional[List[Message]] = None
  metrics: Optional[Metrics] = None
  additional_input: Optional[List[Message]] = None

  tools: Optional[List[ToolExecution]] = None

  images: Optional[List[Image]] = None  # Images attached to the response
  videos: Optional[List[Video]] = None  # Videos attached to the response
  audio: Optional[List[Audio]] = None  # Audio attached to the response
  files: Optional[List[File]] = None  # Files attached to the response
  response_audio: Optional[Audio] = None  # Model audio response

  citations: Optional[Citations] = None
  references: Optional[List[MessageReferences]] = None

  metadata: Optional[Dict[str, Any]] = None
  session_state: Optional[Dict[str, Any]] = None

  created_at: int = field(default_factory=lambda: int(time()))

  events: Optional[List[RunOutputEvent]] = None

  # Pipeline phase metrics (timing, skip status)
  phase_metrics: Optional[List["PhaseMetric"]] = None

  status: RunStatus = RunStatus.running

  # User control flow (HITL) requirements to continue a run when paused, in order of arrival
  requirements: Optional[list[RunRequirement]] = None

  @property
  def active_requirements(self) -> list[RunRequirement]:
    if not self.requirements:
      return []
    return [requirement for requirement in self.requirements if not requirement.is_resolved()]

  @property
  def is_paused(self):
    return self.status == RunStatus.paused

  @property
  def is_cancelled(self):
    return self.status == RunStatus.cancelled

  @property
  def tools_requiring_confirmation(self):
    return [t for t in self.tools if t.requires_confirmation] if self.tools else []

  @property
  def tools_requiring_user_input(self):
    return [t for t in self.tools if t.requires_user_input] if self.tools else []

  @property
  def tools_awaiting_external_execution(self):
    return [t for t in self.tools if t.external_execution_required] if self.tools else []

  def to_dict(self) -> Dict[str, Any]:
    _dict = {
      k: v
      for k, v in asdict(self).items()
      if v is not None
      and k
      not in [
        "messages",
        "metrics",
        "tools",
        "metadata",
        "images",
        "videos",
        "audio",
        "files",
        "response_audio",
        "input",
        "citations",
        "events",
        "additional_input",
        "reasoning_steps",
        "reasoning_messages",
        "references",
        "requirements",
      ]
    }

    if self.metrics is not None:
      _dict["metrics"] = self.metrics.to_dict() if isinstance(self.metrics, Metrics) else self.metrics

    if self.events is not None:
      _dict["events"] = [e.to_dict() for e in self.events]

    if self.status is not None:
      _dict["status"] = self.status.value if isinstance(self.status, RunStatus) else self.status

    if self.messages is not None:
      _dict["messages"] = [m.to_dict() for m in self.messages]

    if self.metadata is not None:
      _dict["metadata"] = self.metadata

    if self.additional_input is not None:
      _dict["additional_input"] = [m.to_dict() for m in self.additional_input]

    if self.reasoning_messages is not None:
      _dict["reasoning_messages"] = [m.to_dict() for m in self.reasoning_messages]

    if self.reasoning_steps is not None:
      _dict["reasoning_steps"] = [rs.model_dump() for rs in self.reasoning_steps]

    if self.references is not None:
      _dict["references"] = [r.model_dump() for r in self.references]

    if self.images is not None:
      _dict["images"] = []
      for img in self.images:
        if isinstance(img, Image):
          _dict["images"].append(img.to_dict())
        else:
          _dict["images"].append(img)  # type: ignore[unreachable]

    if self.videos is not None:
      _dict["videos"] = []
      for vid in self.videos:
        if isinstance(vid, Video):
          _dict["videos"].append(vid.to_dict())
        else:
          _dict["videos"].append(vid)  # type: ignore[unreachable]

    if self.audio is not None:
      _dict["audio"] = []
      for aud in self.audio:
        if isinstance(aud, Audio):
          _dict["audio"].append(aud.to_dict())
        else:
          _dict["audio"].append(aud)  # type: ignore[unreachable]

    if self.files is not None:
      _dict["files"] = []
      for file in self.files:
        if isinstance(file, File):
          _dict["files"].append(file.to_dict())
        else:
          _dict["files"].append(file)  # type: ignore[unreachable]

    if self.response_audio is not None:
      if isinstance(self.response_audio, Audio):
        _dict["response_audio"] = self.response_audio.to_dict()
      else:
        _dict["response_audio"] = self.response_audio  # type: ignore[unreachable]

    if self.citations is not None:
      if isinstance(self.citations, Citations):
        _dict["citations"] = self.citations.model_dump(exclude_none=True)
      else:
        _dict["citations"] = self.citations  # type: ignore[unreachable]

    if self.content and isinstance(self.content, BaseModel):
      _dict["content"] = self.content.model_dump(exclude_none=True, mode="json")

    if self.tools is not None:
      _dict["tools"] = []
      for tool in self.tools:
        if isinstance(tool, ToolExecution):
          _dict["tools"].append(tool.to_dict())
        else:
          _dict["tools"].append(tool)  # type: ignore[unreachable]

    if self.requirements is not None:
      _dict["requirements"] = [req.to_dict() if hasattr(req, "to_dict") else req for req in self.requirements]

    if self.input is not None:
      _dict["input"] = self.input.to_dict()

    return _dict

  def to_json(self, separators=(", ", ": "), indent: Optional[int] = 2) -> str:
    import json

    try:
      _dict = self.to_dict()
    except Exception:
      logger.error("Failed to convert response to json", exc_info=True)
      raise

    if indent is None:
      return json.dumps(_dict, separators=separators)
    else:
      return json.dumps(_dict, indent=indent, separators=separators)

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "RunOutput":
    if "run" in data:
      data = data.pop("run")

    events = data.pop("events", None)
    final_events = []
    for event in events or []:
      if "agent_id" in event:
        event = run_output_event_from_dict(event)
      final_events.append(event)
    events = final_events

    messages = data.pop("messages", None)
    messages = [Message.from_dict(message) for message in messages] if messages else None

    citations = data.pop("citations", None)
    citations = Citations.model_validate(citations) if citations else None

    tools = data.pop("tools", [])
    tools = [ToolExecution.from_dict(tool) for tool in tools] if tools else None

    # Handle requirements
    requirements_data = data.pop("requirements", None)
    requirements: Optional[List[RunRequirement]] = None
    if requirements_data is not None:
      requirements_list: List[RunRequirement] = []
      for item in requirements_data:
        if isinstance(item, RunRequirement):
          requirements_list.append(item)
        elif isinstance(item, dict):
          requirements_list.append(RunRequirement.from_dict(item))
      requirements = requirements_list or None

    images = reconstruct_images(data.pop("images", []))
    videos = reconstruct_videos(data.pop("videos", []))
    audio = reconstruct_audio_list(data.pop("audio", []))
    files = reconstruct_files(data.pop("files", []))
    response_audio = reconstruct_response_audio(data.pop("response_audio", None))

    input_data = data.pop("input", None)
    input_obj = None
    if input_data:
      input_obj = RunInput.from_dict(input_data)

    metrics = data.pop("metrics", None)
    if metrics:
      metrics = Metrics(**metrics)

    additional_input = data.pop("additional_input", None)

    if additional_input is not None:
      additional_input = [Message.from_dict(message) for message in additional_input]

    reasoning_steps = data.pop("reasoning_steps", None)
    if reasoning_steps is not None:
      reasoning_steps = [ReasoningStep.model_validate(step) for step in reasoning_steps]

    reasoning_messages = data.pop("reasoning_messages", None)
    if reasoning_messages is not None:
      reasoning_messages = [Message.from_dict(message) for message in reasoning_messages]

    references = data.pop("references", None)
    if references is not None:
      references = [MessageReferences.model_validate(reference) for reference in references]

    # Filter data to only include fields that are actually defined in the RunOutput dataclass
    from dataclasses import fields

    supported_fields = {f.name for f in fields(cls)}
    filtered_data = {k: v for k, v in data.items() if k in supported_fields}

    return cls(
      messages=messages,
      metrics=metrics,
      citations=citations,
      tools=tools,
      images=images,
      audio=audio,
      videos=videos,
      files=files,
      response_audio=response_audio,
      input=input_obj,
      events=events,
      additional_input=additional_input,
      reasoning_steps=reasoning_steps,
      reasoning_messages=reasoning_messages,
      references=references,
      requirements=requirements,
      **filtered_data,
    )

  def get_content_as_string(self, **kwargs) -> str:
    import json

    from pydantic import BaseModel

    if isinstance(self.content, str):
      return self.content
    elif isinstance(self.content, BaseModel):
      return self.content.model_dump_json(exclude_none=True, **kwargs)
    else:
      return json.dumps(self.content, **kwargs)
