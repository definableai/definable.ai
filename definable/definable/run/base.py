from __future__ import annotations

from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Type, Union

from definable.filters import FilterExpr
from definable.media import Audio, Image, Video
from definable.model.message import Citations, Message, MessageReferences
from definable.model.metrics import Metrics
from definable.run.reasoning_step import ReasoningStep
from definable.utils.log import log_error
from pydantic import BaseModel

if TYPE_CHECKING:
  from definable.knowledge.document import Document


@dataclass
class RunContext:
  """
  Context passed through the agent execution pipeline.

  Carries run identifiers, dependencies, state, and knowledge retrieval results.
  """

  run_id: str
  session_id: str
  user_id: Optional[str] = None

  dependencies: Optional[Dict[str, Any]] = None
  knowledge_filters: Optional[Union[Dict[str, Any], List[FilterExpr]]] = None
  metadata: Optional[Dict[str, Any]] = None
  session_state: Optional[Dict[str, Any]] = None
  output_schema: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None

  # Knowledge retrieval results (populated by KnowledgeMiddleware)
  knowledge_context: Optional[str] = None  # Formatted context string for injection
  knowledge_documents: Optional[List[Document]] = None  # Retrieved Document objects

  # Memory retrieval results (populated by Memory)
  memory_context: Optional[str] = None  # Formatted memory payload for injection

  # Deep research results (populated by research pipeline)
  research_context: Optional[str] = None  # Formatted research for system prompt injection
  research_result: Optional[object] = None  # Full ResearchResult (for inspection)

  # File reader results (populated by readers module)
  readers_context: Optional[str] = None  # Extracted file content for injection

  # Tracks which layers were actually fetched this turn (e.g. {"knowledge", "memory"})
  active_layers: Set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Ambient RunContext — accessible from any tool without explicit threading
# ---------------------------------------------------------------------------

_CURRENT_RUN_CONTEXT: ContextVar[Optional[RunContext]] = ContextVar("_CURRENT_RUN_CONTEXT", default=None)


def get_current_run_context() -> Optional[RunContext]:
  """Get the RunContext for the currently executing agent run.

  Returns None if called outside of an agent run. This is the
  recommended way for tools to access run context without requiring
  it as an explicit parameter::

      from definable.run.base import get_current_run_context

      @tool
      def my_tool(query: str) -> str:
          ctx = get_current_run_context()
          if ctx:
              print(f"Running in session {ctx.session_id}")
          return f"Result for {query}"
  """
  return _CURRENT_RUN_CONTEXT.get()


def set_current_run_context(ctx: Optional[RunContext]) -> None:
  """Set the ambient RunContext. Called by the agent loop at run start/end.

  Not intended for direct use by tool authors.
  """
  _CURRENT_RUN_CONTEXT.set(ctx)


@dataclass
class BaseRunOutputEvent:
  # Fields that require custom serialization — excluded from the base asdict() pass,
  # then serialized individually in to_dict(). Does NOT include 'content' because
  # content is included by the base pass and only overridden when it's a BaseModel.
  _COMPLEX_FIELDS: frozenset = field(
    default=frozenset({
      "tools",
      "tool",
      "metadata",
      "image",
      "images",
      "videos",
      "audio",
      "response_audio",
      "citations",
      "member_responses",
      "reasoning_messages",
      "reasoning_steps",
      "references",
      "additional_input",
      "session_summary",
      "metrics",
      "run_input",
      "requirements",
    }),
    init=False,
    repr=False,
  )

  def to_dict(self) -> Dict[str, Any]:
    # Build base dict from all dataclass fields, excluding None values and complex fields
    _dict = {k: v for k, v in asdict(self).items() if v is not None and k not in self._COMPLEX_FIELDS}

    # Introspect actual fields on this instance (not hasattr — uses the dataclass definition)
    own_fields = {f.name for f in fields(self)}

    # --- Passthrough fields (include as-is if present and not None) ---
    if "metadata" in own_fields:
      val = self.metadata  # type: ignore[attr-defined]
      if val is not None:
        _dict["metadata"] = val

    # --- List-of-serializable fields (each item has .to_dict() or .model_dump()) ---
    if "additional_input" in own_fields and self.additional_input is not None:  # type: ignore[attr-defined]
      _dict["additional_input"] = [m.to_dict() for m in self.additional_input]  # type: ignore[attr-defined]

    if "reasoning_messages" in own_fields and self.reasoning_messages is not None:  # type: ignore[attr-defined]
      _dict["reasoning_messages"] = [m.to_dict() for m in self.reasoning_messages]  # type: ignore[attr-defined]

    if "reasoning_steps" in own_fields and self.reasoning_steps is not None:  # type: ignore[attr-defined]
      _dict["reasoning_steps"] = [rs.model_dump() for rs in self.reasoning_steps]  # type: ignore[attr-defined]

    if "references" in own_fields and self.references is not None:  # type: ignore[attr-defined]
      _dict["references"] = [r.model_dump() for r in self.references]  # type: ignore[attr-defined]

    if "member_responses" in own_fields and self.member_responses:  # type: ignore[attr-defined]
      _dict["member_responses"] = [response.to_dict() for response in self.member_responses]  # type: ignore[attr-defined]

    if "requirements" in own_fields and self.requirements is not None:  # type: ignore[attr-defined]
      _dict["requirements"] = [  # type: ignore[attr-defined]
        req.to_dict() if callable(getattr(req, "to_dict", None)) else req
        for req in self.requirements  # type: ignore[attr-defined]
      ]

    # --- Media list fields (each item may be a media object or a raw dict) ---
    for media_field, media_type in (("images", Image), ("videos", Video), ("audio", Audio)):
      if media_field in own_fields:
        items = getattr(self, media_field)
        if items is not None:
          _dict[media_field] = [item.to_dict() if isinstance(item, media_type) else item for item in items]

    # --- Single media field ---
    if "response_audio" in own_fields:
      val = self.response_audio  # type: ignore[attr-defined]
      if val is not None:
        _dict["response_audio"] = val.to_dict() if isinstance(val, Audio) else val

    # --- Pydantic model fields ---
    if "citations" in own_fields:
      val = self.citations  # type: ignore[attr-defined]
      if val is not None:
        _dict["citations"] = val.model_dump(exclude_none=True) if isinstance(val, Citations) else val

    # content is already in _dict from the base pass; override only if it's a BaseModel
    if "content" in own_fields:
      val = self.content  # type: ignore[attr-defined]
      if val is not None and isinstance(val, BaseModel):
        _dict["content"] = val.model_dump(exclude_none=True)

    # --- Tool execution fields ---
    if "tools" in own_fields:
      from definable.model.response import ToolExecution

      val = self.tools  # type: ignore[attr-defined]
      if val is not None:
        _dict["tools"] = [t.to_dict() if isinstance(t, ToolExecution) else t for t in val]

    if "tool" in own_fields:
      from definable.model.response import ToolExecution

      val = self.tool  # type: ignore[attr-defined]
      if val is not None:
        _dict["tool"] = val.to_dict() if isinstance(val, ToolExecution) else val

    # --- Objects with .to_dict() ---
    for obj_field in ("metrics", "session_summary", "run_input"):
      if obj_field in own_fields:
        val = getattr(self, obj_field)
        if val is not None:
          _dict[obj_field] = val.to_dict()

    return _dict

  def to_json(self, separators=(", ", ": "), indent: Optional[int] = 2) -> str:
    import json

    from definable.utils.serialize import json_serializer

    try:
      _dict = self.to_dict()
    except Exception:
      log_error("Failed to convert response event to json", exc_info=True)
      raise

    if indent is None:
      return json.dumps(_dict, separators=separators, default=json_serializer, ensure_ascii=False)
    else:
      return json.dumps(_dict, indent=indent, separators=separators, default=json_serializer, ensure_ascii=False)

  @classmethod
  def from_dict(cls, data: Dict[str, Any]):
    run_input = data.pop("run_input", None)
    if run_input:
      from definable.run.agent import RunInput

      data["run_input"] = RunInput.from_dict(run_input)
    tool = data.pop("tool", None)
    if tool:
      from definable.model.response import ToolExecution

      data["tool"] = ToolExecution.from_dict(tool)

    images = data.pop("images", None)
    if images:
      data["images"] = [Image.model_validate(image) for image in images]

    videos = data.pop("videos", None)
    if videos:
      data["videos"] = [Video.model_validate(video) for video in videos]

    audio = data.pop("audio", None)
    if audio:
      data["audio"] = [Audio.model_validate(audio) for audio in audio]

    response_audio = data.pop("response_audio", None)
    if response_audio:
      data["response_audio"] = Audio.model_validate(response_audio)

    additional_input = data.pop("additional_input", None)
    if additional_input is not None:
      data["additional_input"] = [Message.model_validate(message) for message in additional_input]

    reasoning_steps = data.pop("reasoning_steps", None)
    if reasoning_steps is not None:
      data["reasoning_steps"] = [ReasoningStep.model_validate(step) for step in reasoning_steps]

    reasoning_messages = data.pop("reasoning_messages", None)
    if reasoning_messages is not None:
      data["reasoning_messages"] = [Message.model_validate(message) for message in reasoning_messages]

    references = data.pop("references", None)
    if references is not None:
      data["references"] = [MessageReferences.model_validate(reference) for reference in references]

    metrics = data.pop("metrics", None)
    if metrics:
      data["metrics"] = Metrics(**metrics)

    # Filter data to only include fields that are actually defined in the target class.
    # Skip init=False fields (e.g. _COMPLEX_FIELDS class config) — passing them as kwargs
    # would fail the constructor.
    from dataclasses import fields

    supported_fields = {f.name for f in fields(cls) if f.init}
    filtered_data = {k: v for k, v in data.items() if k in supported_fields}

    return cls(**filtered_data)

  @property
  def is_paused(self):
    return False

  @property
  def is_cancelled(self):
    return False


class RunStatus(str, Enum):
  """State of the main run response"""

  pending = "PENDING"
  running = "RUNNING"
  completed = "COMPLETED"
  paused = "PAUSED"
  cancelled = "CANCELLED"
  blocked = "BLOCKED"
  error = "ERROR"
