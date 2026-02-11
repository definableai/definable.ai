from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

from definable.filters import FilterExpr
from definable.media import Audio, Image, Video
from definable.models.message import Citations, Message, MessageReferences
from definable.models.metrics import Metrics
from definable.reasoning.step import ReasoningStep
from definable.utils.log import log_error
from pydantic import BaseModel


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
  knowledge_documents: Optional[List[Any]] = None  # Retrieved Document objects

  # Memory retrieval results (populated by CognitiveMemory)
  memory_context: Optional[str] = None  # Formatted memory payload for injection

  # File reader results (populated by readers module)
  readers_context: Optional[str] = None  # Extracted file content for injection


@dataclass
class BaseRunOutputEvent:
  def to_dict(self) -> Dict[str, Any]:
    _dict = {
      k: v
      for k, v in asdict(self).items()
      if v is not None
      and k
      not in [
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
      ]
    }

    if hasattr(self, "metadata") and self.metadata is not None:
      _dict["metadata"] = self.metadata

    if hasattr(self, "additional_input") and self.additional_input is not None:
      _dict["additional_input"] = [m.to_dict() for m in self.additional_input]

    if hasattr(self, "reasoning_messages") and self.reasoning_messages is not None:
      _dict["reasoning_messages"] = [m.to_dict() for m in self.reasoning_messages]

    if hasattr(self, "reasoning_steps") and self.reasoning_steps is not None:
      _dict["reasoning_steps"] = [rs.model_dump() for rs in self.reasoning_steps]

    if hasattr(self, "references") and self.references is not None:
      _dict["references"] = [r.model_dump() for r in self.references]

    if hasattr(self, "member_responses") and self.member_responses:
      _dict["member_responses"] = [response.to_dict() for response in self.member_responses]

    if hasattr(self, "images") and self.images is not None:
      _dict["images"] = []
      for img in self.images:
        if isinstance(img, Image):
          _dict["images"].append(img.to_dict())
        else:
          _dict["images"].append(img)

    if hasattr(self, "videos") and self.videos is not None:
      _dict["videos"] = []
      for vid in self.videos:
        if isinstance(vid, Video):
          _dict["videos"].append(vid.to_dict())
        else:
          _dict["videos"].append(vid)

    if hasattr(self, "audio") and self.audio is not None:
      _dict["audio"] = []
      for aud in self.audio:
        if isinstance(aud, Audio):
          _dict["audio"].append(aud.to_dict())
        else:
          _dict["audio"].append(aud)

    if hasattr(self, "response_audio") and self.response_audio is not None:
      if isinstance(self.response_audio, Audio):
        _dict["response_audio"] = self.response_audio.to_dict()
      else:
        _dict["response_audio"] = self.response_audio

    if hasattr(self, "citations") and self.citations is not None:
      if isinstance(self.citations, Citations):
        _dict["citations"] = self.citations.model_dump(exclude_none=True)
      else:
        _dict["citations"] = self.citations

    if hasattr(self, "content") and self.content and isinstance(self.content, BaseModel):
      _dict["content"] = self.content.model_dump(exclude_none=True)

    if hasattr(self, "tools") and self.tools is not None:
      from definable.models.response import ToolExecution

      _dict["tools"] = []
      for tool in self.tools:
        if isinstance(tool, ToolExecution):
          _dict["tools"].append(tool.to_dict())
        else:
          _dict["tools"].append(tool)

    if hasattr(self, "tool") and self.tool is not None:
      from definable.models.response import ToolExecution

      if isinstance(self.tool, ToolExecution):
        _dict["tool"] = self.tool.to_dict()
      else:
        _dict["tool"] = self.tool

    if hasattr(self, "metrics") and self.metrics is not None:
      _dict["metrics"] = self.metrics.to_dict()

    if hasattr(self, "session_summary") and self.session_summary is not None:
      _dict["session_summary"] = self.session_summary.to_dict()

    if hasattr(self, "run_input") and self.run_input is not None:
      _dict["run_input"] = self.run_input.to_dict()

    if hasattr(self, "requirements") and self.requirements is not None:
      _dict["requirements"] = [req.to_dict() if hasattr(req, "to_dict") else req for req in self.requirements]

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
      from definable.models.response import ToolExecution

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

    # Filter data to only include fields that are actually defined in the target class
    from dataclasses import fields

    supported_fields = {f.name for f in fields(cls)}
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
  error = "ERROR"
