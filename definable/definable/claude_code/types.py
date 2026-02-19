"""Claude Code CLI JSONL protocol message types.

Mirrors the stream-json protocol used by the `claude` CLI.
No external dependencies — only stdlib + dataclasses.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from definable.utils.log import log_warning


# ---------------------------------------------------------------------------
# Content blocks (inside AssistantMessage.content)
# ---------------------------------------------------------------------------


@dataclass
class TextBlock:
  type: str = "text"
  text: str = ""


@dataclass
class ThinkingBlock:
  type: str = "thinking"
  thinking: str = ""
  signature: str = ""


@dataclass
class ToolUseBlock:
  type: str = "tool_use"
  id: str = ""
  name: str = ""
  input: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResultBlock:
  type: str = "tool_result"
  tool_use_id: str = ""
  content: Optional[Union[str, List[Dict[str, Any]]]] = None
  is_error: bool = False


ContentBlock = Union[TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock]


# ---------------------------------------------------------------------------
# Top-level message types
# ---------------------------------------------------------------------------


@dataclass
class AssistantMessage:
  """Claude's response containing content blocks."""

  content: List[ContentBlock] = field(default_factory=list)
  model: str = ""


@dataclass
class SystemMessage:
  """System-level messages from the CLI (init, status, etc.)."""

  subtype: str = ""
  data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResultMessage:
  """Final message indicating run completion with metrics."""

  subtype: str = ""  # "success" | "error"
  session_id: str = ""
  duration_ms: int = 0
  duration_api_ms: int = 0
  is_error: bool = False
  turn_count: int = 0
  total_cost_usd: Optional[float] = None
  input_tokens: int = 0
  output_tokens: int = 0
  result: Optional[str] = None
  structured_output: Any = None


@dataclass
class StreamEvent:
  """Partial content during streaming."""

  uuid: str = ""
  session_id: str = ""
  event: Dict[str, Any] = field(default_factory=dict)
  parent_tool_use_id: Optional[str] = None


@dataclass
class ControlRequest:
  """Control protocol request from CLI (permissions, hooks, MCP tool calls)."""

  id: str = ""
  subtype: str = ""  # "can_use_tool" | "hook_callback" | "tool_call"
  tool_name: Optional[str] = None
  input: Optional[Dict[str, Any]] = None


@dataclass
class UserMessage:
  """Echo of user input from the CLI."""

  content: str = ""


Message = Union[AssistantMessage, SystemMessage, ResultMessage, StreamEvent, ControlRequest, UserMessage]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_content_block(block: dict) -> ContentBlock:
  """Parse a single content block from the CLI."""
  block_type = block.get("type", "")
  if block_type == "text":
    return TextBlock(text=block.get("text", ""))
  elif block_type == "thinking":
    return ThinkingBlock(
      thinking=block.get("thinking", ""),
      signature=block.get("signature", ""),
    )
  elif block_type == "tool_use":
    return ToolUseBlock(
      id=block.get("id", ""),
      name=block.get("name", ""),
      input=block.get("input", {}),
    )
  elif block_type == "tool_result":
    return ToolResultBlock(
      tool_use_id=block.get("tool_use_id", ""),
      content=block.get("content"),
      is_error=block.get("is_error", False),
    )
  # Fallback: treat as text
  return TextBlock(text=block.get("text", str(block)))


def _parse_assistant(data: dict) -> AssistantMessage:
  msg = data.get("message", {})
  raw_content = msg.get("content", [])
  blocks = [_parse_content_block(b) for b in raw_content] if isinstance(raw_content, list) else []
  return AssistantMessage(
    content=blocks,
    model=msg.get("model", ""),
  )


def _parse_result(data: dict) -> ResultMessage:
  # Tokens may be top-level or nested under "usage"
  usage = data.get("usage", {})
  input_tokens = data.get("input_tokens", 0) or usage.get("input_tokens", 0)
  output_tokens = data.get("output_tokens", 0) or usage.get("output_tokens", 0)
  # Include cache tokens in total input count
  cache_read = usage.get("cache_read_input_tokens", 0)
  cache_create = usage.get("cache_creation_input_tokens", 0)
  if cache_read or cache_create:
    input_tokens = input_tokens + cache_read + cache_create

  return ResultMessage(
    subtype=data.get("subtype", ""),
    session_id=data.get("session_id", ""),
    duration_ms=data.get("duration_ms", 0),
    duration_api_ms=data.get("duration_api_ms", 0),
    is_error=data.get("is_error", False),
    turn_count=data.get("num_turns", data.get("turn_count", 0)),
    total_cost_usd=data.get("total_cost_usd"),
    input_tokens=input_tokens,
    output_tokens=output_tokens,
    result=data.get("result"),
    structured_output=data.get("structured_output"),
  )


def _parse_system(data: dict) -> SystemMessage:
  return SystemMessage(
    subtype=data.get("subtype", ""),
    data={k: v for k, v in data.items() if k not in ("type", "subtype")},
  )


def _parse_stream_event(data: dict) -> StreamEvent:
  return StreamEvent(
    uuid=data.get("uuid", ""),
    session_id=data.get("session_id", ""),
    event=data.get("event", {}),
    parent_tool_use_id=data.get("parent_tool_use_id"),
  )


def _parse_control_request(data: dict) -> ControlRequest:
  return ControlRequest(
    id=data.get("id", ""),
    subtype=data.get("subtype", ""),
    tool_name=data.get("tool_name"),
    input=data.get("input"),
  )


def _parse_user(data: dict) -> UserMessage:
  msg = data.get("message", {})
  content = msg.get("content", "")
  if isinstance(content, list):
    # Content may be a list of blocks — extract text
    parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
    content = "".join(parts)
  return UserMessage(content=content if isinstance(content, str) else str(content))


def parse_message(data: dict) -> Message:
  """Parse a raw JSONL dict into a typed Message.

  Raises ValueError for unknown message types.
  """
  msg_type = data.get("type")
  if msg_type == "assistant":
    return _parse_assistant(data)
  elif msg_type == "result":
    return _parse_result(data)
  elif msg_type == "system":
    return _parse_system(data)
  elif msg_type == "stream_event":
    return _parse_stream_event(data)
  elif msg_type == "control_request":
    return _parse_control_request(data)
  elif msg_type == "user":
    return _parse_user(data)
  log_warning(f"Unknown CLI message type: {msg_type}")
  raise ValueError(f"Unknown message type: {msg_type}")
