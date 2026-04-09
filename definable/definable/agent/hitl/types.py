"""Shared types for the HITL (Human-in-the-Loop) system.

Two independent subsystems:
  - **Permission**: gates tool execution with allow/deny/always-allow.
  - **Question**: agent asks the user questions mid-run via a regular tool.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol


# ---------------------------------------------------------------------------
# Permission types
# ---------------------------------------------------------------------------


class PermissionAction(str, Enum):
  """Default permission policy for a tool (set by agent creator or settings)."""

  allow = "allow"
  deny = "deny"
  ask = "ask"


class PermissionDecision(str, Enum):
  """User's response to a permission prompt."""

  allow_once = "allow_once"
  allow_always = "allow_always"
  deny = "deny"


@dataclass(frozen=True)
class PermissionRequest:
  """Sent to the resolver when a tool needs permission."""

  tool_name: str
  tool_args: Dict[str, Any]
  tool_call_id: Optional[str] = None


@dataclass(frozen=True)
class PermissionResponse:
  """Returned by the resolver."""

  decision: PermissionDecision
  feedback: Optional[str] = None


class PermissionResolver(Protocol):
  """Async callback that resolves permission requests.

  Implementors: CLI prompt, web socket handler, test stub, etc.
  """

  async def __call__(self, request: PermissionRequest) -> PermissionResponse: ...


# ---------------------------------------------------------------------------
# Question types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuestionOption:
  """A single option in a question."""

  label: str
  description: Optional[str] = None


@dataclass(frozen=True)
class Question:
  """A single question the agent wants to ask."""

  text: str
  header: Optional[str] = None
  options: Optional[List[QuestionOption]] = None
  allow_multiple: bool = False
  allow_custom: bool = False


@dataclass(frozen=True)
class Answer:
  """User's answer to a question."""

  question_text: str
  selected: List[str] = field(default_factory=list)
  custom_text: Optional[str] = None


class QuestionResolver(Protocol):
  """Async callback that resolves questions from the agent.

  Receives a list of questions, returns a list of answers (same order).
  """

  async def __call__(self, questions: List[Question]) -> List[Answer]: ...
