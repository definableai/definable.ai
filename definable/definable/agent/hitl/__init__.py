"""Human-in-the-Loop (HITL) — permission gating and agent-driven questions.

Two independent subsystems:

Permission Service
  Gates tool execution with **Allow / Deny / Always Allow**.
  "Always allow" persists to ``.definable/settings.json``.

Question Tool
  A regular ``@tool`` the agent calls to ask the user questions
  with multiple-choice options.  No special loop mechanics.

Quick start::

    from definable.agent.hitl import (
        PermissionAction,
        PermissionDecision,
        PermissionRequest,
        PermissionResponse,
        PermissionService,
        Question,
        Answer,
        Settings,
        build_ask_user_tool,
    )
"""

from definable.agent.hitl.permissions import PermissionService
from definable.agent.hitl.question import build_ask_user_tool
from definable.agent.hitl.settings import Settings
from definable.agent.hitl.types import (
  Answer,
  PermissionAction,
  PermissionDecision,
  PermissionRequest,
  PermissionResolver,
  PermissionResponse,
  Question,
  QuestionOption,
  QuestionResolver,
)

__all__ = [
  "Answer",
  "PermissionAction",
  "PermissionDecision",
  "PermissionRequest",
  "PermissionResolver",
  "PermissionResponse",
  "PermissionService",
  "Question",
  "QuestionOption",
  "QuestionResolver",
  "Settings",
  "build_ask_user_tool",
]
