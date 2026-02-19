"""Guardrails & Firewall Layer — check, modify, or block content at three checkpoints.

Checkpoints:
  - **input**: Before the LLM call (user message text).
  - **output**: After the LLM call (model response text).
  - **tool**: Before each tool execution (tool name + args).

Quick Start::

    from definable.agent import Agent
    from definable.agent.guardrail import Guardrails, max_tokens, pii_filter, tool_blocklist

    agent = Agent(
        model=model,
        guardrails=Guardrails(
            input=[max_tokens(500)],
            output=[pii_filter()],
            tool=[tool_blocklist({"dangerous_tool"})],
        ),
    )
"""

from definable.agent.guardrail.base import (
  Guardrails,
  GuardrailResult,
  InputGuardrail,
  OutputGuardrail,
  ToolGuardrail,
)
from definable.agent.guardrail.builtin import (
  block_topics,
  max_output_tokens,
  max_tokens,
  pii_filter,
  regex_filter,
  tool_allowlist,
  tool_blocklist,
)
from definable.agent.guardrail.composable import ALL, ANY, NOT, when
from definable.agent.guardrail.decorators import input_guardrail, output_guardrail, tool_guardrail
from definable.agent.guardrail.events import GuardrailBlockedEvent, GuardrailCheckedEvent

__all__ = [
  # Core types
  "GuardrailResult",
  "InputGuardrail",
  "OutputGuardrail",
  "ToolGuardrail",
  "Guardrails",
  # Decorators
  "input_guardrail",
  "output_guardrail",
  "tool_guardrail",
  # Composable
  "ALL",
  "ANY",
  "NOT",
  "when",
  # Built-ins
  "max_tokens",
  "block_topics",
  "regex_filter",
  "pii_filter",
  "max_output_tokens",
  "tool_allowlist",
  "tool_blocklist",
  # Events
  "GuardrailCheckedEvent",
  "GuardrailBlockedEvent",
]
