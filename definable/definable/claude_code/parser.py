"""Parser — converts Claude Code CLI messages into Definable RunOutput and events.

Maps the CLI's JSONL protocol messages to Definable's native types so that
the rest of the framework (interfaces, tracing, middleware) works seamlessly.
"""

import json
from typing import Any, Dict, List, Optional

from definable.claude_code.types import (
  AssistantMessage,
  Message,
  ResultMessage,
  StreamEvent,
  TextBlock,
  ThinkingBlock,
  ToolResultBlock,
  ToolUseBlock,
)
from definable.model.metrics import Metrics
from definable.model.response import ToolExecution
from definable.agent.events import (
  BaseAgentRunEvent,
  ReasoningContentDeltaEvent,
  RunCompletedEvent,
  RunContentEvent,
  RunContext,
  RunOutput,
  RunStatus,
  ToolCallCompletedEvent,
  ToolCallStartedEvent,
)


def parse_to_run_output(
  messages: List[Message],
  context: RunContext,
  model_id: str,
  agent_id: Optional[str] = None,
  agent_name: Optional[str] = None,
) -> RunOutput:
  """Convert a sequence of CLI messages into a Definable RunOutput.

  Extracts text content, reasoning/thinking blocks, tool executions,
  metrics, structured output, and session information.
  """
  content_parts: List[str] = []
  tool_executions: List[ToolExecution] = []
  reasoning_content: Optional[str] = None
  metrics = Metrics()
  session_id: Optional[str] = None
  structured_output: Any = None
  is_error = False

  for msg in messages:
    if isinstance(msg, AssistantMessage):
      for block in msg.content:
        if isinstance(block, TextBlock):
          if block.text:
            content_parts.append(block.text)

        elif isinstance(block, ThinkingBlock):
          if block.thinking:
            reasoning_content = (reasoning_content or "") + block.thinking

        elif isinstance(block, ToolUseBlock):
          tool_executions.append(
            ToolExecution(
              tool_name=block.name,
              tool_args=block.input,
              tool_call_id=block.id,
            )
          )

        elif isinstance(block, ToolResultBlock):
          # Match result back to its tool execution
          for te in tool_executions:
            if te.tool_call_id == block.tool_use_id:
              if isinstance(block.content, str):
                te.result = block.content
              elif block.content is not None:
                te.result = json.dumps(block.content)
              te.tool_call_error = block.is_error or None
              break

    elif isinstance(msg, ResultMessage):
      session_id = msg.session_id
      is_error = msg.is_error
      metrics.input_tokens = msg.input_tokens
      metrics.output_tokens = msg.output_tokens
      metrics.total_tokens = msg.input_tokens + msg.output_tokens
      if msg.duration_ms:
        metrics.duration = msg.duration_ms / 1000.0
      if msg.total_cost_usd is not None:
        metrics.cost = msg.total_cost_usd
      structured_output = msg.structured_output
      # Use result.result as fallback text if no assistant text blocks found
      if msg.result and not content_parts:
        content_parts.append(msg.result)

    elif isinstance(msg, StreamEvent):
      # Extract text deltas from stream events
      event_data = msg.event
      if event_data.get("type") == "content_block_delta":
        delta = event_data.get("delta", {})
        if delta.get("type") == "text_delta":
          text = delta.get("text", "")
          if text:
            content_parts.append(text)
        elif delta.get("type") == "thinking_delta":
          thinking = delta.get("thinking", "")
          if thinking:
            reasoning_content = (reasoning_content or "") + thinking

  # Build final content
  final_content: Any = structured_output
  if final_content is None:
    joined = "".join(content_parts)
    final_content = joined or None

  return RunOutput(
    run_id=context.run_id,
    session_id=session_id or context.session_id,
    agent_id=agent_id,
    agent_name=agent_name,
    user_id=context.user_id,
    content=final_content,
    content_type="str",
    reasoning_content=reasoning_content,
    tools=tool_executions or None,
    metrics=metrics,
    model=model_id,
    model_provider="Anthropic",
    status=RunStatus.error if is_error else RunStatus.completed,
  )


def message_to_events(
  msg: Message,
  context: RunContext,
  agent_id: str = "",
  agent_name: str = "",
) -> List[BaseAgentRunEvent]:
  """Convert a single CLI message to a list of Definable tracing events.

  Returns an empty list for messages that don't map to events (e.g. SystemMessage, UserMessage).
  Emits one event per content block in AssistantMessages (not just the first).
  """
  base_kwargs: Dict[str, Any] = {
    "agent_id": agent_id,
    "agent_name": agent_name,
    "run_id": context.run_id,
    "session_id": context.session_id,
  }

  events: List[BaseAgentRunEvent] = []

  if isinstance(msg, AssistantMessage):
    for block in msg.content:
      if isinstance(block, TextBlock) and block.text:
        events.append(RunContentEvent(content=block.text, **base_kwargs))
      elif isinstance(block, ThinkingBlock) and block.thinking:
        events.append(ReasoningContentDeltaEvent(reasoning_content=block.thinking, **base_kwargs))
      elif isinstance(block, ToolUseBlock):
        events.append(
          ToolCallStartedEvent(
            tool=ToolExecution(
              tool_name=block.name,
              tool_args=block.input,
              tool_call_id=block.id,
            ),
            **base_kwargs,
          )
        )
      elif isinstance(block, ToolResultBlock):
        # Match result back to a tool execution
        result_text = block.content if isinstance(block.content, str) else json.dumps(block.content) if block.content else ""
        events.append(
          ToolCallCompletedEvent(
            tool=ToolExecution(
              tool_call_id=block.tool_use_id,
              result=result_text,
              tool_call_error=block.is_error or None,
            ),
            content=result_text,
            **base_kwargs,
          )
        )

  elif isinstance(msg, ResultMessage):
    metrics = Metrics(
      input_tokens=msg.input_tokens,
      output_tokens=msg.output_tokens,
      total_tokens=msg.input_tokens + msg.output_tokens,
    )
    if msg.duration_ms:
      metrics.duration = msg.duration_ms / 1000.0
    if msg.total_cost_usd is not None:
      metrics.cost = msg.total_cost_usd
    events.append(RunCompletedEvent(metrics=metrics, content=msg.result, **base_kwargs))

  elif isinstance(msg, StreamEvent):
    event_data = msg.event
    if event_data.get("type") == "content_block_delta":
      delta = event_data.get("delta", {})
      if delta.get("type") == "text_delta":
        events.append(RunContentEvent(content=delta.get("text", ""), **base_kwargs))
      elif delta.get("type") == "thinking_delta":
        events.append(ReasoningContentDeltaEvent(reasoning_content=delta.get("thinking", ""), **base_kwargs))

  # SystemMessage, ControlRequest, UserMessage don't map to events
  return events


# Backward compat alias — returns first event or None
def message_to_event(
  msg: Message,
  context: RunContext,
  agent_id: str = "",
  agent_name: str = "",
) -> Optional[BaseAgentRunEvent]:
  """Convert a single CLI message to the first matching Definable event (legacy)."""
  events = message_to_events(msg, context, agent_id=agent_id, agent_name=agent_name)
  return events[0] if events else None
