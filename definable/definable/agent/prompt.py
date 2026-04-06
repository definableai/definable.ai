"""Prompt assembly extracted from Agent — system prompt construction, layer guide, skill instructions.

Every function here was an Agent method.  The only change is
``self`` → ``agent: "Agent"`` (first parameter) and
``self.X`` → ``agent.X`` everywhere.
``@staticmethod`` methods become plain functions without ``agent`` param.
"""

from typing import (
  TYPE_CHECKING,
  Dict,
  List,
  Optional,
  Union,
)

from definable.agent.events import RunContext
from definable.model.message import Message
from definable.tool.function import Function

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.agent.reasoning.step import ReasoningStep, ThinkingOutput


# ---------------------------------------------------------------------------
# Layer guide
# ---------------------------------------------------------------------------


def build_layer_guide(agent: "Agent", context: Optional[RunContext] = None) -> str:
  """Build a capabilities menu section for the system prompt.

  Only included when at least one layer has a custom description
  or a non-default trigger (i.e., the model guides activation).

  When ``context`` is provided, reflects the actual fetch state:
  layers that were retrieved are marked as such; layers that were
  configured with trigger="auto" but not fetched this turn are noted
  as available-but-not-retrieved.
  """
  from definable.agent.config import (
    DEFAULT_KNOWLEDGE_DESCRIPTION,
    DEFAULT_RESEARCH_DESCRIPTION,
    DEFAULT_THINKING_DESCRIPTION,
  )

  active = context.active_layers if context is not None else set()
  items: List[str] = []

  # Memory layer
  if agent.memory and agent.memory.enabled:
    if agent.memory.description:
      desc = agent.memory.description
      if "memory" in active:
        items.append(f"- **Memory** [retrieved this turn]: {desc}")
      else:
        items.append(f"- **Memory**: {desc}")

  # Knowledge layer
  if agent._knowledge:
    needs_guide = bool(agent._knowledge.description) or agent._knowledge.trigger != "always"
    if needs_guide:
      desc = agent._knowledge.description or DEFAULT_KNOWLEDGE_DESCRIPTION
      if "knowledge" in active:
        items.append(f"- **Knowledge Base** [retrieved this turn]: {desc}")
      elif agent._knowledge.trigger == "auto":
        items.append(f"- **Knowledge Base** [available, not retrieved this turn]: {desc}")
      else:
        items.append(f"- **Knowledge Base**: {desc}")

  # Thinking layer
  if agent._thinking and agent._thinking.enabled:
    needs_guide = bool(agent._thinking.description) or agent._thinking.trigger != "always"
    if needs_guide:
      desc = agent._thinking.description or DEFAULT_THINKING_DESCRIPTION
      items.append(f"- **Analysis**: {desc}")

  # Deep research layer
  if agent._researcher and agent._deep_research_config:
    needs_guide = bool(agent._deep_research_config.description) or agent._deep_research_config.trigger != "always"
    if needs_guide:
      desc = agent._deep_research_config.description or DEFAULT_RESEARCH_DESCRIPTION
      items.append(f"- **Research**: {desc}")

  if not items:
    return ""

  lines = [
    "## Capabilities Available",
    "",
    "The following capabilities are available and will activate when relevant:",
    "",
  ] + items
  return "\n".join(lines)


# ---------------------------------------------------------------------------
# Skill instructions
# ---------------------------------------------------------------------------


def build_skill_instructions(agent: "Agent") -> str:
  """Collect instructions from all skills into a merged block.

  Returns:
    A single string with all skill instructions separated by
    blank lines, or empty string if no skills provide instructions.
  """
  parts: List[str] = []
  for skill in agent.skills:
    try:
      text = skill.get_instructions()
    except Exception:
      text = ""
    if text and text.strip():
      parts.append(text.strip())
  return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Reasoning context formatting
# ---------------------------------------------------------------------------


def format_reasoning_context(steps: "list[ReasoningStep]") -> str:
  """Format reasoning steps into XML context for system prompt injection."""
  if not steps:
    return ""

  lines = ["<reasoning>"]
  for i, step in enumerate(steps, 1):
    lines.append(f'  <step number="{i}">')
    if step.title:
      lines.append(f"    <title>{step.title}</title>")
    if step.reasoning:
      lines.append(f"    <reasoning>{step.reasoning}</reasoning>")
    if step.action:
      lines.append(f"    <action>{step.action}</action>")
    if step.confidence is not None:
      lines.append(f"    <confidence>{step.confidence}</confidence>")
    lines.append("  </step>")
  lines.append("</reasoning>")
  return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core prompt builder
# ---------------------------------------------------------------------------


async def build_invoke_messages(
  agent: "Agent",
  context: RunContext,
  messages: List[Message],
  tools: Dict[str, Function],
  *,
  thinking_output: "Optional[ThinkingOutput]" = None,
  thinking_text: Optional[str] = None,
  reasoning_steps: "Optional[list[ReasoningStep]]" = None,
  reasoning_messages: "Optional[list[Message]]" = None,
) -> tuple:
  """Build the invoke message list with system prompt, thinking, knowledge, memory, readers.

  Args:
      agent: The Agent instance.
      context: Current run context.
      messages: Conversation messages.
      tools: Flattened tool dict.
      thinking_output: Pre-computed structured thinking output from ThinkPhase.
      thinking_text: Pre-computed free-form thinking text (non-structured models).
      reasoning_steps: Pre-computed reasoning steps from ThinkPhase.
      reasoning_messages: Pre-computed reasoning messages from ThinkPhase.

  Returns:
      (invoke_messages, reasoning_steps, reasoning_agent_messages)
  """
  # Thinking phase (BEFORE system prompt assembly)
  # If pre-computed thinking is provided (from ThinkPhase), use it directly.
  # Otherwise, compute inline for backward compatibility.
  reasoning_agent_messages = reasoning_messages
  invoke_messages = messages.copy()
  if thinking_output is None and thinking_text is None:
    if await agent._thinking_should_run(invoke_messages):
      if agent._thinking and agent._thinking.should_use_native(agent.model):
        # Native thinking: configure the model, let the loop handle events.
        # No separate call needed — the model will return reasoning_content.
        agent._enable_native_thinking()
      else:
        # Definable's fallback thinking layer (separate LLM call) — inline path
        async for item in agent._execute_thinking(context, invoke_messages, tools):
          if isinstance(item, tuple):
            thinking_output, thinking_text, reasoning_steps, reasoning_agent_messages = item

  # Format thinking injection (used by both code paths)
  thinking_injection: Optional[str] = None
  injection_source: "Union[ThinkingOutput, str, None]" = thinking_output or thinking_text
  if injection_source:
    effort = agent._thinking.effort if agent._thinking and hasattr(agent._thinking, "effort") else "medium"
    from definable.agent.layers import format_thinking_injection

    thinking_injection = format_thinking_injection(injection_source, effort=effort)

  # -- Context Manager path (structured, priority-based) -------------------
  if agent._context_manager is not None:
    # Trim history (tool-pair-aware, async for summarize strategy)
    invoke_messages = await agent._context_manager.atrim_history(invoke_messages)

    # Deferred tools: inject catalog into layer guide, swap tool set
    _layer_guide = build_layer_guide(agent, context)
    if agent._deferred_tool_manager is not None:
      agent._deferred_tool_manager.prepare_for_run()
      catalog = agent._deferred_tool_manager.build_catalog()
      if catalog:
        _layer_guide = f"{_layer_guide}\n\n{catalog}" if _layer_guide else catalog

    # Common kwargs for system prompt assembly
    _prompt_kwargs = dict(
      instructions=agent.instructions,
      skill_instructions=build_skill_instructions(agent),
      layer_guide=_layer_guide,
      thinking_injection=thinking_injection,
      knowledge_context=context.knowledge_context if (context.metadata or {}).get("_knowledge_position", "system") == "system" else None,
      research_context=context.research_context,
      memory_context=context.memory_context,
    )

    # Build system prompt via layered prompt
    system_content = agent._context_manager.build_system_prompt(**_prompt_kwargs)
  # -- Legacy path (flat concatenation) ------------------------------------
  else:
    system_content = agent.instructions or ""

    # Append skill instructions
    skill_instructions = build_skill_instructions(agent)
    if skill_instructions:
      system_content = f"{system_content}\n\n{skill_instructions}" if system_content else skill_instructions

    # Layer guide
    layer_guide = build_layer_guide(agent, context)
    if layer_guide:
      system_content = f"{system_content}\n\n{layer_guide}" if system_content else layer_guide

    # Inject thinking results into system prompt
    if thinking_injection:
      system_content = f"{system_content}\n\n{thinking_injection}" if system_content else thinking_injection

    # Append knowledge context
    if context.knowledge_context:
      position = (context.metadata or {}).get("_knowledge_position", "system")
      if position == "system":
        system_content = f"{system_content}\n\n{context.knowledge_context}" if system_content else context.knowledge_context

    # Append research context
    if context.research_context:
      system_content = f"{system_content}\n\n{context.research_context}" if system_content else context.research_context

    # Append memory context
    if context.memory_context:
      system_content = f"{system_content}\n\n{context.memory_context}" if system_content else context.memory_context

  if system_content:
    system_msg = Message(role="system", content=system_content)
    # Cache optimization: attach static/dynamic split for Claude adapter
    if agent._context_manager is not None and agent._context_manager.config.cache_optimization:
      static, dynamic = agent._context_manager.build_system_prompt_split(**_prompt_kwargs)  # type: ignore[name-defined]
      if static:
        system_msg._cache_blocks = [  # type: ignore[attr-defined]
          {"text": static, "type": "text", "cache_control": {"type": "ephemeral"}},
          {"text": dynamic, "type": "text"},
        ]
    invoke_messages.insert(0, system_msg)

  # Inject extracted file content into the last user message
  if context.readers_context:
    for i in range(len(invoke_messages) - 1, -1, -1):
      if invoke_messages[i].role == "user":
        original_content = invoke_messages[i].content or ""
        invoke_messages[i] = Message(
          role="user",
          content=f"{context.readers_context}\n\n{original_content}",
          images=invoke_messages[i].images,
          videos=invoke_messages[i].videos,
          audio=invoke_messages[i].audio,
        )
        break

  return invoke_messages, reasoning_steps, reasoning_agent_messages
