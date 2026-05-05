"""Layer operations extracted from Agent — knowledge, memory, research, readers, guardrails, thinking, orchestration.

Every function here was an Agent method.  The only change is
``self`` → ``agent: "Agent"`` (first parameter) and
``self.X`` → ``agent.X`` everywhere.
"""

import asyncio
import json
from typing import (
  TYPE_CHECKING,
  Any,
  AsyncGenerator,
  Awaitable,
  Callable,
  Dict,
  List,
  Literal,
  Optional,
  Union,
)

from definable.agent.events import (
  DeepResearchCompletedEvent,
  DeepResearchStartedEvent,
  FileReadCompletedEvent,
  FileReadStartedEvent,
  KnowledgeRetrievalCompletedEvent,
  KnowledgeRetrievalStartedEvent,
  MemoryRecallCompletedEvent,
  MemoryRecallStartedEvent,
  MemoryUpdateCompletedEvent,
  MemoryUpdateStartedEvent,
  RunContext,
  RunOutput,
  RunOutputEvent,
  RunStatus,
)
from definable.model.message import Message
from definable.tool.function import Function

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.agent.reasoning.step import ReasoningStep, ThinkingOutput
  from definable.agent.research.config import DeepResearchConfig
  from definable.agent.research.engine import DeepResearch
  from definable.model.base import Model
  from definable.model.response import ToolExecution


# ---------------------------------------------------------------------------
# Knowledge
# ---------------------------------------------------------------------------


async def knowledge_retrieve(agent: "Agent", context: RunContext) -> List[RunOutputEvent]:
  """Retrieve knowledge documents, emit events, inject into context."""
  kc = agent._knowledge
  if not (kc and kc.enabled):
    return []

  messages = context.metadata.get("_messages") if context.metadata else None
  if not messages:
    return []

  from definable.agent.middleware import KnowledgeMiddleware

  km = KnowledgeMiddleware(kc)
  query = km._extract_query(messages)
  if not query:
    return []

  import time

  events: List[RunOutputEvent] = []
  started = KnowledgeRetrievalStartedEvent(
    run_id=context.run_id,
    session_id=context.session_id,
    agent_id=agent.agent_id,
    agent_name=agent.agent_name,
    query=query,
  )
  agent._emit(started)
  events.append(started)

  start_time = time.perf_counter()
  try:
    documents = await kc.asearch(
      query=query,
      top_k=kc.top_k,
      rerank=kc.rerank,
    )
  except Exception:
    elapsed = (time.perf_counter() - start_time) * 1000
    completed = KnowledgeRetrievalCompletedEvent(
      run_id=context.run_id,
      session_id=context.session_id,
      agent_id=agent.agent_id,
      agent_name=agent.agent_name,
      query=query,
      documents_found=0,
      documents_used=0,
      duration_ms=elapsed,
    )
    agent._emit(completed)
    events.append(completed)
    return events

  documents_found = len(documents)

  # Filter by min_score
  if kc.min_score is not None:
    documents = [d for d in documents if d.reranking_score is not None and d.reranking_score >= kc.min_score]

  if documents:
    context_text = km._format_context(documents)
    context.knowledge_context = context_text
    context.knowledge_documents = documents
    context.active_layers.add("knowledge")
    if context.metadata is None:
      context.metadata = {}
    context.metadata["_knowledge_position"] = kc.context_position

  elapsed = (time.perf_counter() - start_time) * 1000
  completed = KnowledgeRetrievalCompletedEvent(
    run_id=context.run_id,
    session_id=context.session_id,
    agent_id=agent.agent_id,
    agent_name=agent.agent_name,
    query=query,
    documents_found=documents_found,
    documents_used=len(documents),
    duration_ms=elapsed,
  )
  agent._emit(completed)
  events.append(completed)
  return events


# ---------------------------------------------------------------------------
# Deep Research
# ---------------------------------------------------------------------------


def init_deep_research(agent: "Agent", config: "DeepResearchConfig") -> Optional["DeepResearch"]:
  """Initialize the deep research engine if configured.

  Non-fatal: returns None with a warning if search capability cannot be found.
  """
  from definable.utils.log import log_debug, log_warning

  if not config or not config.enabled:
    return None

  try:
    from definable.agent.research.engine import DeepResearch
    from definable.agent.research.search import create_search_provider

    # Try explicit search_fn or provider first
    if config.search_fn is not None or config.search_provider != "duckduckgo":
      provider = create_search_provider(
        provider=config.search_provider,
        config=config.search_provider_config,
        search_fn=config.search_fn,
      )
    else:
      # Try auto-discovering from WebSearch skill
      provider = discover_search_provider(agent)  # type: ignore[assignment]
      if provider is None:
        # Fall back to DuckDuckGo
        provider = create_search_provider("duckduckgo")  # type: ignore[unreachable]

    compression_model = config.compression_model or agent.model
    log_debug("Deep research engine initialized")
    return DeepResearch(
      model=agent.model,
      search_provider=provider,
      compression_model=compression_model,
      config=config,
    )
  except Exception as e:
    log_warning(f"Failed to initialize deep research: {e}")
    return None


def discover_search_provider(agent: "Agent") -> object:
  """Try to auto-discover a search provider from WebSearch skill."""
  from definable.utils.log import log_debug

  for skill in agent.skills:
    # Check for WebSearch skill with its _search_fn
    skill_cls_name = type(skill).__name__
    if skill_cls_name == "WebSearch" and hasattr(skill, "_search_fn"):
      from definable.agent.research.search import CallableSearchProvider
      from definable.agent.research.search.base import SearchResult

      raw_fn = skill._search_fn

      async def _wrapped(query: str, max_results: int = 10) -> list:
        import asyncio

        text = await asyncio.to_thread(raw_fn, query, max_results)
        # WebSearch._search_fn returns formatted string, not SearchResult list.
        # Parse it back into SearchResult objects.
        results = []
        for block in text.split("\n\n---\n\n"):
          lines = block.strip().split("\n", 2)
          if len(lines) >= 2:
            title = lines[0].strip("*").strip()
            url = lines[1].strip()
            snippet = lines[2] if len(lines) > 2 else ""
            results.append(SearchResult(title=title, url=url, snippet=snippet))
        return results

      log_debug("Auto-discovered search provider from WebSearch skill")
      return CallableSearchProvider(_wrapped)

  return None


async def deep_research(agent: "Agent", context: RunContext) -> List[RunOutputEvent]:
  """Execute deep research pipeline, emit events, inject context."""
  if not agent._researcher:
    return []

  config = agent._deep_research_config or (agent._researcher._config if agent._researcher else None)
  if not config or not config.enabled:
    return []

  # Extract query from last user message
  messages = context.metadata.get("_messages") if context.metadata else None
  if not messages:
    return []

  query = None
  for msg in reversed(messages):
    if hasattr(msg, "role") and msg.role == "user" and msg.content:
      query = msg.content if isinstance(msg.content, str) else str(msg.content)
      break
  if not query:
    return []

  # Auto trigger: ask model if research is needed
  if config.trigger == "auto":
    try:
      needs = await agent._researcher.needs_research(query)
      if not needs:
        return []
    except Exception:
      pass  # Default to running research on failure

  import time

  events: List[RunOutputEvent] = []
  started = DeepResearchStartedEvent(
    run_id=context.run_id,
    session_id=context.session_id,
    agent_id=agent.agent_id,
    agent_name=agent.agent_name,
    query=query,
    depth=config.depth,
  )
  agent._emit(started)
  events.append(started)

  start_time = time.perf_counter()
  try:
    result = await agent._researcher.arun(query)
    context.research_context = result.context
    context.research_result = result
  except Exception as e:
    from definable.utils.log import log_warning

    log_warning(f"Deep research failed: {e}")
    elapsed = (time.perf_counter() - start_time) * 1000
    completed = DeepResearchCompletedEvent(
      run_id=context.run_id,
      session_id=context.session_id,
      agent_id=agent.agent_id,
      agent_name=agent.agent_name,
      query=query,
      duration_ms=elapsed,
    )
    agent._emit(completed)
    events.append(completed)
    return events

  elapsed = (time.perf_counter() - start_time) * 1000
  completed = DeepResearchCompletedEvent(
    run_id=context.run_id,
    session_id=context.session_id,
    agent_id=agent.agent_id,
    agent_name=agent.agent_name,
    query=query,
    sources_used=result.metrics.total_sources_read,
    facts_extracted=result.metrics.unique_facts,
    contradictions_found=result.metrics.contradictions_found,
    waves_executed=result.metrics.waves_executed,
    duration_ms=elapsed,
    compression_ratio=result.metrics.compression_ratio_avg,
  )
  agent._emit(completed)
  events.append(completed)
  return events


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


async def memory_recall(agent: "Agent", context: RunContext, new_messages: List[Message]) -> List[RunOutputEvent]:
  """Recall session history, emit events, inject into context.

  When the memory has an embedder (semantic search), recall produces a
  dual-layer context:
    - Short-term: recent raw messages (conversation continuity)
    - Long-term: top-K atoms ranked by similarity to the query

  Without an embedder, falls back to chronological dump of all entries.

  For v2 memory (tool-based), injects the working memory block into the
  system prompt. The LLM handles recall via tool calls during execution.
  """
  assert agent.memory is not None
  from definable.agent.lifecycle import drain_memory_tasks

  await drain_memory_tasks(agent)
  import time

  session_id = context.session_id or "default"
  user_id = context.user_id or "default"

  events: List[RunOutputEvent] = []

  # Extract the last user message as the query (for event metadata + search)
  query = None
  for msg in reversed(new_messages):
    if msg.role == "user" and msg.content:
      query = msg.content if isinstance(msg.content, str) else str(msg.content)
      break

  started = MemoryRecallStartedEvent(
    run_id=context.run_id,
    session_id=context.session_id,
    agent_id=agent.agent_id,
    agent_name=agent.agent_name,
    query=query or "",
  )
  agent._emit(started)
  events.append(started)

  start_time = time.perf_counter()

  if hasattr(agent.memory, "get_prompt_injection"):
    context.memory_context = await agent.memory.get_prompt_injection(user_id)
    preamble = await agent.memory.get_session_preamble(user_id)  # type: ignore[attr-defined]
    if preamble:
      context.memory_context = f"{context.memory_context}\n\n{preamble}" if context.memory_context else preamble
    if context.memory_context:
      context.active_layers.add("memory")
    completed = MemoryRecallCompletedEvent(
      run_id=context.run_id,
      session_id=context.session_id,
      agent_id=agent.agent_id,
      agent_name=agent.agent_name,
      chunks_included=0,
      chunks_available=0,
      duration_ms=(time.perf_counter() - start_time) * 1000,
      tokens_used=len(context.memory_context or "") // 4,
    )
    agent._emit(completed)
    events.append(completed)
    return events

  # Ensure store is initialized
  await agent.memory._ensure_initialized()

  chunks_included = 0

  if agent.memory.has_semantic_search and query:
    # Dual-layer recall: short-term (recent messages) + long-term (semantic atoms).
    context.memory_context = await memory_recall_semantic(agent, session_id, user_id, query)
    # Count chunks from the context (approximate).
    chunks_included = context.memory_context.count("\n") if context.memory_context else 0
  else:
    # Chronological recall (existing behavior).
    entries = await agent.memory.get_entries(session_id, user_id)
    chunks_included = len(entries)
    if entries:
      lines = []
      for e in entries:
        if e.role == "summary":
          lines.append(f"[Summary]: {e.content}")
        elif e.entry_type == "atom":
          lines.append(f"[Fact]: {e.lossless_content or e.content}")
        else:
          lines.append(f"{e.role}: {e.content}")
      context.memory_context = "<conversation_history>\n" + "\n".join(lines) + "\n</conversation_history>"

  if context.memory_context:
    context.active_layers.add("memory")

  elapsed = (time.perf_counter() - start_time) * 1000

  completed = MemoryRecallCompletedEvent(
    run_id=context.run_id,
    session_id=context.session_id,
    agent_id=agent.agent_id,
    agent_name=agent.agent_name,
    query=query or "",
    tokens_used=len(context.memory_context or "") // 4,
    chunks_included=chunks_included,
    chunks_available=chunks_included,
    duration_ms=elapsed,
  )
  agent._emit(completed)
  events.append(completed)
  return events


async def memory_recall_semantic(agent: "Agent", session_id: str, user_id: str, query: str) -> str:
  """Build dual-layer memory context: short-term messages + long-term atoms."""
  assert agent.memory is not None
  entries = await agent.memory.get_entries(session_id, user_id)

  # Split into recent messages (STM) and search for relevant atoms (LTM).
  recent_messages = [e for e in entries if e.entry_type == "message"][-agent.memory.recent_count :]
  relevant_atoms = await agent.memory.search(query, session_id, user_id)

  parts: list[str] = []

  # Long-term memory: relevant facts from semantic search.
  if relevant_atoms:
    ltm_lines = [f"- {a.lossless_content or a.content}" for a in relevant_atoms]
    parts.append("<long_term_memory>\n" + "\n".join(ltm_lines) + "\n</long_term_memory>")

  # Short-term memory: recent conversation turns.
  if recent_messages:
    stm_lines = []
    for e in recent_messages:
      if e.role == "summary":
        stm_lines.append(f"[Summary]: {e.content}")
      else:
        stm_lines.append(f"{e.role}: {e.content}")
    parts.append("<short_term_memory>\n" + "\n".join(stm_lines) + "\n</short_term_memory>")

  return "\n".join(parts)


def memory_store(agent: "Agent", new_messages: List[Message], context: RunContext) -> List[RunOutputEvent]:
  """Store new messages in session memory (fire-and-forget), emit events."""
  assert agent.memory is not None
  import time

  if not agent.memory.enabled:
    return []

  events: List[RunOutputEvent] = []
  message_count = len(new_messages)

  started = MemoryUpdateStartedEvent(
    run_id=context.run_id,
    session_id=context.session_id,
    agent_id=agent.agent_id,
    agent_name=agent.agent_name,
    message_count=message_count,
  )
  agent._emit(started)
  events.append(started)

  try:
    loop = asyncio.get_running_loop()
    memory = agent.memory
    session_id = context.session_id or "default"
    user_id = context.user_id or "default"

    # Ensure the memory has a model for auto-optimization
    if memory.model is None:
      memory.model = agent.model

    async def _store_and_emit() -> None:
      from definable.utils.log import log_warning

      start_time = time.perf_counter()
      try:
        await memory._ensure_initialized()
        for msg in new_messages:
          await memory.add(msg, session_id=session_id, user_id=user_id)
      except Exception as e:
        log_warning(f"Memory store failed: {type(e).__name__}: {e}")
      finally:
        elapsed = (time.perf_counter() - start_time) * 1000
        completed = MemoryUpdateCompletedEvent(
          run_id=context.run_id,
          session_id=context.session_id,
          agent_id=agent.agent_id,
          agent_name=agent.agent_name,
          message_count=message_count,
          duration_ms=elapsed,
        )
        agent._emit(completed)

    task = loop.create_task(_store_and_emit())
    agent._pending_memory_tasks.append(task)
    task.add_done_callback(lambda t: agent._pending_memory_tasks.remove(t) if t in agent._pending_memory_tasks else None)
  except RuntimeError:
    pass  # No running loop — skip memory storage

  return events


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


async def readers_extract(agent: "Agent", context: RunContext, new_messages: List[Message]) -> List[RunOutputEvent]:
  """Extract text from files in new_messages, inject into context."""
  if not agent.readers:
    return []

  # Collect files from all new messages
  from definable.media import File

  files: List[File] = []
  for msg in new_messages:
    if msg.files:
      files.extend(msg.files)
  if not files:
    return []

  import time

  events: List[RunOutputEvent] = []
  started = FileReadStartedEvent(
    run_id=context.run_id,
    session_id=context.session_id,
    agent_id=agent.agent_id,
    agent_name=agent.agent_name,
    file_count=len(files),
  )
  agent._emit(started)
  events.append(started)

  start_time = time.perf_counter()
  try:
    results = await agent.readers.aread_all(files)
  except Exception:
    from definable.utils.log import log_warning

    log_warning("File reading failed", exc_info=True)
    elapsed = (time.perf_counter() - start_time) * 1000
    completed = FileReadCompletedEvent(
      run_id=context.run_id,
      session_id=context.session_id,
      agent_id=agent.agent_id,
      agent_name=agent.agent_name,
      file_count=len(files),
      files_read=0,
      files_failed=len(files),
      duration_ms=elapsed,
    )
    agent._emit(completed)
    events.append(completed)
    return events

  # Format successful results into context block
  file_blocks: List[str] = []
  files_read = 0
  files_failed = 0
  for result in results:
    if result.error:
      files_failed += 1
    elif result.content:
      files_read += 1
      mime_attr = f' type="{result.mime_type}"' if result.mime_type else ""
      file_blocks.append(f'<file name="{result.filename}"{mime_attr}>\n{result.content}\n</file>')

  if file_blocks:
    context.readers_context = "<file_contents>\n" + "\n".join(file_blocks) + "\n</file_contents>"

  elapsed = (time.perf_counter() - start_time) * 1000
  completed = FileReadCompletedEvent(
    run_id=context.run_id,
    session_id=context.session_id,
    agent_id=agent.agent_id,
    agent_name=agent.agent_name,
    file_count=len(files),
    files_read=files_read,
    files_failed=files_failed,
    duration_ms=elapsed,
  )
  agent._emit(completed)
  events.append(completed)
  return events


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


def extract_input_text(new_messages: List[Message]) -> str:
  """Extract text content from the new user messages for guardrail checking."""
  parts: List[str] = []
  for msg in new_messages:
    if msg.role == "user" and msg.content:
      parts.append(msg.content if isinstance(msg.content, str) else str(msg.content))
  return "\n".join(parts)


async def run_input_guardrails(agent: "Agent", context: RunContext, new_messages: List[Message]) -> Optional[RunOutput]:
  """Run input guardrails. Returns RunOutput if blocked, None if allowed."""
  assert agent.guardrails is not None

  from definable.agent.guardrail.events import GuardrailBlockedEvent, GuardrailCheckedEvent

  text = extract_input_text(new_messages)
  if not text:
    return None

  results = await agent.guardrails.run_input_checks(text, context)

  for result in results:
    gname = (result.metadata or {}).get("guardrail_name", "unknown")
    duration = (result.metadata or {}).get("duration_ms")

    agent._emit(
      GuardrailCheckedEvent(
        run_id=context.run_id,
        session_id=context.session_id,
        agent_id=agent.agent_id,
        agent_name=agent.agent_name,
        guardrail_name=gname,
        guardrail_type="input",
        action=result.action,
        message=result.message,
        duration_ms=duration,
      )
    )

    if result.action == "block":
      agent._emit(
        GuardrailBlockedEvent(
          run_id=context.run_id,
          session_id=context.session_id,
          agent_id=agent.agent_id,
          agent_name=agent.agent_name,
          guardrail_name=gname,
          guardrail_type="input",
          reason=result.message or "Blocked by input guardrail",
        )
      )

      reason = result.message or "Blocked by input guardrail"
      if agent.guardrails.on_block == "raise":
        from definable.exceptions import CheckTrigger, InputCheckError

        raise InputCheckError(reason, check_trigger=CheckTrigger.GUARDRAIL_BLOCKED)

      return RunOutput(
        run_id=context.run_id,
        session_id=context.session_id,
        agent_id=agent.agent_id,
        agent_name=agent.agent_name,
        content=reason,
        status=RunStatus.blocked,
      )

    if result.action == "modify" and result.modified_text is not None:
      # Replace the last user message content
      for i in range(len(new_messages) - 1, -1, -1):
        if new_messages[i].role == "user":
          new_messages[i] = Message(
            role="user",
            content=result.modified_text,
            images=new_messages[i].images,
            videos=new_messages[i].videos,
            audio=new_messages[i].audio,
            files=new_messages[i].files,
          )
          break
      # Also update all_messages in context metadata
      all_messages = context.metadata.get("_messages") if context.metadata else None
      if all_messages:
        for i in range(len(all_messages) - 1, -1, -1):
          if all_messages[i].role == "user":
            all_messages[i] = Message(
              role="user",
              content=result.modified_text,
              images=all_messages[i].images,
              videos=all_messages[i].videos,
              audio=all_messages[i].audio,
              files=all_messages[i].files,
            )
            break

  return None


async def run_output_guardrails(agent: "Agent", context: RunContext, result: RunOutput) -> Optional[RunOutput]:
  """Run output guardrails. Returns modified RunOutput if blocked/modified, None if allowed."""
  assert agent.guardrails is not None

  from definable.agent.guardrail.events import GuardrailBlockedEvent, GuardrailCheckedEvent

  text = result.content if isinstance(result.content, str) else str(result.content or "")
  if not text:
    return None

  results = await agent.guardrails.run_output_checks(text, context)

  modified = False
  for gr in results:
    gname = (gr.metadata or {}).get("guardrail_name", "unknown")
    duration = (gr.metadata or {}).get("duration_ms")

    agent._emit(
      GuardrailCheckedEvent(
        run_id=context.run_id,
        session_id=context.session_id,
        agent_id=agent.agent_id,
        agent_name=agent.agent_name,
        guardrail_name=gname,
        guardrail_type="output",
        action=gr.action,
        message=gr.message,
        duration_ms=duration,
      )
    )

    if gr.action == "block":
      agent._emit(
        GuardrailBlockedEvent(
          run_id=context.run_id,
          session_id=context.session_id,
          agent_id=agent.agent_id,
          agent_name=agent.agent_name,
          guardrail_name=gname,
          guardrail_type="output",
          reason=gr.message or "Blocked by output guardrail",
        )
      )

      reason = gr.message or "Blocked by output guardrail"
      if agent.guardrails.on_block == "raise":
        from definable.exceptions import CheckTrigger, OutputCheckError

        raise OutputCheckError(reason, check_trigger=CheckTrigger.GUARDRAIL_BLOCKED)

      return RunOutput(
        run_id=context.run_id,
        session_id=context.session_id,
        agent_id=agent.agent_id,
        agent_name=agent.agent_name,
        content=reason,
        status=RunStatus.blocked,
        messages=result.messages,
        metrics=result.metrics,
      )

    if gr.action == "modify" and gr.modified_text is not None:
      result.content = gr.modified_text
      if result.metadata is None:
        result.metadata = {}
      result.metadata["guardrail_modified"] = True
      modified = True

  return result if modified else None


async def run_tool_guardrails(agent: "Agent", context: RunContext, tool_execution: "ToolExecution") -> Optional[str]:
  """Run tool guardrails. Returns block reason string if blocked, None if allowed."""
  assert agent.guardrails is not None

  from definable.agent.guardrail.events import GuardrailBlockedEvent, GuardrailCheckedEvent

  tool_name = tool_execution.tool_name or ""
  tool_args = tool_execution.tool_args or {}

  results = await agent.guardrails.run_tool_checks(tool_name, tool_args, context)

  for gr in results:
    gname = (gr.metadata or {}).get("guardrail_name", "unknown")
    duration = (gr.metadata or {}).get("duration_ms")

    agent._emit(
      GuardrailCheckedEvent(
        run_id=context.run_id,
        session_id=context.session_id,
        agent_id=agent.agent_id,
        agent_name=agent.agent_name,
        guardrail_name=gname,
        guardrail_type="tool",
        action=gr.action,
        message=gr.message,
        duration_ms=duration,
      )
    )

    if gr.action == "block":
      reason = gr.message or f"Tool '{tool_name}' blocked by guardrail"
      agent._emit(
        GuardrailBlockedEvent(
          run_id=context.run_id,
          session_id=context.session_id,
          agent_id=agent.agent_id,
          agent_name=agent.agent_name,
          guardrail_name=gname,
          guardrail_type="tool",
          reason=reason,
        )
      )
      return reason

  return None


# ---------------------------------------------------------------------------
# Orchestration  (routing, triggers, pre-execution pipeline)
# ---------------------------------------------------------------------------


def extract_last_user_query(messages: List[Message]) -> Optional[str]:
  """Extract the content of the last user message (for trigger pre-checks)."""
  for msg in reversed(messages):
    if hasattr(msg, "role") and msg.role == "user" and msg.content:
      return msg.content if isinstance(msg.content, str) else str(msg.content)
  return None


def build_routing_prompt(layer_name: str, query: str, context_str: str) -> str:
  """Build a precise, layer-specific YES/NO routing prompt.

  Generic prompts cause routing models to over-fire (always YES). Explicit
  criteria with positive/negative signals produce accurate routing decisions.
  """
  ctx_block = f"\nRecent conversation:\n{context_str}\n" if context_str else ""
  q = query[:300]

  if layer_name == "knowledge base":
    return (
      f"You are a routing system. Answer ONLY with YES or NO.\n\n"
      f"QUESTION: Does this query need the knowledge base?\n\n"
      f"KNOWLEDGE BASE contains factual documents: company policies, product info, procedures, uploaded content.\n\n"
      f"Answer YES when the query asks about:\n"
      f"- Company rules, benefits, policies (PTO, salary, leave, procedures)\n"
      f"- Product details, features, or documentation\n"
      f"- Factual questions about the organization or domain\n"
      f"- 'How does X work?', 'What is the policy for Y?', 'Tell me about Z'\n\n"
      f"Answer NO when the query is:\n"
      f"- Simple math or calculations ('add 1 and 2', 'what is 5*7')\n"
      f"- General conversation, greetings, or chit-chat\n"
      f"- Coding tasks, logic puzzles, or general reasoning\n"
      f"- Questions answerable from common world knowledge (no documents needed)\n"
      f"- Personal-only questions about the user (memory handles those){ctx_block}\n"
      f"Query: '{q}'\n\n"
      f"Answer YES or NO only:"
    )

  if layer_name == "memory":
    return (
      f"You are a routing system. Answer ONLY with YES or NO.\n\n"
      f"QUESTION: Does this query need personal memory recall?\n\n"
      f"MEMORY stores personal information about this user from past conversations.\n\n"
      f"Answer YES when the query involves:\n"
      f"- User's name, role, preferences, or personal details\n"
      f"- References to past interactions ('what did I tell you', 'remember when', 'last time')\n"
      f"- Possessive questions ('my name', 'my preference', 'my project')\n"
      f"- Follow-ups that require knowing who the user is or what they said before\n\n"
      f"Answer NO when the query is:\n"
      f"- Simple math or calculations ('add 1 and 2')\n"
      f"- General factual questions not specific to this user\n"
      f"- Topics fully answerable without user-specific context\n"
      f"- The very first message with no personal reference{ctx_block}\n"
      f"Query: '{q}'\n\n"
      f"Answer YES or NO only:"
    )

  if layer_name == "analysis/thinking":
    return (
      f"You are a routing system. Answer ONLY with YES or NO.\n\n"
      f"QUESTION: Does this query need extended step-by-step reasoning?\n\n"
      f"THINKING enables slow, careful analysis before the assistant responds.\n\n"
      f"Answer YES when the query involves:\n"
      f"- Multi-step math, logic proofs, or complex reasoning\n"
      f"- Code architecture, algorithm design, or debugging\n"
      f"- Strategic planning, trade-off analysis, or ambiguous decisions\n"
      f"- Tasks where rushing to an answer risks being wrong\n\n"
      f"Answer NO when the query is:\n"
      f"- Simple arithmetic ('add 1 and 2', 'what is 5+3')\n"
      f"- Direct factual lookups ('what is the PTO policy')\n"
      f"- Simple instructions ('send an email to X')\n"
      f"- Casual conversation or greetings{ctx_block}\n"
      f"Query: '{q}'\n\n"
      f"Answer YES or NO only:"
    )

  # Fallback for custom layer names
  ctx_section = f"Recent conversation:\n{context_str}\n" if context_str else ""
  return (
    f"You are a routing system. Answer ONLY with YES or NO.\n\n"
    f"QUESTION: Does this query require accessing the {layer_name}?\n\n"
    f"{ctx_section}"
    f"Query: '{q}'\n\n"
    f"Answer YES or NO only:"
  )


async def should_invoke_layer(
  agent: "Agent",
  layer_name: str,
  query: str,
  decision_prompt: Optional[str] = None,
  routing_model: Optional["Model"] = None,
  messages: Optional[List[Message]] = None,
) -> bool:
  """Lightweight YES/NO pre-check: does this query need the given layer?

  Uses routing_model (if provided) or falls back to the agent's model.
  Includes recent conversation context so the gate has enough signal.
  Returns True on failure to default to running the layer (fail-open).
  """
  model = routing_model or agent.model

  context_str = ""
  if messages:
    recent = messages[-3:]
    context_str = "\n".join(
      f"{m.role}: {(m.content[:200] if isinstance(m.content, str) else str(m.content)[:200])}"
      for m in recent
      if m.role in ("user", "assistant") and m.content
    )

  if decision_prompt:
    prompt = decision_prompt
  else:
    prompt = build_routing_prompt(layer_name, query, context_str)

  try:
    response = await model.ainvoke(
      messages=[Message(role="user", content=prompt)],
      assistant_message=Message(role="assistant", content=""),
    )
    answer = (response.content or "").strip().upper()
    return answer in ("YES", "Y")
  except Exception as e:
    from definable.utils.log import log_warning

    log_warning(f"Layer routing check failed for '{layer_name}', defaulting to run: {e}")
    return True  # fail-open: run the layer if routing fails


async def evaluate_layer_trigger(
  agent: "Agent",
  trigger: Literal["always", "auto", "never"],
  callback: Callable[[], Awaitable[List[RunOutputEvent]]],
  *,
  layer_name: str = "",
  query_messages: Optional[List[Message]] = None,
  all_messages: Optional[List[Message]] = None,
  decision_prompt: Optional[str] = None,
  routing_model: Optional["Model"] = None,
) -> List[RunOutputEvent]:
  """Evaluate a layer trigger and conditionally run the callback.

  Returns callback's events if the layer runs, [] if skipped.
  Fails open on 'auto' gate errors (returns callback result).
  """
  if trigger == "always":
    return await callback()
  if trigger == "auto":
    if query_messages is None:
      return []
    query = extract_last_user_query(query_messages)
    if query and await should_invoke_layer(
      agent,
      layer_name,
      query,
      decision_prompt,
      routing_model,
      all_messages,
    ):
      return await callback()
  # "never" falls through
  return []


def should_store_memory(agent: "Agent") -> bool:
  """Return True if memory store should run this turn."""
  if not agent.memory:
    return False
  # v2 memory is tool-managed — LLM stores via tool calls, no batch store phase
  if hasattr(agent.memory, "get_tools"):
    return False
  return agent.memory.enabled


async def run_pre_execution_pipeline(
  agent: "Agent",
  context: RunContext,
  new_messages: List[Message],
  all_messages: List[Message],
) -> List[RunOutputEvent]:
  """Pre-execution pipeline: readers -> knowledge -> research -> memory recall.

  Populates context fields (knowledge_context, research_context, memory_context,
  readers_context, active_layers) consumed by _execute_run() and arun_stream().
  """
  events: List[RunOutputEvent] = []

  # File reading (before knowledge — extracted content may inform the query)
  events.extend(await readers_extract(agent, context, new_messages))

  # Knowledge retrieval
  if agent._knowledge and agent._knowledge.enabled:
    events.extend(
      await evaluate_layer_trigger(
        agent,
        trigger=agent._knowledge.trigger,
        callback=lambda: knowledge_retrieve(agent, context),
        layer_name="knowledge base",
        query_messages=all_messages,
        all_messages=all_messages,
        decision_prompt=agent._knowledge.decision_prompt,
        routing_model=agent._knowledge.routing_model,
      )
    )

  # Deep research (after knowledge, before memory)
  events.extend(await deep_research(agent, context))

  # Memory recall
  if agent.memory and agent.memory.enabled:
    events.extend(await memory_recall(agent, context, new_messages))

  return events


# ---------------------------------------------------------------------------
# Thinking
# ---------------------------------------------------------------------------

# Effort-scaled thinking prompts (research-backed design):
# - LOW: Chain-of-Draft style — minimal tokens, fast assessment
# - MEDIUM: Standard reasoning with step-back abstraction + self-check
# - HIGH: Full deliberative reasoning — metacognitive, multi-perspective, verified
_THINKING_PROMPTS: Dict[str, str] = {
  "low": (
    "You are the planning layer. Your job is to produce an execution STRATEGY — not the answer itself.\n\n"
    "Quickly assess this request:\n"
    "- What is the user asking?\n"
    "- What approach should be used? (tools, knowledge, direct answer)\n"
    "- What tools are needed, if any?\n\n"
    "Output a brief strategy (2-4 sentences). DO NOT solve the problem or write the answer."
  ),
  "medium": (
    "You are the planning layer. Your job is to produce an execution STRATEGY — not the answer itself.\n"
    "The main model will receive your strategy and use it to produce the actual response.\n\n"
    "1. INTENT: What does the user actually need? Look past the surface request.\n"
    "2. APPROACH: What's the best way to fulfill this? Direct answer from knowledge, tool usage, multi-step reasoning?\n"
    "3. TOOLS: If tools are needed, list them in order with why each is needed and what depends on what.\n"
    "4. CONSTRAINTS: Any edge cases, risks, or things to watch for?\n\n"
    "Output a concise strategy. DO NOT write the answer, solve the problem, or produce the final response.\n"
    "The main model handles that — you only plan."
  ),
  "high": (
    "You are the planning layer. Your job is to produce a detailed execution STRATEGY — not the answer itself.\n"
    "The main model will receive your strategy and use it to produce the actual response.\n\n"
    "INTENT: What is the user really asking? What are the underlying goals and constraints?\n"
    "Identify any ambiguity or implicit requirements.\n\n"
    "APPROACH: What's the best strategy? Consider:\n"
    "- Can this be answered directly from knowledge, or does it need tools/research?\n"
    "- If complex, break into sub-tasks. Which are independent vs sequential?\n"
    "- If tools are available, which ones and in what order? Why each is needed?\n\n"
    "ALTERNATIVES: Consider at least one alternative approach. Why is your chosen strategy better?\n\n"
    "RISKS: Edge cases, failure modes, assumptions that need validation.\n\n"
    "VERIFICATION: Does your strategy fully address the original request? Any gaps?\n\n"
    "Output a detailed strategy. DO NOT solve the problem, write the answer, or produce the final response.\n"
    "You are the planner — the main model is the executor."
  ),
}


def build_thinking_prompt(
  agent: "Agent",
  context: RunContext,
  tools: Dict[str, Function],
) -> str:
  """Build an effort-scaled, context-aware thinking prompt.

  The prompt varies by effort level:
  - low: Chain-of-Draft style — minimal overhead, fast assessment
  - medium: Standard reasoning with step-back abstraction + self-check
  - high: Full deliberative reasoning — metacognitive, multi-perspective, verified

  Incorporates agent role, tool catalog, and context availability flags.
  """
  effort = agent._thinking.effort if agent._thinking else "medium"
  base_prompt = _THINKING_PROMPTS.get(effort, _THINKING_PROMPTS["medium"])
  parts = [base_prompt]

  # Agent role (first 500 chars)
  if agent.instructions:
    truncated = agent.instructions[:500]
    if len(agent.instructions) > 500:
      truncated += "..."
    parts.append(f"\nYour role: {truncated}")

  # Tool catalog (name + one-line description)
  if tools:
    tool_lines = []
    for name, fn in tools.items():
      desc = (fn.description or "").split("\n")[0][:100]
      tool_lines.append(f"- {name}: {desc}" if desc else f"- {name}")
    parts.append("\nAvailable tools:\n" + "\n".join(tool_lines))

  # Context availability flags (NOT the full content)
  flags = []
  if context.knowledge_context:
    flags.append("knowledge base context is available")
  if context.memory_context:
    flags.append("conversation memory is available")
  if flags:
    parts.append(f"\nContext: {'; '.join(flags)}.")

  parts.append("\nRemember: Output only a strategy. The main model will handle the actual response.")

  return "\n".join(parts)


def format_thinking_injection(output: "Union[ThinkingOutput, str]", effort: str = "medium") -> str:
  """Format thinking result into a system prompt injection.

  Accepts either a ThinkingOutput (structured models) or a plain str
  (non-structured models' free-form thinking text).

  The injection is framed as an execution strategy directive — the main
  model should follow it to produce the actual response.
  """
  framing = "The following strategy was developed for this request. Use it to guide your response:"

  if isinstance(output, str):
    # Free-form text from non-structured models
    return f"<execution_strategy>\n{framing}\n\n{output}\n</execution_strategy>"

  # Structured ThinkingOutput
  tool_names = output.flat_tool_names()

  if effort == "high":
    lines = [
      "<execution_strategy>",
      framing,
      "",
      f"Approach: {output.approach}",
    ]
    if tool_names:
      lines.append(f"Tools: {', '.join(tool_names)}")
    if output.verification:
      lines.append(f"Verification: {output.verification}")
    if output.considerations:
      lines.append(f"Considerations: {output.considerations}")
    if output.confidence:
      lines.append(f"Confidence: {output.confidence}")
    lines.append("</execution_strategy>")
    return "\n".join(lines)

  # low / medium — compact
  parts = [f"<execution_strategy>\nStrategy: {output.approach}"]
  if tool_names:
    parts.append(f" Tools: {', '.join(tool_names)}.")
  parts.append("\n</execution_strategy>")
  return "".join(parts)


async def thinking_should_run(agent: "Agent", messages: List[Message]) -> bool:
  """Return True if the thinking layer should execute this turn."""
  if not (agent._thinking and agent._thinking.enabled):
    return False
  trigger = agent._thinking.trigger
  if trigger == "always":
    return True
  if trigger == "auto":
    query = extract_last_user_query(messages)
    if query:
      return await should_invoke_layer(agent, "analysis/thinking", query)
  return False  # "never"


def build_thinking_messages(
  agent: "Agent",
  context: RunContext,
  invoke_messages: List[Message],
  tools: Dict[str, Function],
) -> "tuple[list[Message], bool]":
  """Build the messages for a thinking LLM call.

  Returns:
    (thinking_messages, use_structured) — the messages and whether structured output is used.

  For structured models (OpenAI, Gemini): uses ThinkingOutput schema via structured output.
  For non-structured models (Moonshot, DeepSeek, xAI): pure natural language, no format
  instructions — produces clean free-form text.
  """
  assert agent._thinking is not None

  thinking_model = agent._thinking.model or agent.model

  # Use custom instructions if set, otherwise build context-aware prompt
  if agent._thinking.instructions:
    thinking_prompt = agent._thinking.instructions
  else:
    thinking_prompt = build_thinking_prompt(agent, context, tools)

  # Build thinking messages: system prompt + user/assistant messages (no tools)
  thinking_messages: list[Message] = [Message(role="system", content=thinking_prompt)]
  for msg in invoke_messages:
    if msg.role in ("user", "assistant"):
      thinking_messages.append(msg)

  use_structured = thinking_model.supports_native_structured_outputs

  # Non-structured models: no format instructions at all.
  # They produce pure natural language reasoning text.

  return thinking_messages, use_structured


async def execute_thinking(
  agent: "Agent",
  context: RunContext,
  invoke_messages: List[Message],
  tools: Dict[str, Function],
) -> "AsyncGenerator[Union[str, tuple[Optional[ThinkingOutput], Optional[str], list[ReasoningStep], list[Message]]], None]":
  """Execute Definable's fallback thinking layer as a unified async generator.

  Always streams via ainvoke_stream. Behavior splits by model capability:

  - Structured models (OpenAI, Gemini): accumulates response, parses ThinkingOutput
    at end, yields chain_of_thought tokens during stream.
  - Non-structured models (Moonshot, DeepSeek, xAI): yields raw text tokens,
    builds thinking_text string at end. No JSON, no XML — pure natural language.

  Yields:
    str: Content delta tokens (for ReasoningContentDelta events).
    tuple: Final result ``(thinking_output_or_none, thinking_text_or_none, reasoning_steps, reasoning_messages)``
      as the last item.
  """
  from definable.agent.reasoning.step import ThinkingOutput, thinking_output_to_reasoning_steps

  assert agent._thinking is not None
  thinking_model = agent._thinking.model or agent.model

  thinking_messages, use_structured = build_thinking_messages(agent, context, invoke_messages, tools)

  assistant_msg = Message(role="assistant")

  if use_structured:
    # Structured output: call with response_format, then yield chain_of_thought
    response = await thinking_model.ainvoke(
      messages=thinking_messages,
      assistant_message=assistant_msg,
      response_format=ThinkingOutput,
    )
    raw_content = response.content if isinstance(response.content, str) else str(response.content or "")

    # Parse structured response
    thinking_output: Optional["ThinkingOutput"] = None
    reasoning_steps: "list[ReasoningStep]" = []
    try:
      parsed = json.loads(raw_content)
      if isinstance(parsed, dict):
        if "analysis" in parsed and "chain_of_thought" not in parsed:
          parsed["chain_of_thought"] = parsed.pop("analysis")
        thinking_output = ThinkingOutput(**parsed)
        reasoning_steps = thinking_output_to_reasoning_steps(thinking_output)
    except Exception:
      from definable.utils.log import log_warning

      log_warning("Failed to parse structured thinking response, using raw content")
      thinking_output = ThinkingOutput(chain_of_thought=raw_content, approach="Respond directly")  # type: ignore[call-arg]
      reasoning_steps = thinking_output_to_reasoning_steps(thinking_output)

    if thinking_output:
      yield thinking_output.chain_of_thought

    reasoning_agent_messages = thinking_messages + [
      Message(role="assistant", content=response.content, metrics=response.response_usage)  # type: ignore[arg-type]
    ]
    yield (thinking_output, None, reasoning_steps, reasoning_agent_messages)
    return

  # Non-structured: stream pure natural language text token-by-token
  accumulated = ""
  response_usage = None

  async for chunk in thinking_model.ainvoke_stream(
    messages=thinking_messages,
    assistant_message=assistant_msg,
  ):
    if chunk.content:
      accumulated += chunk.content
      yield chunk.content
    if chunk.response_usage:
      response_usage = chunk.response_usage

  # Build reasoning messages
  msg_kwargs: Dict[str, Any] = {"role": "assistant", "content": accumulated}
  if response_usage is not None:
    msg_kwargs["metrics"] = response_usage
  reasoning_agent_messages = thinking_messages + [Message(**msg_kwargs)]

  # No ThinkingOutput for non-structured — just free-form text
  yield (None, accumulated or None, [], reasoning_agent_messages)


def enable_native_thinking(agent: "Agent") -> None:
  """Configure the model to use native extended thinking.

  Called when thinking=True (or Thinking(...)) and the model supports it.
  Sets the model's thinking parameter (e.g. Claude's thinking dict,
  Gemini's thinking_budget, OpenAI's reasoning_effort).
  """
  assert agent._thinking is not None
  budget = agent._thinking.resolve_budget_tokens()

  # Claude: thinking = {"type": "enabled", "budget_tokens": N}
  from definable.model.anthropic.claude import Claude

  if isinstance(agent.model, Claude):
    if not agent.model.thinking:
      agent.model.thinking = {"type": "enabled", "budget_tokens": budget}
    return

  # Gemini: thinking_budget
  from definable.model.google.gemini import Gemini

  if isinstance(agent.model, Gemini):
    if agent.model.thinking_budget is None:
      agent.model.thinking_budget = budget
    return

  # OpenAI (o1/o3): reasoning_effort
  from definable.model.openai.chat import OpenAIChat

  if isinstance(agent.model, OpenAIChat):
    if agent.model.reasoning_effort is None:
      agent.model.reasoning_effort = agent._thinking.effort
    return
