"""Agent parameter resolution — converts user-facing Union[bool, Config, Instance] to resolved types.

All functions are called once during Agent.__init__. They are pure (no side effects beyond
lazy imports) and take explicit parameters instead of accessing self.
"""

import dataclasses
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
  from definable.agent.compression import Compression
  from definable.agent.compression.manager import CompressionManager
  from definable.agent.config import AgentConfig
  from definable.agent.context import Context
  from definable.agent.context.deferred import DeferredToolManager
  from definable.agent.context.manager import ContextManager
  from definable.agent.guardrail.base import Guardrails
  from definable.agent.observability.config import ObservabilityConfig
  from definable.agent.pipeline.debug import DebugConfig
  from definable.agent.pipeline.sub_agent import SubAgentPolicy
  from definable.agent.reasoning.thinking import Thinking
  from definable.agent.toolkit import Toolkit
  from definable.agent.tracing.base import TraceWriter, Tracing
  from definable.knowledge import Knowledge
  from definable.memory.manager import Memory
  from definable.model.base import Model
  from definable.reader.base import BaseReader
  from definable.skill.base import Skill
  from definable.tool.function import Function


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------


def resolve_model(model: Union[str, "Model", None]) -> "Model":
  """Resolve model param: string shorthand → Model instance."""
  if model is None:
    raise TypeError(
      "Agent requires a 'model' argument. Pass a Model instance "
      "(e.g., OpenAIChat(id='gpt-4o-mini')) or a string shorthand (e.g., 'openai/gpt-4o-mini')."
    )
  if isinstance(model, str):
    from definable.model.utils import resolve_model_string

    return resolve_model_string(model)
  return model


# ---------------------------------------------------------------------------
# Memory resolution
# ---------------------------------------------------------------------------


def resolve_memory(memory: "Union[Memory, bool, None]") -> "Optional[Memory]":
  """Resolve memory param to Memory | None.

  Accepts:
    - False/None → None
    - True → Memory with SemanticStrategy + ConsolidationPolicy (smart defaults)
    - Memory instance → pass through
  """
  if memory is False or memory is None:
    return None
  if memory is True:
    from definable.memory.consolidation import ConsolidationPolicy
    from definable.memory.manager import Memory
    from definable.memory.store.in_memory import InMemoryStore
    from definable.memory.strategies.semantic import SemanticStrategy

    return Memory(
      store=InMemoryStore(),
      strategy=SemanticStrategy(),
      consolidation=ConsolidationPolicy(),
    )

  # Memory instance — pass through
  return memory


def resolve_memory_embedder(memory: "Optional[Memory]", model: "Model") -> None:
  """Auto-resolve memory embedder from model provider if not set. Mutates memory in-place."""
  if memory is None or getattr(memory, "embedder", None) is not None:
    return

  # Only auto-resolve for Memory instances with strategies (not CortexMemory etc.)
  if not hasattr(memory, "_resolve_strategy"):
    return

  from definable.memory.strategies.semantic import SemanticStrategy

  strategy = memory._resolve_strategy()
  if not isinstance(strategy, SemanticStrategy):
    return

  embedder = create_embedder_for_model(model)
  if embedder is not None:
    memory.embedder = embedder


def create_embedder_for_model(model: "Model") -> Any:
  """Create an embedder matching the model provider. Returns None if unavailable."""
  model_cls = type(model).__name__
  try:
    if model_cls == "OpenAIChat":
      from definable.knowledge.embedder.openai import OpenAIEmbedder

      return OpenAIEmbedder()
    if model_cls == "Gemini":
      from definable.knowledge.embedder.google import GoogleEmbedder

      return GoogleEmbedder()
    if model_cls == "MistralChat":
      from definable.knowledge.embedder.mistral import MistralEmbedder

      return MistralEmbedder()
  except Exception:
    pass

  # Default: try OpenAI embedder (most commonly available).
  try:
    from definable.knowledge.embedder.openai import OpenAIEmbedder

    return OpenAIEmbedder()
  except Exception:
    return None


# ---------------------------------------------------------------------------
# Knowledge resolution
# ---------------------------------------------------------------------------


def resolve_knowledge(knowledge: "Union[Knowledge, str, bool, None]") -> "Optional[Knowledge]":
  """Resolve knowledge param to Knowledge | None.

  Accepts:
    - False/None → None
    - True → ValueError (ambiguous — no path to load from)
    - str → Knowledge.from_path(path) with auto-configured RAG pipeline
    - Knowledge instance → pass through
  """
  if knowledge is False or knowledge is None:
    return None
  if knowledge is True:
    raise ValueError(
      "knowledge=True is not supported. Pass a path string or Knowledge instance:"
      " Agent(knowledge='./docs/') or Agent(knowledge=Knowledge(vector_db=..., top_k=5))."
    )
  if isinstance(knowledge, str):
    from definable.knowledge.base import Knowledge as _Knowledge

    return _Knowledge.from_path(knowledge)

  return knowledge


# ---------------------------------------------------------------------------
# Tracing / debug / observability resolution
# ---------------------------------------------------------------------------


def resolve_tracing(tracing_param: "Union[Tracing, bool, None]", config: "Optional[AgentConfig]") -> "Optional[Tracing]":
  """Resolve tracing param to Tracing | None."""
  from definable.agent.tracing.base import Tracing as _Tracing

  if tracing_param is False:
    return config.tracing if config else None
  if tracing_param is True:
    return _Tracing()
  if isinstance(tracing_param, _Tracing):
    return tracing_param
  return config.tracing if config else None


def resolve_debug(
  debug: "Union[bool, DebugConfig, None]",
  tracing_config: "Optional[Tracing]",
) -> "tuple[Optional[DebugConfig], Optional[Tracing]]":
  """Resolve debug param. Returns (debug_config, updated_tracing_config)."""
  from definable.agent.pipeline.debug import DebugConfig as _DebugConfig

  debug_config: "Optional[DebugConfig]" = None
  if isinstance(debug, _DebugConfig):
    debug_config = debug
    debug_enabled = True
  else:
    debug_enabled = bool(debug)

  if debug_enabled:
    from definable.agent.tracing.base import Tracing as _Tracing
    from definable.agent.tracing.debug import DebugExporter

    if tracing_config is None:
      tracing_config = _Tracing(exporters=[DebugExporter()])
    else:
      existing = tracing_config.exporters or []
      tracing_config = dataclasses.replace(tracing_config, exporters=[*existing, DebugExporter()])

  return debug_config, tracing_config


def resolve_observability(
  observability: "Union[bool, ObservabilityConfig, None]",
  tracing_config: "Optional[Tracing]",
) -> "tuple[Optional[ObservabilityConfig], Any, Optional[Tracing]]":
  """Resolve observability param. Returns (obs_config, obs_exporter, updated_tracing_config)."""
  from definable.agent.observability.config import ObservabilityConfig as _ObsConfig

  obs_config: "Optional[ObservabilityConfig]" = None
  obs_exporter: Any = None
  if isinstance(observability, _ObsConfig):
    obs_config = observability
  elif observability is True:
    obs_config = _ObsConfig(enabled=True)

  if obs_config is not None and obs_config.enabled:
    from definable.agent.observability.collector import ObservabilityExporter as _ObsExporter
    from definable.agent.tracing.base import Tracing as _Tracing

    obs_exporter = _ObsExporter(buffer_size=obs_config.buffer_size)
    if tracing_config is None:
      tracing_config = _Tracing(exporters=[obs_exporter])
    else:
      existing_exporters = tracing_config.exporters or []
      tracing_config = dataclasses.replace(tracing_config, exporters=[*existing_exporters, obs_exporter])

  return obs_config, obs_exporter, tracing_config


def init_tracing(tracing_config: "Optional[Tracing]") -> "Optional[TraceWriter]":
  """Initialize trace writer using resolved tracing config."""
  from definable.agent.tracing.base import TraceWriter

  if tracing_config and tracing_config.exporters:
    return TraceWriter(tracing_config)
  return None


# ---------------------------------------------------------------------------
# Thinking / research / sub-agents resolution
# ---------------------------------------------------------------------------


def resolve_thinking(thinking: "Union[bool, Thinking, None]") -> "Optional[Thinking]":
  """Resolve thinking param to Thinking | None."""
  from definable.agent.reasoning.thinking import Thinking as _Thinking

  if thinking is True:
    return _Thinking()
  if isinstance(thinking, _Thinking):
    return thinking
  return None


def resolve_deep_research(
  deep_research: "Union[bool, Any, None]",
) -> "tuple[Any, Any]":
  """Resolve deep_research param. Returns (config, prebuilt_engine)."""
  from definable.agent.research.config import DeepResearchConfig as _DRConfig
  from definable.agent.research.engine import DeepResearch as _DREngine

  if isinstance(deep_research, _DREngine):
    return None, deep_research
  if deep_research is True:
    return _DRConfig(), None
  if isinstance(deep_research, _DRConfig):
    return deep_research, None
  return None, None


def resolve_sub_agents(sub_agents: "Union[bool, SubAgentPolicy, None]") -> "Optional[SubAgentPolicy]":
  """Resolve sub_agents param to SubAgentPolicy | None."""
  from definable.agent.pipeline.sub_agent import SubAgentPolicy as _SubAgentPolicy

  if sub_agents is True:
    return _SubAgentPolicy()
  if isinstance(sub_agents, _SubAgentPolicy):
    return sub_agents
  return None


# ---------------------------------------------------------------------------
# Audio / security / usage resolution
# ---------------------------------------------------------------------------


def resolve_audio_transcriber(audio_transcriber: "Union[bool, Any, None]") -> Any:
  """Resolve audio_transcriber param."""
  from definable.reader.audio import AudioTranscriber as _AudioTranscriber, OpenAITranscriber as _OpenAITranscriber

  if audio_transcriber is True:
    return _OpenAITranscriber()
  if isinstance(audio_transcriber, _AudioTranscriber):
    return audio_transcriber
  return None


def resolve_security(
  security: "Union[bool, Any, None]",
  guardrails: "Optional[Guardrails]",
) -> "tuple[Any, Optional[Guardrails]]":
  """Resolve security param. Returns (security_config, updated_guardrails)."""
  from definable.agent.security import SecurityConfig as _SecurityConfig

  security_config = None
  if security is True:
    security_config = _SecurityConfig()
  elif isinstance(security, _SecurityConfig):
    security_config = security

  # Auto-inject security guardrails
  if security_config is not None:
    from definable.agent.guardrail.base import Guardrails as _Guardrails

    if guardrails is None:
      guardrails = _Guardrails()
    if security_config.tool_policy is not None:
      from definable.agent.security.tool_policy import ToolPolicyGuardrail

      guardrails.tool.append(ToolPolicyGuardrail(policy=security_config.tool_policy))
    if security_config.content_defense is not None and security_config.content_defense.injection_detection:
      from definable.agent.security.content_defense import ContentDefenseGuardrail

      cd = security_config.content_defense
      guardrails.input.append(
        ContentDefenseGuardrail(
          sensitivity=cd.injection_sensitivity,
          extra_patterns=cd.extra_patterns,
        )
      )

  return security_config, guardrails


def resolve_usage(usage: "Union[bool, Any, None]") -> Any:
  """Resolve usage param to UsageTracker | None."""
  from definable.agent.usage import UsageTracker as _UsageTracker

  if usage is True:
    return _UsageTracker()
  if isinstance(usage, _UsageTracker):
    return usage
  return None


def resolve_plugins(plugins: Optional[List[Any]]) -> "tuple[Any, bool]":
  """Resolve plugins param. Returns (plugin_registry, plugins_loaded)."""
  from definable.agent.plugin.registry import PluginRegistry as _PluginRegistry

  registry = _PluginRegistry()
  if plugins:
    for p in plugins:
      registry.add(p)
  return registry, False


# ---------------------------------------------------------------------------
# Context / compression / deferred tools resolution
# ---------------------------------------------------------------------------


def resolve_context(context: "Union[bool, Context, None]", model: "Model") -> "Optional[ContextManager]":
  """Resolve context param into a ContextManager (or None)."""
  from definable.agent.context import Context as _Context
  from definable.agent.context.manager import ContextManager

  if context is True:
    return ContextManager(_Context(), model=model)
  if isinstance(context, _Context):
    return ContextManager(context, model=model)
  return None


def resolve_deferred_tools(
  context_manager: "Optional[ContextManager]",
  tools_dict: "Dict[str, Function]",
) -> "Optional[DeferredToolManager]":
  """Create a DeferredToolManager if deferred_tools is enabled."""
  if context_manager is None:
    return None
  if not context_manager.config.deferred_tools:
    return None

  from definable.agent.context.deferred import DeferredToolManager

  return DeferredToolManager(tools_dict)


def resolve_compression(
  compression: "Union[bool, Compression, None]",
  model: "Model",
) -> "Optional[CompressionManager]":
  """Resolve compression param into a CompressionManager (or None)."""
  from definable.agent.compression import Compression as _Compression

  if compression is True:
    return build_compression_manager(_Compression(), model)
  if isinstance(compression, _Compression):
    return build_compression_manager(compression, model)
  return None


def build_compression_manager(compression: "Compression", model: "Model") -> "CompressionManager":
  """Build a CompressionManager from a Compression config."""
  from definable.agent.compression import CompressionManager

  compression_model: "Optional[Model]" = None
  if compression.model is None:
    compression_model = model
  elif isinstance(compression.model, str):
    from definable.model.utils import resolve_model_string

    compression_model = resolve_model_string(compression.model)
  else:
    compression_model = compression.model

  return CompressionManager(
    model=compression_model,
    compress_tool_results=True,
    compress_tool_results_limit=compression.tool_results_limit,
    compress_token_limit=compression.token_limit,
    compress_tool_call_instructions=compression.instructions,
    compress_single_result_size=compression.single_result_size,
  )


# ---------------------------------------------------------------------------
# Readers / skills / tools init
# ---------------------------------------------------------------------------


def init_readers(readers: "Union[List[BaseReader], BaseReader, bool, None]") -> "Optional[BaseReader]":
  """Resolve the readers= parameter into a BaseReader or None."""
  if readers is None or readers is False:
    return None
  if readers is True:
    from definable.reader import BaseReader

    return BaseReader()
  from definable.reader.parsers.base_parser import BaseParser

  if isinstance(readers, BaseParser):
    from definable.reader import BaseReader
    from definable.reader.registry import ParserRegistry

    registry = ParserRegistry(include_defaults=False)
    registry.register(readers)
    return BaseReader(registry=registry)
  return readers  # type: ignore[return-value]


def init_skills(skills: "List[Skill]") -> None:
  """Initialize skills: call setup(), validate names. Mutates skills in-place."""
  seen_names: Dict[str, Any] = {}
  for skill in skills:
    if skill.name in seen_names:
      from definable.utils.log import log_warning

      log_warning(f"Duplicate skill name '{skill.name}' — tools from the later skill will override earlier ones.")
    seen_names[skill.name] = skill

    try:
      skill.setup()
      skill._initialized = True
    except Exception as e:
      from definable.utils.log import log_error

      log_error(f"Skill '{skill.name}' setup() failed: {e}")


def flatten_tools(
  skills: "List[Skill]",
  toolkits: "List[Toolkit]",
  tools: "List[Function]",
) -> "Dict[str, Function]":
  """Flatten tools from skills, toolkits, and direct tools into a single dict.

  Processing order (later entries override earlier ones):
    1. Skill tools (lowest priority)
    2. Toolkit tools
    3. Direct tools (highest priority — explicit always wins)
  """
  result: "Dict[str, Function]" = {}

  for skill in skills:
    try:
      skill_tools = skill.tools
    except Exception:
      skill_tools = []
    for fn in skill_tools:
      if skill.dependencies:
        existing_deps = getattr(fn, "_dependencies", None) or {}
        fn._dependencies = {**existing_deps, **skill.dependencies}
      result[fn.name] = fn

  for toolkit in toolkits:
    for fn in toolkit.tools:
      if toolkit.dependencies:
        existing_deps = getattr(fn, "_dependencies", None) or {}
        fn._dependencies = {**existing_deps, **toolkit.dependencies}
      result[fn.name] = fn

  for fn in tools:
    result[fn.name] = fn

  return result
