"""Research synthesis — formats accumulated knowledge into context blocks."""

from typing import TYPE_CHECKING, List

from definable.agent.research.config import DeepResearchConfig
from definable.agent.research.knowledge_graph import KnowledgeGraph
from definable.agent.research.prompts import SYNTHESIS_PROMPT
from definable.utils.log import log_debug, log_warning

if TYPE_CHECKING:
  from definable.model.base import Model


def _format_facts_for_prompt(kg: KnowledgeGraph, sub_questions: List[str]) -> str:
  """Organize facts by topic for the synthesis prompt."""
  sections = []
  for sq in sub_questions:
    facts = kg.get_facts_by_topic(sq)
    if facts:
      fact_lines = []
      for f in facts:
        citation = f" (confidence: {f.confidence:.1f})" if f.confidence < 0.9 else ""
        fact_lines.append(f"  - {f.content}{citation}")
      sections.append(f"### {sq}\n" + "\n".join(fact_lines))
  return "\n\n".join(sections) if sections else "(no facts collected)"


def _format_contradictions(kg: KnowledgeGraph) -> str:
  """Format contradictions for the synthesis prompt."""
  contradictions = kg.get_contradictions()
  if not contradictions:
    return ""
  lines = ["Contradictions found:"]
  for c in contradictions:
    lines.append(f'- "{c.fact_a.content}" vs "{c.fact_b.content}"')
  return "\n".join(lines)


async def synthesize(
  query: str,
  knowledge_graph: KnowledgeGraph,
  sub_questions: List[str],
  config: DeepResearchConfig,
  model: "Model",
) -> str:
  """Synthesize accumulated research into a formatted context block.

  Args:
    query: Original user query.
    knowledge_graph: Accumulated research knowledge.
    sub_questions: List of research sub-questions.
    config: Research configuration.
    model: Model for synthesis LLM call.

  Returns:
    Formatted context string for system prompt injection.
  """
  from definable.model.message import Message

  organized_facts = _format_facts_for_prompt(knowledge_graph, sub_questions)
  contradictions_section = _format_contradictions(knowledge_graph) if config.include_contradictions else ""

  prompt = SYNTHESIS_PROMPT.format(
    query=query,
    organized_facts=organized_facts,
    source_count=len(knowledge_graph.get_sources()),
    fact_count=knowledge_graph.total_facts,
    contradictions_section=contradictions_section,
    format=config.context_format,
    max_tokens=config.max_context_tokens,
  )

  try:
    response = await model.ainvoke(
      messages=[Message(role="user", content=prompt)],
      assistant_message=Message(role="assistant"),
    )
    context = response.content or ""
    log_debug(f"Synthesized research context: {len(context)} chars")
    return context.strip()
  except Exception as e:
    log_warning(f"Research synthesis failed, using fallback: {e}")
    return _fallback_context(query, knowledge_graph, sub_questions, config)


def _fallback_context(
  query: str,
  kg: KnowledgeGraph,
  sub_questions: List[str],
  config: DeepResearchConfig,
) -> str:
  """Generate a simple context block without LLM synthesis."""
  if config.context_format == "xml":
    return _fallback_xml(query, kg, sub_questions, config)
  return _fallback_markdown(query, kg, sub_questions, config)


def _fallback_xml(
  query: str,
  kg: KnowledgeGraph,
  sub_questions: List[str],
  config: DeepResearchConfig,
) -> str:
  parts = [f'<research_context query="{query}">']
  for sq in sub_questions:
    facts = kg.get_facts_by_topic(sq)
    if facts:
      parts.append(f'  <topic question="{sq}">')
      for f in facts:
        parts.append(f'    <fact confidence="{f.confidence:.1f}">{f.content}</fact>')
      parts.append("  </topic>")
  if config.include_citations:
    parts.append("  <sources>")
    for s in kg.get_sources():
      parts.append(f'    <source url="{s.url}" title="{s.title}" facts="{s.fact_count}"/>')
    parts.append("  </sources>")
  parts.append("</research_context>")
  return "\n".join(parts)


def _fallback_markdown(
  query: str,
  kg: KnowledgeGraph,
  sub_questions: List[str],
  config: DeepResearchConfig,
) -> str:
  parts = [f"## Research Context: {query}\n"]
  for sq in sub_questions:
    facts = kg.get_facts_by_topic(sq)
    if facts:
      parts.append(f"### {sq}")
      for f in facts:
        parts.append(f"- {f.content}")
      parts.append("")
  if config.include_citations:
    parts.append("### Sources")
    for s in kg.get_sources():
      parts.append(f"- [{s.title}]({s.url}) ({s.fact_count} facts)")
  return "\n".join(parts)
