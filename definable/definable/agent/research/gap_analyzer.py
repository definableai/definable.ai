"""Coverage analysis — identifies gaps and generates follow-up queries."""

import json
from typing import TYPE_CHECKING, List, Tuple

from definable.agent.research.knowledge_graph import KnowledgeGraph
from definable.agent.research.models import TopicGap
from definable.agent.research.prompts import GAP_ANALYSIS_PROMPT
from definable.utils.log import log_debug, log_warning

if TYPE_CHECKING:
  from definable.model.base import Model


async def analyze(
  query: str,
  sub_questions: List[str],
  knowledge_graph: KnowledgeGraph,
  model: "Model",
) -> Tuple[List[TopicGap], List[str]]:
  """Analyze research coverage and generate gap-filling queries.

  Args:
    query: Original user query.
    sub_questions: List of research sub-questions.
    knowledge_graph: Current accumulated knowledge.
    model: Model for LLM call (cheap/fast preferred).

  Returns:
    Tuple of (gap assessments, new search queries to fill gaps).
  """
  from definable.model.message import Message

  # Build coverage summary
  lines = []
  for sq in sub_questions:
    count = knowledge_graph.fact_count_for_topic(sq)
    facts = knowledge_graph.get_facts_by_topic(sq)
    fact_preview = "; ".join(f.content[:80] for f in facts[:3])
    lines.append(f'- "{sq}": {count} facts. Preview: {fact_preview or "(none)"}')
  coverage_summary = "\n".join(lines)

  prompt = GAP_ANALYSIS_PROMPT.format(
    query=query,
    coverage_summary=coverage_summary,
  )

  try:
    response = await model.ainvoke(
      messages=[Message(role="user", content=prompt)],
      assistant_message=Message(role="assistant"),
    )
    content = response.content or ""
    # Strip markdown code fences
    if content.startswith("```"):
      content = content.split("\n", 1)[-1]
      if content.endswith("```"):
        content = content[:-3]
      content = content.strip()

    data = json.loads(content)
  except (json.JSONDecodeError, Exception) as e:
    log_warning(f"Gap analysis failed: {e}")
    return [], []

  gaps: List[TopicGap] = []
  for assessment in data.get("assessments", []):
    gaps.append(
      TopicGap(
        topic=assessment.get("topic", ""),
        status=assessment.get("status", "missing"),
        confidence=float(assessment.get("confidence", 0.0)),
        suggested_queries=assessment.get("suggested_queries", []),
      )
    )

  new_queries = data.get("new_queries", [])
  log_debug(f"Gap analysis: {len(gaps)} topics assessed, {len(new_queries)} new queries")
  return gaps, new_queries
