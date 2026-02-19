"""Deep research module — multi-wave web research with CKU compression.

Provides the DeepResearch engine for automated research pipelines
and DeepResearchConfig for controlling research behavior.

Example:
  from definable.agent.research import DeepResearch, DeepResearchConfig
  from definable.agent.research.search import create_search_provider

  researcher = DeepResearch(
      model=my_model,
      search_provider=create_search_provider("duckduckgo"),
      config=DeepResearchConfig(depth="deep"),
  )
  result = await researcher.arun("What are the latest developments in quantum computing?")
  print(result.context)  # Formatted context for system prompt injection
  print(result.metrics)  # Performance metrics
"""

from definable.agent.research.config import DeepResearchConfig
from definable.agent.research.engine import DeepResearch
from definable.agent.research.models import (
  CKU,
  Contradiction,
  Fact,
  PageContent,
  ResearchMetrics,
  ResearchResult,
  SourceInfo,
  TopicGap,
)

__all__ = [
  "DeepResearch",
  "DeepResearchConfig",
  "ResearchResult",
  "ResearchMetrics",
  "CKU",
  "Fact",
  "PageContent",
  "SourceInfo",
  "TopicGap",
  "Contradiction",
]
