"""Data models for the deep research pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Fact:
  """A single extracted fact from a web page."""

  content: str
  fact_type: str = "factual"  # factual, statistical, opinion, definition
  confidence: float = 0.8
  entities: List[str] = field(default_factory=list)
  source_sentence: str = ""
  contradicts_expectation: bool = False


@dataclass
class CKU:
  """Compressed Knowledge Unit — structured extraction from a single page."""

  source_url: str
  source_title: str
  query_context: str
  facts: List[Fact] = field(default_factory=list)
  relevance_score: float = 0.0
  raw_token_count: int = 0
  compressed_token_count: int = 0
  compression_ratio: float = 0.0
  page_summary: str = ""
  suggested_followup: str = ""


@dataclass
class PageContent:
  """Raw page content from the reader."""

  url: str
  title: str = ""
  content: str = ""
  error: Optional[str] = None


@dataclass
class SourceInfo:
  """Metadata about a source used in research."""

  url: str
  title: str = ""
  relevance_score: float = 0.0
  fact_count: int = 0


@dataclass
class TopicGap:
  """Gap assessment for a sub-topic."""

  topic: str
  status: str = "missing"  # sufficient, partial, missing
  confidence: float = 0.0
  suggested_queries: List[str] = field(default_factory=list)


@dataclass
class Contradiction:
  """Two facts that contradict each other."""

  fact_a: Fact
  fact_b: Fact
  source_a: str = ""
  source_b: str = ""


@dataclass
class ResearchMetrics:
  """Performance metrics for a research run."""

  total_time_ms: float = 0.0
  total_sources_found: int = 0
  total_sources_read: int = 0
  total_facts_extracted: int = 0
  unique_facts: int = 0
  contradictions_found: int = 0
  gaps_identified: int = 0
  gaps_filled: int = 0
  waves_executed: int = 0
  compression_ratio_avg: float = 0.0


@dataclass
class ResearchResult:
  """Complete output of a deep research run."""

  context: str = ""  # Formatted context for system prompt injection
  report: str = ""  # Standalone research report
  sources: List[SourceInfo] = field(default_factory=list)
  facts: List[Fact] = field(default_factory=list)
  gaps: List[TopicGap] = field(default_factory=list)
  contradictions: List[Contradiction] = field(default_factory=list)
  sub_questions: List[str] = field(default_factory=list)
  metrics: ResearchMetrics = field(default_factory=ResearchMetrics)
