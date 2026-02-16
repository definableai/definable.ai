"""DeepResearch engine — orchestrates the full multi-wave research pipeline."""

import asyncio
import json
import time
from typing import TYPE_CHECKING, Callable, List, Optional, Set

from definable.research.config import DeepResearchConfig
from definable.research.knowledge_graph import KnowledgeGraph
from definable.research.models import ResearchMetrics, ResearchResult
from definable.research.prompts import NEEDS_RESEARCH_PROMPT
from definable.research.search.base import SearchResult
from definable.utils.log import log_debug, log_info, log_warning

if TYPE_CHECKING:
  from definable.models.base import Model
  from definable.research.search.base import SearchProvider


class DeepResearch:
  """Multi-wave deep research orchestrator.

  Conducts iterative web research: decompose query into sub-questions,
  search, read pages, compress into CKUs, accumulate knowledge,
  detect gaps, and repeat until coverage is sufficient.

  Args:
    model: Model for planning and synthesis (agent's main model).
    search_provider: Search backend for web queries.
    compression_model: Model for CKU extraction (cheap/fast).
        Defaults to the main model if not provided.
    config: Research configuration. Defaults to standard depth.

  Example:
    from definable.research import DeepResearch, DeepResearchConfig
    from definable.research.search import create_search_provider

    researcher = DeepResearch(
        model=my_model,
        search_provider=create_search_provider("duckduckgo"),
    )
    result = await researcher.arun("What are the latest AI safety developments?")
    print(result.context)
  """

  def __init__(
    self,
    model: "Model",
    search_provider: "SearchProvider",
    compression_model: Optional["Model"] = None,
    config: Optional[DeepResearchConfig] = None,
  ):
    self._model = model
    self._search_provider = search_provider
    self._compression_model = compression_model or model
    self._config = (config or DeepResearchConfig()).with_depth_preset()

  async def arun(
    self,
    query: str,
    *,
    on_progress: Optional[Callable] = None,
  ) -> ResearchResult:
    """Execute the full research pipeline.

    Args:
      query: The user's research question.
      on_progress: Optional callback(wave, sources_read, facts_extracted, gaps_remaining, message).

    Returns:
      ResearchResult with context, report, sources, facts, and metrics.
    """
    from definable.research import compressor, gap_analyzer, planner, reader, synthesizer

    start_time = time.perf_counter()
    config = self._config
    kg = KnowledgeGraph()
    seen_urls: Set[str] = set()
    total_sources_found = 0
    total_sources_read = 0
    gaps_identified = 0
    gaps_filled = 0

    # Step 1: Decompose query
    sub_questions = await planner.decompose(query, self._model)
    log_info(f"Deep research: {len(sub_questions)} sub-questions for '{query[:80]}'")

    current_queries = list(sub_questions)
    all_ckus: list = []
    wave = 0

    # Step 2: Multi-wave research
    for wave in range(config.max_waves):
      log_debug(f"Research wave {wave + 1}/{config.max_waves}: {len(current_queries)} queries")

      # Search concurrently
      search_tasks = [
        self._search_provider.search(q, max_results=max(config.max_sources // len(current_queries), 3))
        for q in current_queries[: config.parallel_searches]
      ]
      search_results_lists = await asyncio.gather(*search_tasks, return_exceptions=True)

      # Flatten and deduplicate URLs
      all_results: List[SearchResult] = []
      for sr_list in search_results_lists:
        if isinstance(sr_list, BaseException):
          log_warning(f"Search failed: {sr_list}")
          continue
        for sr in sr_list:
          if sr.url not in seen_urls:
            seen_urls.add(sr.url)
            all_results.append(sr)
            total_sources_found += 1

      if not all_results:
        log_debug(f"Wave {wave + 1}: no new results, stopping")
        break

      # Respect max_sources limit
      remaining = config.max_sources - total_sources_read
      urls_to_read = [r.url for r in all_results[:remaining]]

      if not urls_to_read:
        log_debug(f"Wave {wave + 1}: max_sources reached, stopping")
        break

      # Read pages concurrently
      pages = await reader.read_pages(urls_to_read, max_concurrent=config.parallel_reads)
      total_sources_read += len([p for p in pages if not p.error])

      # Compress: extract CKUs for each sub-question
      all_ckus = []
      for sq in current_queries[: config.parallel_searches]:
        ckus = await compressor.compress_batch(
          pages,
          sq,
          query,
          self._compression_model,
          max_concurrent=5,
        )
        # Filter by min_relevance
        ckus = [c for c in ckus if c.relevance_score >= config.min_relevance]
        all_ckus.extend(ckus)

      # Accumulate into knowledge graph
      new_facts = kg.ingest(all_ckus)
      log_debug(f"Wave {wave + 1}: {new_facts} new facts from {len(all_ckus)} CKUs")

      # Progress callback
      if on_progress:
        on_progress(wave + 1, total_sources_read, kg.total_facts, 0, f"Wave {wave + 1} complete")

      # Gap analysis (skip on last wave)
      if wave < config.max_waves - 1:
        gaps, gap_queries = await gap_analyzer.analyze(
          query,
          sub_questions,
          kg,
          self._compression_model,
        )
        gaps_identified += len([g for g in gaps if g.status != "sufficient"])

        # Early termination check
        if new_facts == 0 or (kg.total_facts > 0 and new_facts / kg.total_facts < config.early_termination_threshold):
          log_debug(f"Wave {wave + 1}: novelty below threshold, stopping early")
          break

        # All topics sufficient?
        if all(g.status == "sufficient" for g in gaps):
          log_debug(f"Wave {wave + 1}: all topics covered, stopping")
          gaps_filled = gaps_identified
          break

        # Use gap queries for next wave
        if gap_queries:
          current_queries = gap_queries
          gaps_filled += len(gap_queries)
        else:
          break

    # Step 3: Synthesize
    context = await synthesizer.synthesize(
      query,
      kg,
      sub_questions,
      config,
      self._model,
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Build metrics
    compression_ratios = [c.compression_ratio for c in all_ckus if c.compression_ratio > 0] if all_ckus else []
    metrics = ResearchMetrics(
      total_time_ms=elapsed_ms,
      total_sources_found=total_sources_found,
      total_sources_read=total_sources_read,
      total_facts_extracted=kg.total_facts,
      unique_facts=kg.total_facts,
      contradictions_found=len(kg.get_contradictions()),
      gaps_identified=gaps_identified,
      gaps_filled=gaps_filled,
      waves_executed=min(wave + 1, config.max_waves),  # noqa: F821 — wave defined in loop
      compression_ratio_avg=sum(compression_ratios) / len(compression_ratios) if compression_ratios else 0.0,
    )

    log_info(
      f"Deep research complete: {metrics.waves_executed} waves, "
      f"{metrics.total_sources_read} sources, {metrics.unique_facts} facts, "
      f"{elapsed_ms:.0f}ms"
    )

    return ResearchResult(
      context=context,
      report=context,  # For now, report = context
      sources=kg.get_sources(),
      facts=kg.get_all_facts(),
      gaps=[],  # Final gaps not re-assessed
      contradictions=kg.get_contradictions(),
      sub_questions=sub_questions,
      metrics=metrics,
    )

  def run(self, query: str) -> ResearchResult:
    """Synchronous wrapper around arun()."""
    return asyncio.run(self.arun(query))

  async def needs_research(self, query: str) -> bool:
    """Classify whether a query needs web research.

    Args:
      query: The user's question.

    Returns:
      True if the query would benefit from web research.
    """
    from definable.models.message import Message

    prompt = NEEDS_RESEARCH_PROMPT.format(query=query)

    try:
      response = await self._model.ainvoke(
        messages=[Message(role="user", content=prompt)],
        assistant_message=Message(role="assistant"),
      )
      content = response.content or ""
      if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        if content.endswith("```"):
          content = content[:-3]
        content = content.strip()

      data = json.loads(content)
      result = bool(data.get("needs_research", True))
      log_debug(f"needs_research('{query[:50]}'): {result} — {data.get('reason', '')}")
      return result
    except Exception as e:
      log_warning(f"needs_research classification failed, defaulting to True: {e}")
      return True
