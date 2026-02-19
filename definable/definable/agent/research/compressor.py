"""CKU extraction — compresses raw page content into structured facts."""

import asyncio
import json
from typing import TYPE_CHECKING, List, Optional

from definable.agent.research.models import CKU, Fact, PageContent
from definable.agent.research.prompts import CKU_EXTRACTION_PROMPT
from definable.utils.log import log_debug, log_warning

if TYPE_CHECKING:
  from definable.model.base import Model


async def compress(
  page: PageContent,
  sub_question: str,
  original_query: str,
  model: "Model",
) -> Optional[CKU]:
  """Extract structured facts from a single page.

  Calls the model with the CKU extraction prompt and parses the JSON response.
  Returns None on parse failure or if the page is irrelevant.

  Args:
    page: Raw page content.
    sub_question: The specific sub-question being researched.
    original_query: The original user query.
    model: Model instance for LLM calls (should be cheap/fast).

  Returns:
    CKU with extracted facts, or None on failure.
  """
  if page.error or not page.content.strip():
    return None

  from definable.model.message import Message

  prompt = CKU_EXTRACTION_PROMPT.format(
    sub_question=sub_question,
    original_query=original_query,
    page_content=page.content[:12000],  # Hard limit on input to model
  )

  try:
    response = await model.ainvoke(
      messages=[Message(role="user", content=prompt)],
      assistant_message=Message(role="assistant"),
    )
    content = response.content or ""
    # Strip markdown code fences if present
    if content.startswith("```"):
      content = content.split("\n", 1)[-1]
      if content.endswith("```"):
        content = content[:-3]
      content = content.strip()

    data = json.loads(content)
  except (json.JSONDecodeError, Exception) as e:
    log_warning(f"CKU extraction failed for {page.url}: {e}")
    return None

  # Parse facts
  facts: List[Fact] = []
  for f in data.get("facts", []):
    facts.append(
      Fact(
        content=f.get("content", ""),
        fact_type=f.get("fact_type", "factual"),
        confidence=float(f.get("confidence", 0.8)),
        entities=f.get("entities", []),
        source_sentence=f.get("source_sentence", ""),
      )
    )

  relevance = float(data.get("relevance_score", 0.0))
  raw_tokens = len(page.content.split())
  compressed_tokens = sum(len(f.content.split()) for f in facts)
  compression_ratio = compressed_tokens / max(raw_tokens, 1)

  log_debug(f"Extracted {len(facts)} facts from {page.url} (relevance={relevance:.2f})")

  return CKU(
    source_url=page.url,
    source_title=page.title,
    query_context=sub_question,
    facts=facts,
    relevance_score=relevance,
    raw_token_count=raw_tokens,
    compressed_token_count=compressed_tokens,
    compression_ratio=compression_ratio,
    page_summary=data.get("page_summary", ""),
    suggested_followup=data.get("suggested_followup", ""),
  )


async def compress_batch(
  pages: List[PageContent],
  sub_question: str,
  original_query: str,
  model: "Model",
  *,
  max_concurrent: int = 5,
) -> List[CKU]:
  """Extract CKUs from multiple pages concurrently.

  Args:
    pages: List of raw page contents.
    sub_question: The specific sub-question being researched.
    original_query: The original user query.
    model: Model instance for LLM calls.
    max_concurrent: Maximum concurrent compression calls.

  Returns:
    List of successfully extracted CKUs (Nones filtered out).
  """
  semaphore = asyncio.Semaphore(max_concurrent)

  async def _compress_one(page: PageContent) -> Optional[CKU]:
    async with semaphore:
      return await compress(page, sub_question, original_query, model)

  results = await asyncio.gather(*[_compress_one(p) for p in pages])
  return [cku for cku in results if cku is not None]
