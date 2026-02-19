"""Query decomposition — breaks complex queries into sub-questions."""

import json
from typing import TYPE_CHECKING, List

from definable.agent.research.prompts import DECOMPOSE_PROMPT
from definable.utils.log import log_debug, log_warning

if TYPE_CHECKING:
  from definable.model.base import Model


async def decompose(query: str, model: "Model") -> List[str]:
  """Decompose a complex query into focused sub-questions.

  Calls the model with the decomposition prompt and parses the JSON array.
  Falls back to [query] on parse failure.

  Args:
    query: The original user query.
    model: Model instance for LLM call (typically the agent's main model).

  Returns:
    List of 3-7 sub-questions, or [query] on failure.
  """
  from definable.model.message import Message

  prompt = DECOMPOSE_PROMPT.format(query=query)

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

    sub_questions = json.loads(content)
    if isinstance(sub_questions, list) and all(isinstance(q, str) for q in sub_questions):
      log_debug(f"Decomposed query into {len(sub_questions)} sub-questions")
      return sub_questions
  except (json.JSONDecodeError, Exception) as e:
    log_warning(f"Query decomposition failed, using original query: {e}")

  return [query]
