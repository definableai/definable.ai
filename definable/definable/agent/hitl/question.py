"""Question tool — lets the agent ask the user questions mid-run.

The ``build_ask_user_tool()`` factory creates a regular ``Function`` that
wraps the caller-supplied ``QuestionResolver`` callback.  The agent sees
it as an ordinary tool; no special loop mechanics are required.
"""

import json as _json
from typing import List

from definable.agent.hitl.types import Answer, Question, QuestionOption, QuestionResolver
from definable.tool.function import Function


def build_ask_user_tool(resolver: QuestionResolver) -> Function:
  """Build the ``ask_user`` tool backed by *resolver*.

  Follows the same injection pattern as ``_build_spawn_agent_function``
  in ``pipeline/sub_agent.py``.
  """

  async def ask_user(questions_json: str) -> str:
    """Ask the user one or more questions and get their answers.

    Args:
      questions_json: A JSON array of question objects. Each object has:
        - text (str, required): The question to ask.
        - header (str, optional): Short label.
        - options (list of {label, description}, optional): Multiple-choice options.
        - allow_multiple (bool, optional): Allow selecting more than one option.
        - allow_custom (bool, optional): Allow a free-text answer.

    Returns:
      Formatted answers from the user.
    """
    raw_questions = _json.loads(questions_json)
    if isinstance(raw_questions, dict):
      raw_questions = [raw_questions]

    parsed: List[Question] = []
    for q in raw_questions:
      options = None
      if q.get("options"):
        options = [QuestionOption(label=o["label"], description=o.get("description")) for o in q["options"]]
      parsed.append(
        Question(
          text=q["text"],
          header=q.get("header"),
          options=options,
          allow_multiple=q.get("allow_multiple", False),
          allow_custom=q.get("allow_custom", False),
        )
      )

    answers: List[Answer] = await resolver(parsed)

    parts: list[str] = []
    for i, answer in enumerate(answers):
      q_text = parsed[i].text if i < len(parsed) else "?"
      if answer.selected:
        parts.append(f"Q: {q_text}\nA: {', '.join(answer.selected)}")
      elif answer.custom_text:
        parts.append(f"Q: {q_text}\nA: {answer.custom_text}")
      else:
        parts.append(f"Q: {q_text}\nA: (no answer)")
    return "\n\n".join(parts)

  return Function.from_callable(ask_user, name="ask_user")
