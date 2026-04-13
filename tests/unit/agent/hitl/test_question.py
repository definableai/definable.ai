"""Tests for HITL question tool."""

import json

from definable.agent.hitl.question import build_ask_user_tool
from definable.agent.hitl.types import Answer, Question


class TestBuildAskUserTool:
  def test_builds_function_with_correct_name(self):
    async def resolver(questions):
      return []

    fn = build_ask_user_tool(resolver)
    assert fn.name == "ask_user"

  def test_function_has_description(self):
    async def resolver(questions):
      return []

    fn = build_ask_user_tool(resolver)
    assert fn.description is not None
    assert "question" in fn.description.lower()

  async def test_formats_selected_answers(self):
    async def resolver(questions):
      return [Answer(question_text="Favorite color?", selected=["blue"])]

    fn = build_ask_user_tool(resolver)
    assert fn.entrypoint is not None
    result = await fn.entrypoint(json.dumps([{"text": "Favorite color?", "options": [{"label": "blue"}, {"label": "red"}]}]))
    assert "Q: Favorite color?" in result
    assert "A: blue" in result

  async def test_formats_custom_text_answer(self):
    async def resolver(questions):
      return [Answer(question_text="Your name?", custom_text="Alice")]

    fn = build_ask_user_tool(resolver)
    assert fn.entrypoint is not None
    result = await fn.entrypoint(json.dumps([{"text": "Your name?"}]))
    assert "A: Alice" in result

  async def test_formats_no_answer(self):
    async def resolver(questions):
      return [Answer(question_text="Anything?")]

    fn = build_ask_user_tool(resolver)
    assert fn.entrypoint is not None
    result = await fn.entrypoint(json.dumps([{"text": "Anything?"}]))
    assert "(no answer)" in result

  async def test_multiple_questions(self):
    async def resolver(questions):
      return [
        Answer(question_text="Q1", selected=["yes"]),
        Answer(question_text="Q2", custom_text="42"),
      ]

    fn = build_ask_user_tool(resolver)
    assert fn.entrypoint is not None
    result = await fn.entrypoint(json.dumps([{"text": "Q1", "options": [{"label": "yes"}]}, {"text": "Q2"}]))
    assert "Q: Q1" in result
    assert "Q: Q2" in result
    assert "A: yes" in result
    assert "A: 42" in result

  async def test_passes_question_options_to_resolver(self):
    received_questions = []

    async def resolver(questions):
      received_questions.extend(questions)
      return [Answer(question_text="Pick", selected=["a"])]

    fn = build_ask_user_tool(resolver)
    assert fn.entrypoint is not None
    await fn.entrypoint(
      json.dumps([
        {
          "text": "Pick one",
          "header": "Choice",
          "options": [{"label": "a", "description": "Option A"}, {"label": "b"}],
          "allow_multiple": True,
        }
      ])
    )

    assert len(received_questions) == 1
    q = received_questions[0]
    assert isinstance(q, Question)
    assert q.text == "Pick one"
    assert q.header == "Choice"
    assert q.allow_multiple is True
    assert q.options is not None
    assert len(q.options) == 2
    assert q.options[0].label == "a"
    assert q.options[0].description == "Option A"
