"""build_messages — system prompt assembly tests."""

from __future__ import annotations

import pytest

from definable.agent.core.messages import build_messages


@pytest.mark.unit
def test_user_input_only_produces_single_user_message() -> None:
  msgs = build_messages(
    instructions=None,
    memory_index=None,
    skill_descriptions=None,
    user_input="hello",
  )
  assert len(msgs) == 1
  assert msgs[0].role == "user"
  assert msgs[0].content == "hello"


@pytest.mark.unit
def test_instructions_only_produces_system_plus_user() -> None:
  msgs = build_messages(
    instructions="You are a helpful assistant.",
    memory_index=None,
    skill_descriptions=None,
    user_input="hi",
  )
  assert len(msgs) == 2
  assert msgs[0].role == "system"
  assert msgs[0].content == "You are a helpful assistant."
  assert msgs[1].role == "user"


@pytest.mark.unit
def test_full_assembly_with_memory_and_skills() -> None:
  msgs = build_messages(
    instructions="Be concise.",
    memory_index="- profile.md: user preferences\n- recent.md: last week",
    skill_descriptions=["search-web: hits Google", "code-search: ripgreps the repo"],
    user_input="What did we talk about last week?",
  )
  assert len(msgs) == 2

  system = msgs[0]
  assert system.role == "system"
  content = system.content
  assert isinstance(content, str)
  assert "Be concise." in content
  assert "# Available Memory" in content
  assert "profile.md" in content
  assert "# Available Skills" in content
  assert "search-web" in content
  assert "code-search" in content


@pytest.mark.unit
def test_empty_skill_strings_are_filtered() -> None:
  msgs = build_messages(
    instructions="hi",
    memory_index=None,
    skill_descriptions=["", "  ", "one"],
    user_input="x",
  )
  system = msgs[0]
  content = system.content
  assert isinstance(content, str)
  assert "one" in content
  # Empty entries collapse — no double newlines from blanks
  assert "Available Skills\n\n" not in content
