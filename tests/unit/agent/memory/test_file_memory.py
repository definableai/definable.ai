"""FileMemory + memory_tools — markdown-file memory tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from definable.agent.memory import FileMemory, build_index, memory_tools


@pytest.mark.unit
def test_write_and_read_roundtrip(tmp_path: Path) -> None:
  m = FileMemory(tmp_path)
  m.write("profile", "name: hash\nrole: principal engineer")
  assert m.read("profile") == "name: hash\nrole: principal engineer"
  assert (tmp_path / "profile.md").exists()


@pytest.mark.unit
def test_read_missing_raises(tmp_path: Path) -> None:
  m = FileMemory(tmp_path)
  with pytest.raises(FileNotFoundError):
    m.read("nope")


@pytest.mark.unit
def test_list_returns_stems_sorted(tmp_path: Path) -> None:
  m = FileMemory(tmp_path)
  m.write("zeta", "z")
  m.write("alpha", "a")
  m.write("mu", "m")
  assert m.names() == ["alpha", "mu", "zeta"]


@pytest.mark.unit
def test_search_finds_substring_with_snippet(tmp_path: Path) -> None:
  m = FileMemory(tmp_path)
  m.write("notes", "long content here. The KEYWORD lives in the middle. More text.")
  m.write("other", "nothing relevant here")
  results = m.search("keyword")
  assert len(results) == 1
  assert results[0][0] == "notes"
  assert "KEYWORD" in results[0][1]


@pytest.mark.unit
def test_search_empty_query_returns_nothing(tmp_path: Path) -> None:
  m = FileMemory(tmp_path)
  m.write("a", "anything")
  assert m.search("") == []


@pytest.mark.unit
def test_index_uses_curated_when_present(tmp_path: Path) -> None:
  m = FileMemory(tmp_path)
  m.write("INDEX", "# Curated\n- profile: my user")
  m.write("profile", "ignored")
  assert "Curated" in m.index()


@pytest.mark.unit
def test_index_auto_builds_when_missing(tmp_path: Path) -> None:
  m = FileMemory(tmp_path)
  m.write("profile", "user prefers concise output")
  m.write("recent", "last task: build harness")
  index = m.index()
  assert "profile: user prefers concise output" in index
  assert "recent: last task: build harness" in index


@pytest.mark.unit
def test_invalid_names_rejected(tmp_path: Path) -> None:
  m = FileMemory(tmp_path)
  for bad in ["", "../escape", "sub/dir", "back\\slash", ".hidden"]:
    with pytest.raises(ValueError):
      m.write(bad, "x")


@pytest.mark.unit
def test_memory_tools_bound_to_instance(tmp_path: Path) -> None:
  m = FileMemory(tmp_path)
  tools = memory_tools(m)
  assert {t.name for t in tools} == {"read_memory", "write_memory", "list_memories", "search_memory"}


@pytest.mark.unit
def test_build_index_no_memories(tmp_path: Path) -> None:
  m = FileMemory(tmp_path)
  assert build_index(m) == "(no memories)"
