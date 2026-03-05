"""Tests for Cortex TagIndex."""

import pytest
from definable.memory.cortex.index.tags import TagIndex


@pytest.fixture
async def tag_index(tmp_path):
  import aiosqlite

  db = await aiosqlite.connect(str(tmp_path / "tags.db"))
  idx = TagIndex()
  await idx.initialize(db)
  yield idx
  await db.close()


@pytest.mark.asyncio
class TestTagIndex:
  async def test_add_and_search_exact(self, tag_index):
    await tag_index.add_tags("r1", ["work/project/alpha"])
    results = await tag_index.search_exact("work/project/alpha")
    assert "r1" in results

  async def test_hierarchy_expansion(self, tag_index):
    await tag_index.add_tags("r1", ["work/project/deadlines"])
    # Should find r1 under parent tags too
    assert "r1" in await tag_index.search_exact("work")
    assert "r1" in await tag_index.search_exact("work/project")
    assert "r1" in await tag_index.search_exact("work/project/deadlines")

  async def test_prefix_search(self, tag_index):
    await tag_index.add_tags("r1", ["work/project/alpha"])
    await tag_index.add_tags("r2", ["work/project/beta"])
    await tag_index.add_tags("r3", ["personal/hobbies"])
    results = await tag_index.search_prefix("work")
    assert "r1" in results
    assert "r2" in results
    assert "r3" not in results

  async def test_get_tags(self, tag_index):
    await tag_index.add_tags("r1", ["work/project", "technical/python"])
    tags = await tag_index.get_tags("r1")
    assert "work" in tags
    assert "work/project" in tags
    assert "technical" in tags
    assert "technical/python" in tags

  async def test_remove_all_tags(self, tag_index):
    await tag_index.add_tags("r1", ["work", "personal"])
    await tag_index.remove_tags("r1")
    tags = await tag_index.get_tags("r1")
    assert tags == []

  async def test_remove_specific_tag(self, tag_index):
    await tag_index.add_tags("r1", ["work", "personal"])
    await tag_index.remove_tags("r1", ["work"])
    tags = await tag_index.get_tags("r1")
    assert "work" not in tags
    assert "personal" in tags

  async def test_get_all_tags(self, tag_index):
    await tag_index.add_tags("r1", ["work/a"])
    await tag_index.add_tags("r2", ["personal/b"])
    all_tags = await tag_index.get_all_tags()
    assert "work" in all_tags
    assert "work/a" in all_tags
    assert "personal" in all_tags

  async def test_count_by_tag(self, tag_index):
    await tag_index.add_tags("r1", ["work"])
    await tag_index.add_tags("r2", ["work"])
    await tag_index.add_tags("r3", ["personal"])
    assert await tag_index.count_by_tag("work") == 2
    assert await tag_index.count_by_tag("personal") == 1

  async def test_duplicate_tags_ignored(self, tag_index):
    await tag_index.add_tags("r1", ["work", "work"])
    tags = await tag_index.get_tags("r1")
    assert tags.count("work") == 1
