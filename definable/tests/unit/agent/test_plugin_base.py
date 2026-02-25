"""Tests for Plugin base class and Protocol."""

import pytest

from definable.agent.plugin.base import Plugin


class DummyPlugin(Plugin):
  """Minimal concrete plugin for testing."""

  @property
  def name(self) -> str:
    return "dummy"

  async def on_load(self, agent):
    pass


class FullPlugin(Plugin):
  """Plugin with all optional properties set."""

  @property
  def name(self) -> str:
    return "full-plugin"

  @property
  def version(self) -> str:
    return "1.2.3"

  @property
  def description(self) -> str:
    return "A full test plugin"

  @property
  def requires(self):
    return frozenset({"dep-a"})

  @property
  def conflicts(self):
    return frozenset({"bad-plugin"})

  @property
  def modifies(self):
    return frozenset({"invoke_loop", "store"})

  async def on_load(self, agent):
    self.loaded = True

  async def on_unload(self, agent):
    self.unloaded = True


class TestPluginDefaults:
  def test_default_version(self):
    p = DummyPlugin()
    assert p.version == "0.1.0"

  def test_default_description(self):
    p = DummyPlugin()
    assert p.description == ""

  def test_default_requires(self):
    p = DummyPlugin()
    assert p.requires == frozenset()

  def test_default_conflicts(self):
    p = DummyPlugin()
    assert p.conflicts == frozenset()

  def test_default_modifies(self):
    p = DummyPlugin()
    assert p.modifies == frozenset()

  def test_default_on_unload_is_noop(self):
    """on_unload default is a no-op (doesn't raise)."""
    import asyncio

    p = DummyPlugin()
    asyncio.get_event_loop().run_until_complete(p.on_unload(None))


class TestPluginProperties:
  def test_name(self):
    p = FullPlugin()
    assert p.name == "full-plugin"

  def test_version(self):
    p = FullPlugin()
    assert p.version == "1.2.3"

  def test_description(self):
    p = FullPlugin()
    assert p.description == "A full test plugin"

  def test_requires(self):
    p = FullPlugin()
    assert "dep-a" in p.requires

  def test_conflicts(self):
    p = FullPlugin()
    assert "bad-plugin" in p.conflicts

  def test_modifies(self):
    p = FullPlugin()
    assert "invoke_loop" in p.modifies
    assert "store" in p.modifies


class TestPluginSerialization:
  def test_to_dict(self):
    p = FullPlugin()
    d = p.to_dict()
    assert d["name"] == "full-plugin"
    assert d["version"] == "1.2.3"
    assert d["description"] == "A full test plugin"
    assert d["requires"] == ["dep-a"]
    assert d["conflicts"] == ["bad-plugin"]
    assert sorted(d["modifies"]) == ["invoke_loop", "store"]

  def test_to_dict_defaults(self):
    p = DummyPlugin()
    d = p.to_dict()
    assert d["name"] == "dummy"
    assert d["version"] == "0.1.0"
    assert d["requires"] == []
    assert d["conflicts"] == []
    assert d["modifies"] == []


class TestPluginRepr:
  def test_repr(self):
    p = DummyPlugin()
    assert "DummyPlugin" in repr(p)
    assert "dummy" in repr(p)
    assert "0.1.0" in repr(p)

  def test_repr_full(self):
    p = FullPlugin()
    assert "FullPlugin" in repr(p)
    assert "full-plugin" in repr(p)
    assert "1.2.3" in repr(p)


class TestPluginABC:
  def test_cannot_instantiate_abstract(self):
    """Plugin is abstract and cannot be instantiated directly."""
    with pytest.raises(TypeError):
      Plugin()  # type: ignore[abstract]
