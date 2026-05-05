"""Tests for PluginRegistry — lifecycle, ordering, conflict detection."""

import pytest

from definable.agent.plugin.base import Plugin
from definable.agent.plugin.registry import PluginRegistry


# --- Test plugins ---


class AlphaPlugin(Plugin):
  @property
  def name(self):
    return "alpha"

  async def on_load(self, agent):
    self.loaded = True

  async def on_unload(self, agent):
    self.loaded = False


class BetaPlugin(Plugin):
  @property
  def name(self):
    return "beta"

  @property
  def requires(self):
    return frozenset({"alpha"})

  async def on_load(self, agent):
    self.loaded = True

  async def on_unload(self, agent):
    self.loaded = False


class GammaPlugin(Plugin):
  @property
  def name(self):
    return "gamma"

  @property
  def requires(self):
    return frozenset({"beta"})

  async def on_load(self, agent):
    self.loaded = True


class ConflictPlugin(Plugin):
  @property
  def name(self):
    return "conflict"

  @property
  def conflicts(self):
    return frozenset({"alpha"})

  async def on_load(self, agent):
    pass


class SharedModPlugin(Plugin):
  """Modifies the same phase as another plugin."""

  def __init__(self, plugin_name: str):
    self._name = plugin_name

  @property
  def name(self):
    return self._name

  @property
  def modifies(self):
    return frozenset({"invoke_loop"})

  async def on_load(self, agent):
    pass


# --- Tests ---


class TestRegistryRegistration:
  def test_add_plugin(self):
    reg = PluginRegistry()
    reg.add(AlphaPlugin())
    assert "alpha" in reg
    assert len(reg) == 1

  def test_add_duplicate_raises(self):
    reg = PluginRegistry()
    reg.add(AlphaPlugin())
    with pytest.raises(ValueError, match="already registered"):
      reg.add(AlphaPlugin())

  def test_remove_plugin(self):
    reg = PluginRegistry()
    reg.add(AlphaPlugin())
    removed = reg.remove("alpha")
    assert removed is not None
    assert "alpha" not in reg

  def test_remove_nonexistent(self):
    reg = PluginRegistry()
    assert reg.remove("nope") is None

  def test_remove_loaded_raises(self):
    """Cannot remove a plugin that's still loaded."""
    reg = PluginRegistry()
    reg.add(AlphaPlugin())
    reg._loaded.add("alpha")  # simulate loaded
    with pytest.raises(RuntimeError, match="still loaded"):
      reg.remove("alpha")

  def test_chaining(self):
    reg = PluginRegistry()
    result = reg.add(AlphaPlugin())
    assert result is reg


class TestRegistryLoading:
  @pytest.mark.asyncio
  async def test_load_all(self):
    reg = PluginRegistry()
    alpha = AlphaPlugin()
    reg.add(alpha)
    await reg.load_all(None)  # type: ignore[arg-type]
    assert reg.is_loaded("alpha")
    assert alpha.loaded is True

  @pytest.mark.asyncio
  async def test_load_order_respects_deps(self):
    reg = PluginRegistry()
    reg.add(BetaPlugin())
    reg.add(AlphaPlugin())
    await reg.load_all(None)  # type: ignore[arg-type]
    # Both should be loaded (alpha first due to dep)
    assert reg.loaded_names == ["alpha", "beta"]

  @pytest.mark.asyncio
  async def test_load_three_deep_deps(self):
    reg = PluginRegistry()
    reg.add(GammaPlugin())
    reg.add(AlphaPlugin())
    reg.add(BetaPlugin())
    await reg.load_all(None)  # type: ignore[arg-type]
    assert reg.loaded_names == ["alpha", "beta", "gamma"]

  @pytest.mark.asyncio
  async def test_unload_all(self):
    reg = PluginRegistry()
    alpha = AlphaPlugin()
    reg.add(alpha)
    await reg.load_all(None)  # type: ignore[arg-type]
    await reg.unload_all(None)  # type: ignore[arg-type]
    assert not reg.is_loaded("alpha")
    assert alpha.loaded is False

  @pytest.mark.asyncio
  async def test_load_one(self):
    reg = PluginRegistry()
    reg.add(AlphaPlugin())
    await reg.load_one("alpha", None)  # type: ignore[arg-type]
    assert reg.is_loaded("alpha")

  @pytest.mark.asyncio
  async def test_load_one_missing_dep(self):
    reg = PluginRegistry()
    reg.add(BetaPlugin())
    with pytest.raises(ValueError, match="requires"):
      await reg.load_one("beta", None)  # type: ignore[arg-type]

  @pytest.mark.asyncio
  async def test_load_one_not_registered(self):
    reg = PluginRegistry()
    with pytest.raises(KeyError, match="not registered"):
      await reg.load_one("nope", None)  # type: ignore[arg-type]

  @pytest.mark.asyncio
  async def test_unload_one(self):
    reg = PluginRegistry()
    reg.add(AlphaPlugin())
    await reg.load_all(None)  # type: ignore[arg-type]
    await reg.unload_one("alpha", None)  # type: ignore[arg-type]
    assert not reg.is_loaded("alpha")

  @pytest.mark.asyncio
  async def test_unload_one_with_dependents_raises(self):
    reg = PluginRegistry()
    reg.add(AlphaPlugin())
    reg.add(BetaPlugin())
    await reg.load_all(None)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="depend on it"):
      await reg.unload_one("alpha", None)  # type: ignore[arg-type]

  @pytest.mark.asyncio
  async def test_load_idempotent(self):
    """Loading an already-loaded plugin is a no-op."""
    reg = PluginRegistry()
    alpha = AlphaPlugin()
    reg.add(alpha)
    await reg.load_all(None)  # type: ignore[arg-type]
    # Load again should not raise
    await reg.load_one("alpha", None)  # type: ignore[arg-type]
    assert reg.is_loaded("alpha")


class TestRegistryValidation:
  @pytest.mark.asyncio
  async def test_missing_dependency(self):
    reg = PluginRegistry()
    reg.add(BetaPlugin())  # requires "alpha" which isn't registered
    with pytest.raises(ValueError, match="requires"):
      await reg.load_all(None)  # type: ignore[arg-type]

  @pytest.mark.asyncio
  async def test_conflict_detection(self):
    reg = PluginRegistry()
    reg.add(AlphaPlugin())
    reg.add(ConflictPlugin())
    with pytest.raises(ValueError, match="conflicts"):
      await reg.load_all(None)  # type: ignore[arg-type]

  @pytest.mark.asyncio
  async def test_shared_modifies_warns(self, capsys):
    """Two plugins modifying the same phase should produce a warning."""
    reg = PluginRegistry()
    reg.add(SharedModPlugin("mod-a"))
    reg.add(SharedModPlugin("mod-b"))
    await reg.load_all(None)  # type: ignore[arg-type]
    captured = capsys.readouterr()
    assert "Multiple plugins modify" in captured.out


class TestRegistryCycleDetection:
  @pytest.mark.asyncio
  async def test_cycle_raises(self):
    """A dependency cycle is detected and reported."""

    class CycA(Plugin):
      @property
      def name(self):
        return "cyc-a"

      @property
      def requires(self):
        return frozenset({"cyc-b"})

      async def on_load(self, agent):
        pass

    class CycB(Plugin):
      @property
      def name(self):
        return "cyc-b"

      @property
      def requires(self):
        return frozenset({"cyc-a"})

      async def on_load(self, agent):
        pass

    reg = PluginRegistry()
    reg.add(CycA())
    reg.add(CycB())
    with pytest.raises(ValueError, match="cycle"):
      await reg.load_all(None)  # type: ignore[arg-type]


class TestRegistryEnableDisable:
  def test_disable(self):
    reg = PluginRegistry()
    reg.add(AlphaPlugin())
    reg.disable("alpha")
    assert "alpha" in reg.disabled_names

  @pytest.mark.asyncio
  async def test_disabled_not_loaded(self):
    reg = PluginRegistry()
    reg.add(AlphaPlugin())
    reg.disable("alpha")
    await reg.load_all(None)  # type: ignore[arg-type]
    assert not reg.is_loaded("alpha")

  def test_enable(self):
    reg = PluginRegistry()
    reg.add(AlphaPlugin())
    reg.disable("alpha")
    reg.enable("alpha")
    assert "alpha" not in reg.disabled_names

  @pytest.mark.asyncio
  async def test_disabled_skips_dependency_check(self):
    """Disabled plugins don't trigger missing-dep errors for others."""
    reg = PluginRegistry()
    reg.add(AlphaPlugin())
    reg.add(BetaPlugin())
    reg.add(ConflictPlugin())
    reg.disable("conflict")
    # Should not raise — conflict plugin is disabled
    await reg.load_all(None)  # type: ignore[arg-type]
    assert reg.is_loaded("alpha")
    assert reg.is_loaded("beta")


class TestRegistryIntrospection:
  def test_plugin_names(self):
    reg = PluginRegistry()
    reg.add(AlphaPlugin())
    reg.add(BetaPlugin())
    assert sorted(reg.plugin_names) == ["alpha", "beta"]

  def test_get(self):
    reg = PluginRegistry()
    alpha = AlphaPlugin()
    reg.add(alpha)
    assert reg.get("alpha") is alpha
    assert reg.get("nope") is None

  def test_len(self):
    reg = PluginRegistry()
    assert len(reg) == 0
    reg.add(AlphaPlugin())
    assert len(reg) == 1

  def test_contains(self):
    reg = PluginRegistry()
    reg.add(AlphaPlugin())
    assert "alpha" in reg
    assert "nope" not in reg

  def test_iter(self):
    reg = PluginRegistry()
    a = AlphaPlugin()
    b = BetaPlugin()
    reg.add(a)
    reg.add(b)
    plugins = list(reg)
    assert a in plugins
    assert b in plugins

  def test_repr(self):
    reg = PluginRegistry()
    reg.add(AlphaPlugin())
    assert "PluginRegistry" in repr(reg)
    assert "plugins=1" in repr(reg)
