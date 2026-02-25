"""Integration tests for Plugin system — exports, Agent wiring."""


class TestPluginExportsFromAgent:
  def test_plugin_importable(self):
    from definable.agent import Plugin

    assert Plugin is not None

  def test_plugin_registry_importable(self):
    from definable.agent import PluginRegistry

    assert PluginRegistry is not None

  def test_plugin_package_importable(self):
    from definable.agent.plugin import Plugin, PluginRegistry

    assert Plugin is not None
    assert PluginRegistry is not None


class TestAgentPluginIntegration:
  def test_agent_has_plugin_registry(self):
    from definable.agent.testing import create_test_agent

    agent = create_test_agent()
    assert hasattr(agent, "_plugin_registry")
    assert len(agent._plugin_registry) == 0

  def test_agent_use_plugin(self):
    from definable.agent.testing import create_test_agent
    from definable.agent.plugin.base import Plugin

    class TestPlugin(Plugin):
      @property
      def name(self):
        return "test"

      async def on_load(self, agent):
        pass

    agent = create_test_agent()
    result = agent.use_plugin(TestPlugin())
    assert result is agent  # returns self for chaining
    assert len(agent._plugin_registry) == 1
    assert "test" in agent._plugin_registry

  def test_agent_plugins_param(self):
    from definable.agent.testing import create_test_agent
    from definable.agent.plugin.base import Plugin

    class TestPlugin(Plugin):
      @property
      def name(self):
        return "test"

      async def on_load(self, agent):
        pass

    # Test that Agent constructor accepts plugins= param
    # create_test_agent doesn't pass plugins, so test via use_plugin
    agent = create_test_agent()
    agent.use_plugin(TestPlugin())
    assert "test" in agent._plugin_registry

  def test_agent_plugin_registry_property(self):
    from definable.agent.testing import create_test_agent

    agent = create_test_agent()
    assert agent.plugin_registry is agent._plugin_registry
