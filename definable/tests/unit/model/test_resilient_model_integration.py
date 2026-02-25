"""Integration tests for model resilience exports and Agent wiring."""


class TestModelExports:
  def test_resilient_model_importable(self):
    from definable.model import ResilientModel

    assert ResilientModel is not None

  def test_key_pool_importable(self):
    from definable.model import KeyPool

    assert KeyPool is not None

  def test_failover_chain_importable(self):
    from definable.model import FailoverChain

    assert FailoverChain is not None

  def test_failover_entry_importable(self):
    from definable.model import FailoverEntry

    assert FailoverEntry is not None

  def test_resilience_package_importable(self):
    from definable.model.resilience import ResilientModel, KeyPool, FailoverChain

    assert ResilientModel is not None
    assert KeyPool is not None
    assert FailoverChain is not None

  def test_events_importable(self):
    from definable.model.resilience.events import KeyRotatedEvent, ProviderFailoverEvent

    assert KeyRotatedEvent is not None
    assert ProviderFailoverEvent is not None


class TestAgentUsageTracking:
  def test_usage_none_by_default(self):
    from definable.agent.testing import create_test_agent

    agent = create_test_agent()
    assert agent.usage_tracker is None

  def test_usage_true_creates_tracker(self):
    from definable.agent.testing import create_test_agent
    from definable.agent.usage import UsageTracker

    agent = create_test_agent(usage=True)
    assert agent.usage_tracker is not None
    assert isinstance(agent.usage_tracker, UsageTracker)

  def test_usage_tracker_instance_accepted(self):
    from definable.agent.testing import create_test_agent
    from definable.agent.usage import UsageTracker

    tracker = UsageTracker()
    agent = create_test_agent(usage=tracker)
    assert agent.usage_tracker is tracker

  def test_usage_tracker_importable_from_agent(self):
    from definable.agent import UsageTracker, UsageSnapshot

    assert UsageTracker is not None
    assert UsageSnapshot is not None
