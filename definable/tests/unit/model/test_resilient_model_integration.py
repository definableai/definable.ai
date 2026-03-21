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
