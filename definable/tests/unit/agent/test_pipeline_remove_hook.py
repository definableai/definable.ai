"""Tests for Pipeline.remove_hook()."""

from definable.agent.pipeline.pipeline import Pipeline


class TestRemoveHook:
  def test_remove_all_hooks_for_spec(self):
    pipeline = Pipeline()

    def cb1(state):
      return state

    def cb2(state):
      return state

    pipeline.hook("before:invoke_loop", cb1)
    pipeline.hook("before:invoke_loop", cb2)
    assert len(pipeline._hooks.get("before:invoke_loop", [])) == 2

    removed = pipeline.remove_hook("before:invoke_loop")
    assert removed is True
    assert "before:invoke_loop" not in pipeline._hooks

  def test_remove_specific_callback(self):
    pipeline = Pipeline()

    def cb1(state):
      return state

    def cb2(state):
      return state

    pipeline.hook("before:invoke_loop", cb1)
    pipeline.hook("before:invoke_loop", cb2)

    removed = pipeline.remove_hook("before:invoke_loop", cb1)
    assert removed is True
    remaining = pipeline._hooks.get("before:invoke_loop", [])
    assert len(remaining) == 1
    assert remaining[0][1] is cb2

  def test_remove_nonexistent_spec(self):
    pipeline = Pipeline()
    removed = pipeline.remove_hook("before:nope")
    assert removed is False

  def test_remove_nonexistent_callback(self):
    pipeline = Pipeline()

    def cb1(state):
      return state

    def cb2(state):
      return state

    pipeline.hook("before:invoke_loop", cb1)
    removed = pipeline.remove_hook("before:invoke_loop", cb2)
    assert removed is False

  def test_remove_last_callback_cleans_key(self):
    """Removing the last callback for a spec removes the key entirely."""
    pipeline = Pipeline()

    def cb(state):
      return state

    pipeline.hook("before:invoke_loop", cb)
    pipeline.remove_hook("before:invoke_loop", cb)
    assert "before:invoke_loop" not in pipeline._hooks

  def test_remove_does_not_affect_other_specs(self):
    pipeline = Pipeline()

    def cb(state):
      return state

    pipeline.hook("before:invoke_loop", cb)
    pipeline.hook("after:invoke_loop", cb)
    pipeline.remove_hook("before:invoke_loop")
    assert "after:invoke_loop" in pipeline._hooks
