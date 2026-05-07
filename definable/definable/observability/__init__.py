"""definable.observability — event-bus subscribers for live observability.

The harness has no built-in observability — it has an EventBus. This
module supplies subscribers that pipe those events somewhere (JSONL on
disk for now; dashboard or OTel-style sinks later).

Usage::

    from definable import Agent
    from definable.observability import Observability

    agent = Agent(name="t", model="...", observability=True)  # auto JSONL
    # or
    obs = Observability(agent.events, jsonl_path=".definable/traces/t.jsonl")
"""

from definable.observability.subscriber import Observability, attach_jsonl

__all__ = ["Observability", "attach_jsonl"]
