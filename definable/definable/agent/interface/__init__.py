"""agent.interface — channel adapters.

The base class is `Interface`. Each channel subpackage (telegram, whatsapp,
discord, email, desktop, call, websocket, slack) supplies a concrete
implementation that subclasses Interface and wires up the platform-specific
receive/send glue.

Usage::

    from definable import Agent
    from definable.agent.interface.websocket import WebSocketInterface

    agent = Agent(name="ws", model="...")
    iface = WebSocketInterface(agent, host="0.0.0.0", port=8765)
    async with iface:
      await iface.serve()
"""

from definable.agent.interface.base import Interface

__all__ = ["Interface"]
