import asyncio

from definable.memory import InMemoryStore, Memory
from definable.model.message import Message


async def main() -> None:
  async with Memory(store=InMemoryStore()) as memory:
    await memory.add(Message(role="user", content="My name is Ada."), session_id="docs")
    await memory.add(Message(role="assistant", content="Hello Ada."), session_id="docs")

    entries = await memory.get_entries("docs")

    assert len(entries) == 2
    assert entries[0].content == "My name is Ada."


asyncio.run(main())
