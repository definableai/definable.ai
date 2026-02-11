"""Integration tests — Agent + readers end-to-end."""

import pytest

from definable.agents.agent import Agent
from definable.agents.config import AgentConfig, TracingConfig
from definable.agents.testing import MockModel
from definable.media import File
from definable.readers.base import BaseReader
from definable.readers.parsers.text import TextParser
from definable.readers.registry import ParserRegistry
from definable.run.agent import FileReadCompletedEvent, FileReadStartedEvent


class TestAgentWithReaders:
  @pytest.mark.asyncio
  async def test_agent_reads_text_file(self):
    model = MockModel(responses=["I see the file content."])
    agent = Agent(
      model=model,
      readers=True,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )

    output = await agent.arun(
      "What does this file say?",
      files=[
        File(
          content=b"The answer is 42.",
          filename="answer.txt",
          mime_type="text/plain",
        )
      ],
    )

    assert output.content == "I see the file content."
    # Verify file content was injected into the messages sent to the model
    call = model.call_history[0]
    messages = call["messages"]
    user_msgs = [m for m in messages if m.role == "user"]
    assert len(user_msgs) >= 1
    last_user = user_msgs[-1]
    assert "The answer is 42." in last_user.content
    assert "<file_contents>" in last_user.content

  @pytest.mark.asyncio
  async def test_agent_reads_multiple_files(self):
    model = MockModel(responses=["Got both files."])
    agent = Agent(
      model=model,
      readers=True,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )

    await agent.arun(
      "Summarize these files.",
      files=[
        File(content=b"File one content.", filename="one.txt", mime_type="text/plain"),
        File(content=b"File two content.", filename="two.txt", mime_type="text/plain"),
      ],
    )

    call = model.call_history[0]
    user_msgs = [m for m in call["messages"] if m.role == "user"]
    content = user_msgs[-1].content
    assert "File one content." in content
    assert "File two content." in content

  @pytest.mark.asyncio
  async def test_agent_without_readers_passes_files_through(self):
    """When readers=None, files should not be processed."""
    model = MockModel(responses=["No readers."])
    agent = Agent(
      model=model,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )

    await agent.arun(
      "Hello",
      files=[File(content=b"data", filename="test.txt", mime_type="text/plain")],
    )

    call = model.call_history[0]
    user_msgs = [m for m in call["messages"] if m.role == "user"]
    assert "<file_contents>" not in (user_msgs[-1].content or "")

  @pytest.mark.asyncio
  async def test_agent_no_files_no_extraction(self):
    """When no files are passed, readers should be a no-op."""
    model = MockModel(responses=["No files."])
    agent = Agent(
      model=model,
      readers=True,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )

    await agent.arun("Hello, no files here.")
    call = model.call_history[0]
    user_msgs = [m for m in call["messages"] if m.role == "user"]
    assert "<file_contents>" not in (user_msgs[-1].content or "")


class TestStreamingWithReaders:
  @pytest.mark.asyncio
  async def test_stream_emits_file_read_events(self):
    model = MockModel(responses=["Streamed response."])
    agent = Agent(
      model=model,
      readers=True,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )

    events = []
    async for event in agent.arun_stream(
      "What is in this file?",
      images=None,
    ):
      events.append(event)

    # No files → no file read events
    file_events = [e for e in events if isinstance(e, (FileReadStartedEvent, FileReadCompletedEvent))]
    assert len(file_events) == 0


class TestCustomReader:
  @pytest.mark.asyncio
  async def test_custom_reader_used(self):
    registry = ParserRegistry(include_defaults=False)
    registry.register(TextParser())
    reader = BaseReader(registry=registry)

    model = MockModel(responses=["Custom reader works."])
    agent = Agent(
      model=model,
      readers=reader,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )

    await agent.arun(
      "Read this file.",
      files=[File(content=b"Custom reader content.", filename="custom.txt", mime_type="text/plain")],
    )

    call = model.call_history[0]
    user_msgs = [m for m in call["messages"] if m.role == "user"]
    assert "Custom reader content." in user_msgs[-1].content
