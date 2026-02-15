"""E2E tests — Agent + Readers Integration.

Scenario: "I want my agent to analyze uploaded files."

All tests require OPENAI_API_KEY.
"""

import pytest

from definable.agents.agent import Agent
from definable.agents.config import AgentConfig, TracingConfig
from definable.agents.testing import MockModel
from definable.media import File
from definable.readers.base import BaseReader
from definable.readers.models import ReaderConfig


# ---------------------------------------------------------------------------
# Agent + Readers (mock model, no API key)
# ---------------------------------------------------------------------------


class TestAgentReadsFilesMocked:
  """Agent file reading with MockModel to verify message injection."""

  @pytest.mark.asyncio
  async def test_agent_reads_text_file(self):
    """Agent with readers=True injects file content into messages."""
    model = MockModel(responses=["I see the file content."])
    agent = Agent(
      model=model,
      readers=True,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )

    output = await agent.arun(
      "What does this file say?",
      files=[File(content=b"The answer is 42.", filename="answer.txt", mime_type="text/plain")],
    )

    assert output.content == "I see the file content."
    call = model.call_history[0]
    user_msgs = [m for m in call["messages"] if m.role == "user"]
    last_user = user_msgs[-1]
    assert "The answer is 42." in last_user.content
    assert "<file_contents>" in last_user.content

  @pytest.mark.asyncio
  async def test_agent_reads_multiple_files(self):
    """Agent injects content from multiple files."""
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
  async def test_readers_false_no_processing(self):
    """Agent without readers does not inject <file_contents>."""
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
  async def test_readers_true_creates_default_reader(self):
    """Agent(readers=True) creates a BaseReader instance."""
    model = MockModel(responses=["OK."])
    agent = Agent(
      model=model,
      readers=True,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    assert agent.readers is not None
    assert isinstance(agent.readers, BaseReader)

  @pytest.mark.asyncio
  async def test_agent_with_custom_reader(self):
    """Agent(readers=BaseReader(config=...)) uses custom config."""
    config = ReaderConfig(max_content_length=20)
    reader = BaseReader(config=config)

    model = MockModel(responses=["Custom reader works."])
    agent = Agent(
      model=model,
      readers=reader,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )

    await agent.arun(
      "Read this file.",
      files=[File(content=b"A" * 200, filename="long.txt", mime_type="text/plain")],
    )

    call = model.call_history[0]
    user_msgs = [m for m in call["messages"] if m.role == "user"]
    file_content_in_msg = user_msgs[-1].content
    # Content should be truncated to 20 chars
    assert "A" * 20 in file_content_in_msg
    assert "A" * 200 not in file_content_in_msg

  @pytest.mark.asyncio
  async def test_image_through_files_produces_placeholder(self, sample_png_bytes):
    """Image file through readers produces placeholder text, not vision analysis.

    NOTE: This documents a known limitation. To get actual image analysis,
    pass images via the `images=` parameter instead of `files=`.
    """
    model = MockModel(responses=["I see the image."])
    agent = Agent(
      model=model,
      readers=True,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )

    await agent.arun(
      "What is in this image?",
      files=[File(content=sample_png_bytes, filename="photo.png", mime_type="image/png")],
    )

    call = model.call_history[0]
    user_msgs = [m for m in call["messages"] if m.role == "user"]
    content = user_msgs[-1].content
    # Image through files= gets placeholder text, not actual analysis
    assert "[image: image/png]" in content


# ---------------------------------------------------------------------------
# Agent + Readers (real OpenAI model)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.openai
class TestAgentReadsFiles:
  """Agent analyzes files with a real model."""

  @pytest.mark.asyncio
  async def test_agent_analyzes_csv_file(self, openai_model, sample_csv_file):
    """Agent describes CSV data."""
    agent = Agent(
      model=openai_model,
      readers=True,
      instructions="Analyze the uploaded file and describe its contents briefly.",
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    output = await agent.arun("What data is in this file?", files=[sample_csv_file])

    assert output.content is not None
    content_lower = output.content.lower()
    assert any(word in content_lower for word in ["product", "widget", "revenue", "units", "sales"])

  @pytest.mark.asyncio
  async def test_agent_analyzes_json_file(self, openai_model, sample_json_file):
    """Agent describes JSON config."""
    agent = Agent(
      model=openai_model,
      readers=True,
      instructions="Describe the configuration in the uploaded JSON file.",
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    output = await agent.arun("What's in this config?", files=[sample_json_file])

    assert output.content is not None
    content_lower = output.content.lower()
    assert any(word in content_lower for word in ["app", "version", "debug", "test", "config"])

  @pytest.mark.asyncio
  async def test_agent_reads_multiple_files(self, openai_model, sample_csv_file, sample_json_file):
    """Agent reads and references content from multiple files."""
    agent = Agent(
      model=openai_model,
      readers=True,
      instructions="Describe all uploaded files briefly.",
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    output = await agent.arun(
      "What files do I have and what do they contain?",
      files=[sample_csv_file, sample_json_file],
    )

    assert output.content is not None
    # Should mention content from both files
    content_lower = output.content.lower()
    assert any(word in content_lower for word in ["csv", "sales", "widget", "product"])
    assert any(word in content_lower for word in ["json", "config", "app", "version"])
