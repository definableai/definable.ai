"""Shared fixtures for interface tests."""

from typing import List, Optional

import pytest

from definable.agents.agent import Agent
from definable.agents.testing import MockModel
from definable.interfaces.message import InterfaceMessage
from definable.run.agent import RunOutput


@pytest.fixture
def mock_model():
  """Basic MockModel with a simple response."""
  return MockModel(responses=["Test response"])


@pytest.fixture
def mock_agent(mock_model):
  """Agent using a MockModel for unit testing."""
  return Agent(model=mock_model, instructions="You are a test assistant.")


@pytest.fixture
def mock_agent_factory():
  """Factory for creating agents with custom mock responses."""

  def _create(responses: Optional[List[str]] = None, **kwargs):
    model = MockModel(responses=responses or ["Mock response"])
    return Agent(model=model, instructions="You are a test assistant.", **kwargs)

  return _create


@pytest.fixture
def sample_interface_message():
  """Factory fixture for creating InterfaceMessage instances."""

  def _create(
    platform: str = "test",
    text: str = "Hello",
    user_id: str = "user123",
    chat_id: str = "chat456",
    message_id: str = "msg789",
  ) -> InterfaceMessage:
    return InterfaceMessage(
      text=text,
      platform=platform,
      platform_user_id=user_id,
      platform_chat_id=chat_id,
      platform_message_id=message_id,
    )

  return _create


@pytest.fixture
def sample_run_output():
  """Factory fixture for creating RunOutput instances."""

  def _create(content: str = "Test response") -> RunOutput:
    output = RunOutput()
    output.content = content
    output.messages = []
    output.images = None
    output.videos = None
    output.audio = None
    output.files = None
    return output

  return _create
