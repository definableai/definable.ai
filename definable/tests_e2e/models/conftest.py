"""Model-specific fixtures for E2E tests."""

from os import getenv

import pytest

from definable.models.anthropic import Claude
from definable.models.deepseek import DeepSeekChat
from definable.models.moonshot import MoonshotChat
from definable.models.openai import OpenAIChat
from definable.models.xai import xAI


@pytest.fixture
def anthropic_model():
    """Return an Anthropic Claude model instance, skip if no API key."""
    api_key = getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY environment variable not set")
    return Claude(api_key=api_key, id="claude-3-5-haiku-latest")


@pytest.fixture
def openai_model():
    """Return an OpenAI model instance, skip if no API key."""
    api_key = getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    return OpenAIChat(api_key=api_key, id="gpt-4o-mini")


@pytest.fixture
def deepseek_model():
    """Return a DeepSeek model instance, skip if no API key."""
    api_key = getenv("DEEPSEEK_API_KEY")
    if not api_key:
        pytest.skip("DEEPSEEK_API_KEY environment variable not set")
    return DeepSeekChat(api_key=api_key)


@pytest.fixture
def moonshot_model():
    """Return a Moonshot model instance, skip if no API key."""
    api_key = getenv("MOONSHOT_API_KEY")
    if not api_key:
        pytest.skip("MOONSHOT_API_KEY environment variable not set")
    return MoonshotChat(api_key=api_key)


@pytest.fixture
def xai_model():
    """Return an xAI model instance, skip if no API key."""
    api_key = getenv("XAI_API_KEY")
    if not api_key:
        pytest.skip("XAI_API_KEY environment variable not set")
    return xAI(api_key=api_key)
