from dataclasses import dataclass, field
from os import getenv
from typing import Any, Dict, Optional

from definable.exceptions import ModelAuthenticationError
from definable.model.openai import OpenAILike


@dataclass
class MoonshotChat(OpenAILike):
  """
  A class for interacting with Moonshot models using the Chat completions API.
  """

  provider: str = "Moonshot"
  id: str = "kimi-k2.5"
  # DeepSeek/Moonshot echo assistant reasoning_content back on follow-up turns.
  send_reasoning_content: bool = True
  supports_native_structured_outputs: bool = False
  api_key: Optional[str] = field(default_factory=lambda: getenv("MOONSHOT_API_KEY"))
  base_url: Optional[str] = "https://api.moonshot.ai/v1"

  def _get_client_params(self) -> Dict[str, Any]:
    # Fetch API key from env if not already set
    if not self.api_key:
      self.api_key = getenv("MOONSHOT_API_KEY")
      if not self.api_key:
        # Raise error immediately if key is missing
        raise ModelAuthenticationError(
          message="MOONSHOT_API_KEY not set. Please set the MOONSHOT_API_KEY environment variable.",
          model_name=self.name,
        )

    # Define base client params
    base_params = {
      "api_key": self.api_key,
      "organization": self.organization,
      "base_url": self.base_url,
      "timeout": self.timeout,
      "max_retries": self.max_retries,
      "default_headers": self.default_headers,
      "default_query": self.default_query,
    }

    # Create client_params dict with non-None values
    client_params = {k: v for k, v in base_params.items() if v is not None}

    # Add additional client params if provided
    if self.client_params:
      client_params.update(self.client_params)
    return client_params
