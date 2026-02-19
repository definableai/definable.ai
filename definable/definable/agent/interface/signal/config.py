"""Signal-specific configuration."""

from dataclasses import dataclass, field
from typing import List, Optional

from definable.agent.interface.config import InterfaceConfig
from definable.agent.interface.errors import InterfaceError


@dataclass(frozen=True)
class SignalConfig(InterfaceConfig):
  """Configuration for the Signal interface.

  Extends InterfaceConfig with Signal-specific settings.
  Uses signal-cli-rest-api as the backend (Docker container).

  Attributes:
    api_base_url: Base URL for the signal-cli-rest-api endpoint.
    phone_number: Registered Signal phone number (required).
    allowed_phone_numbers: Restrict to specific sender phone numbers.
    allowed_group_ids: Restrict to specific Signal group IDs.
    polling_interval: Seconds between polling for messages.
    connect_timeout: HTTP connection timeout in seconds.
    request_timeout: HTTP request timeout in seconds.
    trust_all_keys: Auto-trust new identity keys.
    manage_container: Auto-start/stop the signal-cli-rest-api Docker container.
    docker_image: Docker image to use for the container.
    docker_container_name: Name for the Docker container.
    docker_host_port: Host port to map to container port 8080.
    docker_data_dir: Host path for persistent signal-cli data. Empty = ephemeral.
    docker_startup_timeout: Max seconds to wait for container to become healthy.
    docker_mode: MODE env var for signal-cli-rest-api (e.g. "native", "json-rpc").
  """

  platform: str = "signal"
  api_base_url: str = "http://localhost:8080"
  phone_number: str = ""

  # Access control
  allowed_phone_numbers: Optional[List[str]] = field(default=None, hash=False)
  allowed_group_ids: Optional[List[str]] = field(default=None, hash=False)

  # Polling settings
  polling_interval: float = 1.0

  # HTTP settings
  connect_timeout: float = 10.0
  request_timeout: float = 60.0

  max_message_length: int = 65536

  # Signal-specific
  trust_all_keys: bool = True

  # Docker auto-management (opt-in)
  manage_container: bool = False
  docker_image: str = "bbernhard/signal-cli-rest-api:latest"
  docker_container_name: str = "definable-signal-api"
  docker_host_port: int = 8080
  docker_data_dir: str = ""
  docker_startup_timeout: float = 30.0
  docker_mode: str = "native"

  def __post_init__(self) -> None:
    if not self.phone_number:
      raise InterfaceError("phone_number is required for SignalConfig", platform="signal")

    if self.manage_container:
      if self.docker_host_port <= 0:
        raise InterfaceError("docker_host_port must be > 0", platform="signal")
      # Auto-derive api_base_url from docker_host_port if still at default
      if self.api_base_url == "http://localhost:8080":
        object.__setattr__(self, "api_base_url", f"http://localhost:{self.docker_host_port}")
