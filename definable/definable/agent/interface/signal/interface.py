"""Signal interface implementation using signal-cli-rest-api."""

import asyncio
import contextlib
import shutil
from typing import Any, Dict, List, Optional

import httpx

from definable.agent.interface.base import BaseInterface
from definable.agent.interface.errors import (
  InterfaceAuthenticationError,
  InterfaceConnectionError,
  InterfaceMessageError,
  InterfaceRateLimitError,
)
from definable.agent.interface.message import InterfaceMessage, InterfaceResponse
from definable.agent.interface.signal.config import SignalConfig
from definable.media import Audio, File, Image
from definable.utils.log import log_debug, log_error, log_info, log_warning


class SignalInterface(BaseInterface):
  """Interface connecting an agent to Signal via signal-cli-rest-api.

  Uses httpx to communicate with a signal-cli-rest-api Docker container.
  Messages are received by polling the REST API.

  Args:
    agent: The Agent instance.
    config: SignalConfig with phone number and API settings.
    session_manager: Optional session manager.
    hooks: Optional list of hooks.

  Example:
    interface = SignalInterface(
      agent=agent,
      config=SignalConfig(
        phone_number="+1234567890",
        api_base_url="http://localhost:8080",
      ),
    )
    async with interface:
      await interface.serve_forever()
  """

  def __init__(self, **kwargs: Any) -> None:
    super().__init__(**kwargs)
    self._sg_config: SignalConfig = self.config  # type: ignore[assignment]
    self._client: Optional[httpx.AsyncClient] = None
    self._poll_task: Optional[asyncio.Task[None]] = None
    self._container_managed: bool = False

  # --- Lifecycle ---

  async def _start_receiver(self) -> None:
    if self._sg_config.manage_container:
      self._ensure_docker_available()
      await self._start_container()

    self._client = httpx.AsyncClient(
      timeout=httpx.Timeout(
        connect=self._sg_config.connect_timeout,
        read=self._sg_config.request_timeout,
        write=self._sg_config.request_timeout,
        pool=self._sg_config.connect_timeout,
      ),
    )

    # Verify connection to signal-cli-rest-api
    await self._verify_connection()

    # Trust all keys if configured
    if self._sg_config.trust_all_keys:
      with contextlib.suppress(Exception):
        await self._api_call(
          "PUT",
          f"/v1/configuration/{self._sg_config.phone_number}/settings",
          json={"trust_new_identities": "always"},
        )

    # Start polling loop
    self._poll_task = asyncio.create_task(self._poll_loop())
    log_info("[signal] Polling started")

  async def _stop_receiver(self) -> None:
    if self._poll_task is not None:
      self._poll_task.cancel()
      with contextlib.suppress(asyncio.CancelledError):
        await self._poll_task
      self._poll_task = None

    if self._client is not None:
      await self._client.aclose()
      self._client = None

    if self._container_managed:
      await self._stop_container()

  # --- Docker container management ---

  def _ensure_docker_available(self) -> None:
    """Check that the docker CLI is available on the system."""
    if shutil.which("docker") is None:
      raise InterfaceConnectionError(
        "Docker CLI not found. Install Docker to use manage_container=True, "
        "or set manage_container=False and run the signal-cli-rest-api container manually.",
        platform="signal",
      )

  async def _start_container(self) -> None:
    """Start the signal-cli-rest-api Docker container if not already running."""
    name = self._sg_config.docker_container_name

    # Check if a container with this name already exists
    inspect_proc = await asyncio.create_subprocess_exec(
      "docker",
      "inspect",
      "--format",
      "{{.State.Running}}",
      name,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await inspect_proc.communicate()

    if inspect_proc.returncode == 0:
      state = stdout.decode().strip()
      if state == "true":
        log_info(f"[signal] Reusing existing container '{name}'")
        self._container_managed = False
        return
      # Container exists but is stopped — remove it before starting fresh
      log_info(f"[signal] Removing stopped container '{name}'")
      rm_proc = await asyncio.create_subprocess_exec(
        "docker",
        "rm",
        name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
      )
      await rm_proc.communicate()

    # Build docker run command
    cmd: List[str] = [
      "docker",
      "run",
      "-d",
      "--name",
      name,
      "-p",
      f"{self._sg_config.docker_host_port}:8080",
      "-e",
      f"MODE={self._sg_config.docker_mode}",
    ]

    if self._sg_config.docker_data_dir:
      cmd.extend(["-v", f"{self._sg_config.docker_data_dir}:/home/.local/share/signal-cli"])

    cmd.append(self._sg_config.docker_image)

    log_info(f"[signal] Starting container '{name}' on port {self._sg_config.docker_host_port}")

    try:
      run_proc = await asyncio.wait_for(
        asyncio.create_subprocess_exec(
          *cmd,
          stdout=asyncio.subprocess.PIPE,
          stderr=asyncio.subprocess.PIPE,
        ),
        timeout=30.0,
      )
      stdout, stderr = await asyncio.wait_for(run_proc.communicate(), timeout=30.0)
    except TimeoutError as e:
      raise InterfaceConnectionError(
        f"Timed out starting Docker container '{name}'",
        platform="signal",
      ) from e

    if run_proc.returncode != 0:
      err_msg = stderr.decode().strip() if stderr else "unknown error"
      raise InterfaceConnectionError(
        f"Failed to start Docker container '{name}': {err_msg}",
        platform="signal",
      )

    container_id = stdout.decode().strip()[:12]
    log_info(f"[signal] Container started (id={container_id})")
    self._container_managed = True

    # Wait for the API to become healthy
    await self._wait_for_container_health()

  async def _wait_for_container_health(self) -> None:
    """Poll the container's /v1/about endpoint until it responds."""
    url = f"{self._sg_config.api_base_url}/v1/about"
    timeout = self._sg_config.docker_startup_timeout
    interval = 1.0
    elapsed = 0.0

    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
      while elapsed < timeout:
        try:
          resp = await client.get(url)
          if resp.status_code < 500:
            log_info("[signal] Container is healthy")
            return
        except (httpx.ConnectError, httpx.TimeoutException):
          pass
        await asyncio.sleep(interval)
        elapsed += interval

    raise InterfaceConnectionError(
      f"Container did not become healthy within {timeout}s. Check docker logs {self._sg_config.docker_container_name} for details.",
      platform="signal",
    )

  async def _stop_container(self) -> None:
    """Stop and remove the Docker container we started."""
    name = self._sg_config.docker_container_name
    log_info(f"[signal] Stopping container '{name}'")

    try:
      stop_proc = await asyncio.create_subprocess_exec(
        "docker",
        "stop",
        "-t",
        "10",
        name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
      )
      await asyncio.wait_for(stop_proc.communicate(), timeout=15.0)

      rm_proc = await asyncio.create_subprocess_exec(
        "docker",
        "rm",
        name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
      )
      await asyncio.wait_for(rm_proc.communicate(), timeout=10.0)

      log_info(f"[signal] Container '{name}' stopped and removed")
    except Exception as e:
      log_warning(f"[signal] Failed to stop/remove container '{name}': {e}")

  # --- Connection verification ---

  async def _verify_connection(self) -> None:
    """Verify connection to the signal-cli-rest-api."""
    try:
      result = await self._api_call("GET", "/v1/about")
      versions = result.get("versions", [])
      log_info(f"[signal] Connected to signal-cli-rest-api (versions: {versions})")
    except InterfaceConnectionError:
      raise
    except Exception as e:
      raise InterfaceConnectionError(
        f"Failed to connect to signal-cli-rest-api at {self._sg_config.api_base_url}: {e}",
        platform="signal",
      ) from e

  # --- Polling ---

  async def _poll_loop(self) -> None:
    """Polling loop that fetches messages from signal-cli-rest-api."""
    while self._running:
      try:
        messages = await self._receive_messages()
        for msg in messages:
          asyncio.create_task(self._process_envelope(msg))
      except asyncio.CancelledError:
        break
      except httpx.TimeoutException:
        continue
      except Exception as e:
        log_error(f"[signal] Polling error: {e}")

      await asyncio.sleep(self._sg_config.polling_interval)

  async def _receive_messages(self) -> List[Dict[str, Any]]:
    """Fetch pending messages via the REST API."""
    result = await self._api_call(
      "GET",
      f"/v1/receive/{self._sg_config.phone_number}",
    )
    if isinstance(result, list):
      return result
    return []

  async def _process_envelope(self, envelope: Dict[str, Any]) -> None:
    """Process a single Signal message envelope."""
    # Only process data messages (not receipts, typing indicators, etc.)
    data_message = envelope.get("envelope", {}).get("dataMessage")
    if data_message is None:
      return
    await self.handle_platform_message(envelope)

  # --- Inbound conversion ---

  async def _convert_inbound(self, raw_message: Dict[str, Any]) -> Optional[InterfaceMessage]:
    """Convert a signal-cli-rest-api envelope to InterfaceMessage."""
    envelope = raw_message.get("envelope", {})
    data_message = envelope.get("dataMessage", {})

    source = envelope.get("sourceNumber") or envelope.get("source", "")
    source_name = envelope.get("sourceName", "")
    timestamp = str(envelope.get("timestamp", ""))

    # Determine chat context (group or direct)
    group_info = data_message.get("groupInfo", {})
    group_id = group_info.get("groupId")
    chat_id = group_id or source

    # Access control — phone numbers
    if self._sg_config.allowed_phone_numbers is not None:
      if source not in self._sg_config.allowed_phone_numbers:
        log_debug(f"[signal] Ignoring message from unauthorized number {source}")
        return None

    # Access control — groups
    if self._sg_config.allowed_group_ids is not None and group_id:
      if group_id not in self._sg_config.allowed_group_ids:
        log_debug(f"[signal] Ignoring message from unauthorized group {group_id}")
        return None

    text = data_message.get("message")

    # Extract attachments
    images: Optional[List[Image]] = None
    audio_list: Optional[List[Audio]] = None
    files: Optional[List[File]] = None

    for attachment in data_message.get("attachments", []):
      content_type = attachment.get("contentType", "")
      attachment_id = attachment.get("id", "")

      if not attachment_id:
        continue

      # Build the download URL for this attachment
      attachment_url = f"{self._sg_config.api_base_url}/v1/attachments/{attachment_id}"

      if content_type.startswith("image/"):
        if images is None:
          images = []
        images.append(Image(url=attachment_url))
      elif content_type.startswith("audio/"):
        if audio_list is None:
          audio_list = []
        audio_list.append(Audio(url=attachment_url, mime_type=content_type))
      else:
        if files is None:
          files = []
        files.append(
          File(
            url=attachment_url,
            mime_type=content_type or None,
            filename=attachment.get("filename"),
            size=attachment.get("size"),
          )
        )

    # Reply context
    reply_to_message_id: Optional[str] = None
    quote = data_message.get("quote")
    if quote:
      reply_to_message_id = str(quote.get("id", ""))

    return InterfaceMessage(
      text=text,
      platform="signal",
      platform_user_id=source,
      platform_chat_id=chat_id,
      platform_message_id=timestamp,
      username=source_name or source,
      images=images,
      audio=audio_list,
      files=files,
      reply_to_message_id=reply_to_message_id,
      metadata={"raw": raw_message},
    )

  # --- Response sending ---

  async def _send_response(
    self,
    original_msg: InterfaceMessage,
    response: InterfaceResponse,
    raw_message: Any,
  ) -> None:
    """Send response back via Signal."""
    envelope = raw_message.get("envelope", {})
    data_message = envelope.get("dataMessage", {})
    source = envelope.get("sourceNumber") or envelope.get("source", "")
    group_info = data_message.get("groupInfo", {})
    group_id = group_info.get("groupId")

    # Send text content (split if needed)
    if response.content:
      max_len = self._sg_config.max_message_length
      chunks = self._split_text(response.content, max_len)

      for chunk in chunks:
        await self._send_message(chunk, source, group_id)

    # Send images
    if response.images:
      for image in response.images:
        await self._send_attachment(source, group_id, image=image)

    # Send files
    if response.files:
      for file in response.files:
        await self._send_attachment(source, group_id, file=file)

  async def _send_message(
    self,
    text: str,
    recipient: str,
    group_id: Optional[str] = None,
  ) -> None:
    """Send a text message via signal-cli-rest-api."""
    payload: Dict[str, Any] = {
      "message": text,
      "number": self._sg_config.phone_number,
      "text_mode": "normal",
    }

    if group_id:
      payload["recipients"] = [group_id]
    else:
      payload["recipients"] = [recipient]

    await self._api_call("POST", "/v2/send", json=payload)

  async def _send_attachment(
    self,
    recipient: str,
    group_id: Optional[str] = None,
    image: Optional[Image] = None,
    file: Optional[File] = None,
  ) -> None:
    """Send a message with an attachment via signal-cli-rest-api."""
    import base64

    attachment_data: Optional[str] = None
    filename = "attachment"

    if image:
      content_bytes = image.get_content_bytes()
      if content_bytes:
        attachment_data = base64.b64encode(content_bytes).decode("utf-8")
        filename = f"image.{image.format or 'png'}"
    elif file:
      if file.content:
        raw = file.content if isinstance(file.content, bytes) else str(file.content).encode()
        attachment_data = base64.b64encode(raw).decode("utf-8")
      elif file.filepath:
        with open(str(file.filepath), "rb") as f:
          attachment_data = base64.b64encode(f.read()).decode("utf-8")
      filename = file.filename or "file"

    if attachment_data is None:
      # Can't send attachment without data; send URL as text instead
      url = (image.url if image else None) or (file.url if file else None)
      if url:
        await self._send_message(url, recipient, group_id)
      return

    payload: Dict[str, Any] = {
      "message": "",
      "number": self._sg_config.phone_number,
      "base64_attachments": [f"data:application/octet-stream;filename={filename};base64,{attachment_data}"],
    }

    if group_id:
      payload["recipients"] = [group_id]
    else:
      payload["recipients"] = [recipient]

    await self._api_call("POST", "/v2/send", json=payload)

  # --- REST API ---

  async def _api_call(
    self,
    method: str,
    path: str,
    json: Optional[Dict[str, Any]] = None,
  ) -> Any:
    """Make a call to the signal-cli-rest-api.

    Args:
      method: HTTP method (GET, POST, PUT, etc.).
      path: API path (e.g. "/v1/receive/+1234567890").
      json: Optional JSON payload.

    Returns:
      Parsed JSON response.

    Raises:
      InterfaceAuthenticationError: On 401/403 responses.
      InterfaceRateLimitError: On 429 responses.
      InterfaceMessageError: On 400 responses.
      InterfaceConnectionError: On connection failures.
    """
    assert self._client is not None
    url = f"{self._sg_config.api_base_url}{path}"

    try:
      response = await self._client.request(method, url, json=json)
    except httpx.ConnectError as e:
      raise InterfaceConnectionError(
        f"Failed to connect to signal-cli-rest-api: {e}",
        platform="signal",
      ) from e
    except httpx.TimeoutException as e:
      raise InterfaceConnectionError(
        f"signal-cli-rest-api request timed out: {e}",
        platform="signal",
      ) from e

    if response.status_code >= 400:
      text = response.text
      if response.status_code in (401, 403):
        raise InterfaceAuthenticationError(
          f"Authentication failed: {text}",
          platform="signal",
        )
      if response.status_code == 429:
        raise InterfaceRateLimitError(
          f"Rate limited: {text}",
          platform="signal",
        )
      if response.status_code == 400:
        raise InterfaceMessageError(
          f"Bad request: {text}",
          platform="signal",
        )
      raise InterfaceConnectionError(
        f"signal-cli-rest-api error ({response.status_code}): {text}",
        platform="signal",
      )

    if response.status_code == 204:
      return {}

    try:
      return response.json()
    except Exception:
      return {}

  # --- Utilities ---

  @staticmethod
  def _split_text(text: str, max_length: int) -> List[str]:
    """Split text into chunks respecting max_length.

    Tries to split at newlines, then at spaces, falling back to
    hard splits if necessary.
    """
    if len(text) <= max_length:
      return [text]

    chunks: List[str] = []
    remaining = text
    while remaining:
      if len(remaining) <= max_length:
        chunks.append(remaining)
        break

      # Try to split at a newline
      split_pos = remaining.rfind("\n", 0, max_length)
      if split_pos == -1:
        # Try to split at a space
        split_pos = remaining.rfind(" ", 0, max_length)
      if split_pos == -1:
        # Hard split
        split_pos = max_length

      chunks.append(remaining[:split_pos])
      remaining = remaining[split_pos:].lstrip("\n")

    return chunks
