"""HTTP client for the Definable Desktop Bridge app.

Provides typed async methods for every bridge endpoint. Token is auto-read
from ``~/.definable/bridge-token`` when not explicitly provided.

Example::

    async with BridgeClient() as client:
        png_bytes = await client.capture_screen()
        await client.click(x=500, y=400)
"""

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx


# ---------------------------------------------------------------------------
# Return-type dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ElementBounds:
  """Screen bounding box for a UI element."""

  x: float
  y: float
  width: float
  height: float

  @property
  def center(self) -> Tuple[float, float]:
    """Return the center (cx, cy) of the element."""
    return (self.x + self.width / 2, self.y + self.height / 2)


@dataclass
class UIElement:
  """Accessibility UI element."""

  role: str
  title: str
  value: str
  bounds: ElementBounds
  children: Optional[List["UIElement"]] = None


@dataclass
class WindowInfo:
  """Info about an open window."""

  id: int
  app: str
  title: str
  bounds: ElementBounds
  minimized: bool


@dataclass
class AppInfo:
  """Info about a running application."""

  name: str
  bundle_id: str
  pid: int
  active: bool


@dataclass
class BridgePermissions:
  """macOS privacy permissions status."""

  accessibility: bool
  screen_recording: bool
  full_disk_access: bool


# ---------------------------------------------------------------------------
# Default token file location
# ---------------------------------------------------------------------------

_DEFAULT_TOKEN_PATH = Path.home() / ".definable" / "bridge-token"


def _read_token_file(path: Path) -> Optional[str]:
  """Read token from file, returning None if not found."""
  try:
    return path.read_text().strip() or None
  except (OSError, IOError):
    return None


# ---------------------------------------------------------------------------
# BridgeClient
# ---------------------------------------------------------------------------


class BridgeClient:
  """Async HTTP client for the Definable Desktop Bridge.

  All public methods are coroutines and map 1:1 to bridge endpoints.
  The underlying ``httpx.AsyncClient`` is lazily created on first use.

  Args:
    host: Bridge host (default: ``"127.0.0.1"``).
    port: Bridge port (default: ``7777``).
    token: Bearer token. If ``None``, reads from ``~/.definable/bridge-token``.
    timeout: Request timeout in seconds (default: ``30``).

  Example::

    client = BridgeClient()
    png_bytes = await client.capture_screen()
    await client.click(x=100, y=200)
    await client.close()

  Or as a context manager::

    async with BridgeClient() as client:
        text = await client.ocr_screen()
  """

  def __init__(
    self,
    *,
    host: str = "127.0.0.1",
    port: int = 7777,
    token: Optional[str] = None,
    timeout: float = 30.0,
  ) -> None:
    self._base_url = f"http://{host}:{port}"
    self._token = token
    self._timeout = timeout
    self._client: Optional[httpx.AsyncClient] = None

  # --- Lifecycle ---

  def _resolve_token(self) -> Optional[str]:
    """Resolve bearer token: explicit > file > None."""
    if self._token is not None:
      return self._token
    return _read_token_file(_DEFAULT_TOKEN_PATH)

  def _get_client(self) -> httpx.AsyncClient:
    """Lazily create and return the httpx client."""
    if self._client is None:
      token = self._resolve_token()
      headers = {}
      if token:
        headers["Authorization"] = f"Bearer {token}"
      self._client = httpx.AsyncClient(
        base_url=self._base_url,
        headers=headers,
        timeout=self._timeout,
      )
    return self._client

  async def close(self) -> None:
    """Close the underlying HTTP client."""
    if self._client is not None:
      await self._client.aclose()
      self._client = None

  async def __aenter__(self) -> "BridgeClient":
    return self

  async def __aexit__(self, *args: Any) -> None:
    await self.close()

  # --- Internal helper ---

  async def _post(self, path: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """POST to the bridge and return the JSON response body.

    Raises:
      httpx.ConnectError: Bridge is not reachable.
      httpx.HTTPStatusError: Bridge returned a non-2xx status.
    """
    client = self._get_client()
    response = await client.post(path, json=data or {})
    response.raise_for_status()
    return response.json()  # type: ignore[no-any-return]

  # --- Health ---

  async def health(self) -> Dict[str, Any]:
    """Check bridge health and permission status.

    Returns:
      Dict with ``status``, ``version``, and ``permissions`` keys.
    """
    return await self._post("/health")

  # --- Screen ---

  async def capture_screen(
    self,
    display: int = 0,
    region: Optional[Dict[str, int]] = None,
    max_width: int = 512,
  ) -> bytes:
    """Capture a screenshot.

    The bridge downscales to ``max_width`` and encodes as JPEG to keep the
    result within LLM context budgets. Full-retina PNG would be 5–15 MB.

    Args:
      display: Display index (0 = primary).
      region: Optional ``{x, y, width, height}`` crop region.
      max_width: Maximum output width in pixels (default: 1280). Aspect ratio is preserved.

    Returns:
      JPEG image as raw bytes.
    """
    result = await self._post("/screen/capture", {"display": display, "region": region, "max_width": max_width})
    return base64.b64decode(result["image"])

  async def ocr_screen(
    self,
    region: Optional[Dict[str, int]] = None,
  ) -> Dict[str, Any]:
    """Run OCR on the screen.

    Args:
      region: Optional ``{x, y, width, height}`` crop region.

    Returns:
      Dict with ``text`` (str) and ``elements`` (list of ``{text, bounds}``).
    """
    return await self._post("/screen/ocr", {"region": region})

  async def find_text_on_screen(self, text: str, nth: int = 0) -> Optional[ElementBounds]:
    """Find the nth occurrence of text on screen.

    Args:
      text: Text to find (case-insensitive).
      nth: Which occurrence to return (0 = first).

    Returns:
      :class:`ElementBounds` if found, ``None`` otherwise.
    """
    result = await self._post("/screen/find_text", {"text": text, "nth": nth})
    if not result.get("found"):
      return None
    b = result["bounds"]
    return ElementBounds(x=b["x"], y=b["y"], width=b["width"], height=b["height"])

  # --- Input ---

  async def click(
    self,
    x: float,
    y: float,
    button: str = "left",
    clicks: int = 1,
    modifiers: Optional[List[str]] = None,
  ) -> None:
    """Simulate a mouse click.

    Args:
      x: X coordinate.
      y: Y coordinate.
      button: ``"left"``, ``"right"``, or ``"middle"``.
      clicks: Number of clicks (2 = double-click).
      modifiers: Keys held during click (e.g. ``["cmd"]``, ``["shift"]``).
    """
    await self._post("/input/click", {"x": x, "y": y, "button": button, "clicks": clicks, "modifiers": modifiers or []})

  async def type_text(self, text: str) -> None:
    """Type text at the current cursor position.

    Args:
      text: Text to type.
    """
    await self._post("/input/type", {"text": text})

  async def press_key(self, key: str, modifiers: Optional[List[str]] = None) -> None:
    """Press a key combination.

    Args:
      key: Key name (e.g. ``"return"``, ``"escape"``, ``"a"``).
      modifiers: Modifier keys (e.g. ``["cmd", "shift"]``).
    """
    await self._post("/input/key", {"key": key, "modifiers": modifiers or []})

  async def mouse_move(self, x: float, y: float) -> None:
    """Move the mouse cursor.

    Args:
      x: X coordinate.
      y: Y coordinate.
    """
    await self._post("/input/mouse_move", {"x": x, "y": y})

  async def scroll(self, x: float, y: float, dx: float = 0, dy: float = -3) -> None:
    """Scroll at the given position.

    Args:
      x: X coordinate to scroll at.
      y: Y coordinate to scroll at.
      dx: Horizontal scroll delta.
      dy: Vertical scroll delta (negative = down).
    """
    await self._post("/input/scroll", {"x": x, "y": y, "dx": dx, "dy": dy})

  async def drag(self, from_x: float, from_y: float, to_x: float, to_y: float, duration: float = 0.5) -> None:
    """Drag from one position to another.

    Args:
      from_x: Starting X coordinate.
      from_y: Starting Y coordinate.
      to_x: Ending X coordinate.
      to_y: Ending Y coordinate.
      duration: Drag duration in seconds.
    """
    await self._post("/input/drag", {"from_x": from_x, "from_y": from_y, "to_x": to_x, "to_y": to_y, "duration": duration})

  # --- Apps ---

  async def list_apps(self) -> List[AppInfo]:
    """List running applications.

    Returns:
      List of :class:`AppInfo` for each running app.
    """
    result = await self._post("/apps/list")
    return [
      AppInfo(
        name=a["name"],
        bundle_id=a.get("bundleId", ""),
        pid=a.get("pid", -1),
        active=a.get("active", False),
      )
      for a in result.get("apps", [])
    ]

  async def open_app(self, name: str) -> int:
    """Launch or activate an application.

    Args:
      name: App name, bundle ID, or full path.

    Returns:
      PID of the launched/activated process.
    """
    result = await self._post("/apps/open", {"name": name})
    return int(result.get("pid", -1))

  async def quit_app(self, name: str, force: bool = False) -> None:
    """Quit an application.

    Args:
      name: Application name.
      force: If ``True``, force-quit (SIGKILL). Otherwise graceful quit.
    """
    await self._post("/apps/quit", {"name": name, "force": force})

  async def activate_app(self, name: str) -> None:
    """Bring an application to the foreground.

    Args:
      name: Application name.
    """
    await self._post("/apps/activate", {"name": name})

  async def open_url(self, url: str) -> None:
    """Open a URL in the default browser.

    Args:
      url: URL to open.
    """
    await self._post("/apps/open_url", {"url": url})

  async def open_file(self, path: str) -> None:
    """Open a file with its default application.

    Args:
      path: Absolute path to the file.
    """
    await self._post("/apps/open_file", {"path": path})

  # --- Windows ---

  async def list_windows(self) -> List[WindowInfo]:
    """List open windows across all applications.

    Returns:
      List of :class:`WindowInfo` for each window.
    """
    result = await self._post("/windows/list")
    windows = []
    for w in result.get("windows", []):
      b = w.get("bounds", {})
      windows.append(
        WindowInfo(
          id=w.get("id", 0),
          app=w.get("app", ""),
          title=w.get("title", ""),
          bounds=ElementBounds(
            x=b.get("x", 0),
            y=b.get("y", 0),
            width=b.get("width", 0),
            height=b.get("height", 0),
          ),
          minimized=w.get("minimized", False),
        )
      )
    return windows

  async def focus_window(self, window_id: Optional[int] = None, title: Optional[str] = None) -> None:
    """Focus a window by ID or title.

    Args:
      window_id: Window ID (from :meth:`list_windows`).
      title: Window title substring to match.
    """
    await self._post("/windows/focus", {"id": window_id, "title": title})

  async def resize_window(self, window_id: int, x: float, y: float, width: float, height: float) -> None:
    """Resize and reposition a window.

    Args:
      window_id: Window ID.
      x: New X position.
      y: New Y position.
      width: New width.
      height: New height.
    """
    await self._post("/windows/resize", {"id": window_id, "x": x, "y": y, "width": width, "height": height})

  async def close_window(self, window_id: int) -> None:
    """Close a window.

    Args:
      window_id: Window ID.
    """
    await self._post("/windows/close", {"id": window_id})

  # --- Accessibility ---

  async def get_focused_element(self) -> Optional[UIElement]:
    """Get the currently focused UI element.

    Returns:
      :class:`UIElement` or ``None`` if nothing is focused.
    """
    result = await self._post("/ax/get_focused_element")
    if not result.get("found"):
      return None
    return _parse_ui_element(result["element"])

  async def get_ui_tree(self, app: str, depth: int = 3) -> Dict[str, Any]:
    """Get the accessibility UI tree for an application.

    Args:
      app: Application name.
      depth: Tree depth to traverse (default: 3).

    Returns:
      Nested dict representing the UI tree.
    """
    return await self._post("/ax/get_ui_tree", {"app": app, "depth": depth})

  async def find_ui_element(self, app: str, role: Optional[str] = None, title: Optional[str] = None) -> Optional[UIElement]:
    """Find a UI element in an application.

    Args:
      app: Application name.
      role: Accessibility role (e.g. ``"AXButton"``, ``"AXTextField"``).
      title: Element title or label.

    Returns:
      :class:`UIElement` or ``None`` if not found.
    """
    result = await self._post("/ax/find_element", {"app": app, "role": role, "title": title})
    if not result.get("found"):
      return None
    return _parse_ui_element(result["element"])

  async def click_ui_element(self, app: str, role: Optional[str] = None, title: Optional[str] = None) -> None:
    """Click a UI element found by accessibility attributes.

    Args:
      app: Application name.
      role: Accessibility role.
      title: Element title or label.
    """
    await self._post("/ax/perform_action", {"app": app, "role": role, "title": title, "action": "AXPress"})

  async def set_ui_value(self, app: str, role: str, title: str, value: str) -> None:
    """Set the value of a UI element (e.g. a text field).

    Args:
      app: Application name.
      role: Accessibility role.
      title: Element title or label.
      value: New value to set.
    """
    await self._post("/ax/set_value", {"app": app, "role": role, "title": title, "value": value})

  # --- AppleScript ---

  async def run_applescript(self, script: str) -> Dict[str, Any]:
    """Run an AppleScript.

    Args:
      script: AppleScript source code.

    Returns:
      Dict with ``output`` (str) and ``error`` (str or None).
    """
    return await self._post("/applescript/run", {"script": script})

  # --- Files ---

  async def list_files(self, path: str, recursive: bool = False) -> List[Dict[str, Any]]:
    """List files in a directory.

    Args:
      path: Absolute directory path.
      recursive: Whether to list recursively.

    Returns:
      List of dicts with ``name``, ``path``, ``kind``, ``size``.
    """
    result = await self._post("/files/list", {"path": path, "recursive": recursive})
    return result.get("files", [])  # type: ignore[no-any-return]

  async def read_file(self, path: str) -> str:
    """Read a file as text.

    Args:
      path: Absolute file path.

    Returns:
      File content as a string.
    """
    result = await self._post("/files/read", {"path": path})
    return str(result.get("content", ""))

  async def write_file(self, path: str, content: str) -> None:
    """Write text content to a file.

    Args:
      path: Absolute file path.
      content: Text content to write.
    """
    await self._post("/files/write", {"path": path, "content": content})

  async def move_file(self, from_path: str, to_path: str) -> None:
    """Move or rename a file.

    Args:
      from_path: Source path.
      to_path: Destination path.
    """
    await self._post("/files/move", {"from": from_path, "to": to_path})

  async def delete_file(self, path: str, to_trash: bool = True) -> None:
    """Delete a file.

    Args:
      path: Absolute file path.
      to_trash: Move to Trash (default) instead of permanent deletion.
    """
    await self._post("/files/delete", {"path": path, "toTrash": to_trash})

  async def file_info(self, path: str) -> Dict[str, Any]:
    """Get metadata for a file.

    Args:
      path: Absolute file path.

    Returns:
      Dict with ``size``, ``created``, ``modified``, ``kind``.
    """
    return await self._post("/files/info", {"path": path})

  # --- Clipboard ---

  async def get_clipboard(self) -> str:
    """Get the clipboard text content.

    Returns:
      Current clipboard text, or empty string if not text.
    """
    result = await self._post("/clipboard/get")
    return str(result.get("text", ""))

  async def set_clipboard(self, text: str) -> None:
    """Set the clipboard text content.

    Args:
      text: Text to put on the clipboard.
    """
    await self._post("/clipboard/set", {"text": text})

  # --- System ---

  async def system_info(self) -> Dict[str, Any]:
    """Get system information.

    Returns:
      Dict with ``hostname``, ``os_version``, ``cpu``, ``memory_gb``.
    """
    return await self._post("/system/info")

  async def get_volume(self) -> int:
    """Get system output volume level (0-100).

    Returns:
      Current volume as integer 0-100.
    """
    result = await self._post("/system/volume")
    return int(result.get("volume", 0))

  async def set_volume(self, volume: int) -> None:
    """Set system output volume level.

    Args:
      volume: Volume level 0-100.
    """
    await self._post("/system/set_volume", {"volume": max(0, min(100, volume))})

  async def get_battery(self) -> Dict[str, Any]:
    """Get battery status.

    Returns:
      Dict with ``level`` (int 0-100), ``charging`` (bool), ``time_remaining`` (int minutes or -1).
    """
    return await self._post("/system/battery")

  async def send_notification(self, title: str, message: str) -> None:
    """Send a macOS notification.

    Args:
      title: Notification title.
      message: Notification body.
    """
    await self._post("/notifications/send", {"title": title, "message": message})

  async def set_dark_mode(self, enabled: bool) -> None:
    """Enable or disable dark mode.

    Args:
      enabled: ``True`` for dark mode, ``False`` for light mode.
    """
    await self._post("/system/set_dark_mode", {"enabled": enabled})

  async def lock_screen(self) -> None:
    """Lock the screen."""
    await self._post("/system/lock")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_ui_element(data: Dict[str, Any]) -> UIElement:
  """Parse a UI element dict from the bridge into a UIElement dataclass."""
  b = data.get("bounds", {})
  bounds = ElementBounds(
    x=b.get("x", 0),
    y=b.get("y", 0),
    width=b.get("width", 0),
    height=b.get("height", 0),
  )
  children_data = data.get("children")
  children = [_parse_ui_element(c) for c in children_data] if children_data else None
  return UIElement(
    role=data.get("role", ""),
    title=data.get("title", ""),
    value=data.get("value", ""),
    bounds=bounds,
    children=children,
  )
