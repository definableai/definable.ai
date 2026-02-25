"""HTTP client for the Definable Desktop Bridge app.

Provides typed async methods for every bridge endpoint. Token is auto-read
from ``~/.definable/bridge-token`` when not explicitly provided.

Example::

    async with BridgeClient() as client:
        png_bytes = await client.capture_screen()
        await client.click(x=500, y=400)
"""

import asyncio
import base64
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    on_event: Optional[Callable] = None,
  ) -> None:
    self._base_url = f"http://{host}:{port}"
    self._token = token
    self._timeout = timeout
    self._client: Optional[httpx.AsyncClient] = None
    self._on_event = on_event

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

  async def __aexit__(self, *args: object) -> None:
    await self.close()

  # --- Internal helper ---

  async def _post(self, path: str, data: Optional[Dict[str, Any]] = None) -> Any:
    """POST to the bridge and return the unwrapped ``data`` field.

    The bridge wraps all responses in ``{"ok": true, "data": ...}``.
    Emits a :class:`BridgeCallEvent` on every call when ``on_event`` is set.

    Raises:
      httpx.ConnectError: Bridge is not reachable.
      httpx.HTTPStatusError: Bridge returned a non-2xx status.
      RuntimeError: Bridge returned ``ok=false``.
    """
    client = self._get_client()
    t0 = time()
    status_code = 0
    error = ""
    try:
      response = await client.post(path, json=data or {})
      status_code = response.status_code
      response.raise_for_status()
      body = response.json()
      if not body.get("ok", False):
        error = body.get("error", "unknown")
        raise RuntimeError(f"Bridge error: {error}")
      return body.get("data", {})
    except Exception as exc:
      if not error:
        error = str(exc)
      raise
    finally:
      self._emit_bridge_call(path, status_code, (time() - t0) * 1000, error)

  def _emit_bridge_call(self, endpoint: str, status_code: int, duration_ms: float, error: str) -> None:
    """Fire BridgeCallEvent — best-effort, never raises."""
    if self._on_event is None:
      return
    try:
      from definable.agent.interface.desktop.events import BridgeCallEvent

      event = BridgeCallEvent(
        endpoint=endpoint,
        status_code=status_code,
        duration_ms=round(duration_ms, 2),
        error=error,
      )
      result = self._on_event(event)
      if asyncio.iscoroutine(result):
        asyncio.ensure_future(result)
    except Exception:
      pass  # Best-effort — never kill the caller

  def _emit_action(self, category: str, action: str, target: str = "", value: str = "", result: str = "", error: str = "") -> None:
    """Fire DesktopActionEvent — best-effort, never raises."""
    if self._on_event is None:
      return
    try:
      from definable.agent.interface.desktop.events import DesktopActionEvent

      event = DesktopActionEvent(
        category=category,
        action=action,
        target=target,
        value=value,
        result=result,
        error=error,
      )
      cb_result = self._on_event(event)
      if asyncio.iscoroutine(cb_result):
        asyncio.ensure_future(cb_result)
    except Exception:
      pass  # Best-effort

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
    result = await self._post("/screen/capture", {"display": display, "region": region, "maxWidth": max_width})
    data = base64.b64decode(result["image"])
    self._emit_action("screen", "screenshot", target=f"display={display}", result=f"{len(data)} bytes")
    return data

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
    result = await self._post("/screen/ocr", {"region": region})
    elem_count = len(result.get("elements", []))
    self._emit_action("screen", "ocr", result=f"{elem_count} elements")
    return result

  async def find_text_on_screen(self, text: str, nth: int = 0) -> Optional[ElementBounds]:
    """Find the nth occurrence of text on screen.

    Args:
      text: Text to find (case-insensitive).
      nth: Which occurrence to return (0 = first).

    Returns:
      :class:`ElementBounds` if found, ``None`` otherwise.
    """
    try:
      result = await self._post("/screen/find_text", {"text": text, "nth": nth})
      bounds = ElementBounds(x=result["x"], y=result["y"], width=result["width"], height=result["height"])
      cx, cy = bounds.center
      self._emit_action("screen", "find_text", target=text, result=f"found at ({cx:.0f},{cy:.0f})")
      return bounds
    except Exception:
      self._emit_action("screen", "find_text", target=text, result="not found")
      return None

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
    self._emit_action("input", "click", target=f"({x:.0f},{y:.0f})", value=f"{button} x{clicks}")

  async def type_text(self, text: str) -> None:
    """Type text at the current cursor position.

    Args:
      text: Text to type.
    """
    await self._post("/input/type", {"text": text})
    self._emit_action("input", "type_text", value=text[:100])

  async def press_key(self, key: str, modifiers: Optional[List[str]] = None) -> None:
    """Press a key combination.

    Args:
      key: Key name (e.g. ``"return"``, ``"escape"``, ``"a"``).
      modifiers: Modifier keys (e.g. ``["cmd", "shift"]``).
    """
    await self._post("/input/key", {"key": key, "modifiers": modifiers or []})
    mod_str = "+".join(modifiers) + "+" if modifiers else ""
    self._emit_action("input", "press_key", value=f"{mod_str}{key}")

  async def mouse_move(self, x: float, y: float) -> None:
    """Move the mouse cursor.

    Args:
      x: X coordinate.
      y: Y coordinate.
    """
    await self._post("/input/mouse_move", {"x": x, "y": y})
    self._emit_action("input", "mouse_move", target=f"({x:.0f},{y:.0f})")

  async def scroll(self, x: float, y: float, dx: float = 0, dy: float = -3) -> None:
    """Scroll at the given position.

    Args:
      x: X coordinate to scroll at.
      y: Y coordinate to scroll at.
      dx: Horizontal scroll delta.
      dy: Vertical scroll delta (negative = down).
    """
    await self._post("/input/scroll", {"x": x, "y": y, "dx": dx, "dy": dy})
    self._emit_action("input", "scroll", target=f"({x:.0f},{y:.0f})", value=f"dx={dx},dy={dy}")

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
    self._emit_action("input", "drag", target=f"({from_x:.0f},{from_y:.0f})->({to_x:.0f},{to_y:.0f})")

  # --- Apps ---

  async def list_apps(self) -> List[AppInfo]:
    """List running applications.

    Returns:
      List of :class:`AppInfo` for each running app.
    """
    result = await self._post("/apps/list")
    apps = result if isinstance(result, list) else result.get("apps", [])
    info_list = [
      AppInfo(
        name=a["name"],
        bundle_id=a.get("bundle_id", a.get("bundleId", "")),
        pid=a.get("pid", -1),
        active=a.get("active", False),
      )
      for a in apps
    ]
    self._emit_action("app", "list_apps", result=f"{len(info_list)} apps")
    return info_list

  async def open_app(self, name: str) -> int:
    """Launch or activate an application.

    Args:
      name: App name, bundle ID, or full path.

    Returns:
      PID of the launched/activated process.
    """
    result = await self._post("/apps/open", {"name": name})
    pid = int(result.get("pid", result.get("PID", -1)))
    self._emit_action("app", "open_app", target=name, result=f"pid={pid}")
    return pid

  async def quit_app(self, name: str, force: bool = False) -> None:
    """Quit an application.

    Args:
      name: Application name.
      force: If ``True``, force-quit (SIGKILL). Otherwise graceful quit.
    """
    await self._post("/apps/quit", {"name": name, "force": force})
    self._emit_action("app", "quit_app", target=name, value="force" if force else "graceful")

  async def activate_app(self, name: str) -> None:
    """Bring an application to the foreground.

    Args:
      name: Application name.
    """
    await self._post("/apps/activate", {"name": name})
    self._emit_action("app", "activate_app", target=name)

  async def open_url(self, url: str) -> None:
    """Open a URL in the default browser.

    Args:
      url: URL to open.
    """
    await self._post("/apps/open_url", {"url": url})
    self._emit_action("app", "open_url", value=url)

  async def open_file(self, path: str) -> None:
    """Open a file with its default application.

    Args:
      path: Absolute path to the file.
    """
    await self._post("/apps/open_file", {"path": path})
    self._emit_action("app", "open_file", target=path)

  # --- Windows ---

  async def list_windows(self) -> List[WindowInfo]:
    """List open windows across all applications.

    Returns:
      List of :class:`WindowInfo` for each window.
    """
    result = await self._post("/windows/list")
    items = result if isinstance(result, list) else result.get("windows", [])
    windows = []
    for w in items:
      windows.append(
        WindowInfo(
          id=w.get("id", 0),
          app=w.get("app", ""),
          title=w.get("title", ""),
          bounds=ElementBounds(
            x=w.get("x", 0),
            y=w.get("y", 0),
            width=w.get("width", 0),
            height=w.get("height", 0),
          ),
          minimized=w.get("minimized", False),
        )
      )
    self._emit_action("window", "list_windows", result=f"{len(windows)} windows")
    return windows

  async def focus_window(self, window_id: Optional[int] = None, title: Optional[str] = None) -> None:
    """Focus a window by ID or title.

    Args:
      window_id: Window ID (from :meth:`list_windows`).
      title: Window title substring to match.
    """
    await self._post("/windows/focus", {"window_id": window_id, "title": title})
    self._emit_action("window", "focus_window", target=title or str(window_id or ""))

  async def resize_window(self, window_id: int, x: float, y: float, width: float, height: float) -> None:
    """Resize and reposition a window.

    Args:
      window_id: Window ID.
      x: New X position.
      y: New Y position.
      width: New width.
      height: New height.
    """
    await self._post("/windows/resize", {"window_id": window_id, "x": x, "y": y, "width": width, "height": height})
    self._emit_action("window", "resize_window", target=str(window_id), value=f"{width:.0f}x{height:.0f}")

  async def close_window(self, window_id: int) -> None:
    """Close a window.

    Args:
      window_id: Window ID.
    """
    await self._post("/windows/close", {"window_id": window_id})
    self._emit_action("window", "close_window", target=str(window_id))

  # --- Accessibility ---

  async def get_focused_element(self) -> Optional[UIElement]:
    """Get the currently focused UI element.

    Returns:
      :class:`UIElement` or ``None`` if nothing is focused.
    """
    try:
      result = await self._post("/ax/get_focused_element")
      elem = _parse_ui_element(result)
      self._emit_action("accessibility", "get_focused_element", result=f"{elem.role} '{elem.title}'")
      return elem
    except Exception:
      return None

  async def get_ui_tree(self, app: str, depth: int = 3) -> Dict[str, Any]:
    """Get the accessibility UI tree for an application.

    Args:
      app: Application name.
      depth: Tree depth to traverse (default: 3).

    Returns:
      Nested dict representing the UI tree.
    """
    result = await self._post("/ax/get_ui_tree", {"app": app, "depth": depth})
    self._emit_action("accessibility", "get_ui_tree", target=app, value=f"depth={depth}")
    return result

  async def find_ui_element(self, app: str, role: Optional[str] = None, title: Optional[str] = None) -> Optional[UIElement]:
    """Find a UI element in an application.

    Args:
      app: Application name.
      role: Accessibility role (e.g. ``"AXButton"``, ``"AXTextField"``).
      title: Element title or label.

    Returns:
      :class:`UIElement` or ``None`` if not found.
    """
    try:
      result = await self._post("/ax/find_element", {"app": app, "role": role, "title": title})
      elem = _parse_ui_element(result)
      self._emit_action("accessibility", "find_element", target=app, value=f"role={role} title={title}", result=f"{elem.role} '{elem.title}'")
      return elem
    except Exception:
      self._emit_action("accessibility", "find_element", target=app, value=f"role={role} title={title}", result="not found")
      return None

  async def click_ui_element(self, app: str, role: Optional[str] = None, title: Optional[str] = None) -> None:
    """Click a UI element found by accessibility attributes.

    Args:
      app: Application name.
      role: Accessibility role.
      title: Element title or label.
    """
    await self._post("/ax/perform_action", {"app": app, "role": role, "title": title, "action": "AXPress"})
    self._emit_action("accessibility", "click_element", target=app, value=f"role={role} title={title}")

  async def set_ui_value(self, app: str, role: str, title: str, value: str) -> None:
    """Set the value of a UI element (e.g. a text field).

    Args:
      app: Application name.
      role: Accessibility role.
      title: Element title or label.
      value: New value to set.
    """
    await self._post("/ax/set_value", {"app": app, "role": role, "title": title, "value": value})
    self._emit_action("accessibility", "set_value", target=app, value=f"{role} '{title}' = '{value[:50]}'")

  # --- AppleScript ---

  async def run_applescript(self, script: str) -> Dict[str, Any]:
    """Run an AppleScript.

    Args:
      script: AppleScript source code.

    Returns:
      Dict with ``output`` (str) and ``error`` (str or None).
    """
    result = await self._post("/applescript/run", {"script": script})
    error = result.get("error", "")
    self._emit_action("applescript", "run", value=script[:100], result=result.get("output", "")[:100], error=error or "")
    return result

  # --- Shell ---

  async def run_shell(
    self,
    command: List[str],
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
  ) -> Dict[str, Any]:
    """Execute a shell command.

    Args:
      command: Command and arguments as a list (e.g. ``["ls", "-la"]``).
      cwd: Working directory.
      env: Environment variables.
      timeout: Timeout in seconds.

    Returns:
      Dict with ``stdout``, ``stderr``, ``exit_code``, ``timed_out``, ``success``.
    """
    data: Dict[str, Any] = {"command": command}
    if cwd is not None:
      data["cwd"] = cwd
    if env is not None:
      data["env"] = env
    if timeout is not None:
      data["timeout"] = timeout
    result = await self._post("/shell/run", data)
    cmd_str = " ".join(command)[:100]
    exit_code = result.get("exit_code", "?")
    stderr = "" if result.get("success") else result.get("stderr", "")[:100]
    self._emit_action("shell", "run", value=cmd_str, result=f"exit={exit_code}", error=stderr)
    return result

  # --- Camera ---

  async def camera_list(self) -> List[Dict[str, str]]:
    """List available camera devices.

    Returns:
      List of dicts with ``id``, ``name``, ``position``.
    """
    result = await self._post("/camera/list")
    devices = result if isinstance(result, list) else []
    self._emit_action("camera", "list_devices", result=f"{len(devices)} cameras")
    return devices

  async def camera_snap(
    self,
    facing: str = "front",
    max_width: int = 1600,
    quality: float = 0.9,
    out_path: Optional[str] = None,
  ) -> bytes:
    """Take a photo with the camera.

    Args:
      facing: ``"front"`` or ``"back"``.
      max_width: Maximum image width.
      quality: JPEG quality (0.0-1.0).
      out_path: If set, the bridge writes the JPEG to this path on disk.

    Returns:
      JPEG image as raw bytes.
    """
    payload: Dict[str, Any] = {
      "facing": facing,
      "max_width": max_width,
      "quality": quality,
    }
    if out_path is not None:
      payload["out_path"] = out_path
    result = await self._post("/camera/snap", payload)
    data = base64.b64decode(result["image"])
    target = out_path or facing
    self._emit_action("camera", "snap", target=target, result=f"{len(data)} bytes")
    return data

  async def camera_clip(
    self,
    facing: str = "front",
    duration_ms: int = 3000,
    include_audio: bool = False,
    out_path: Optional[str] = None,
  ) -> Dict[str, Any]:
    """Record a video clip with the camera.

    Args:
      facing: ``"front"`` or ``"back"``.
      duration_ms: Duration in milliseconds.
      include_audio: Whether to include audio.
      out_path: Optional output file path.

    Returns:
      Dict with ``path``, ``duration_ms``, ``has_audio``.
    """
    result = await self._post(
      "/camera/clip",
      {
        "facing": facing,
        "duration_ms": duration_ms,
        "include_audio": include_audio,
        "out_path": out_path,
      },
    )
    self._emit_action("camera", "clip", target=facing, value=f"{duration_ms}ms", result=result.get("path", ""))
    return result

  # --- Screen Recording ---

  async def screen_record(
    self,
    screen_index: int = 0,
    duration_ms: int = 10000,
    fps: float = 10,
    include_audio: bool = False,
    out_path: Optional[str] = None,
  ) -> Dict[str, Any]:
    """Record the screen.

    Args:
      screen_index: Display index (0 = primary).
      duration_ms: Duration in milliseconds.
      fps: Frames per second.
      include_audio: Whether to include audio.
      out_path: Optional output file path.

    Returns:
      Dict with ``path``, ``has_audio``.
    """
    result = await self._post(
      "/screen/record",
      {
        "screen_index": screen_index,
        "duration_ms": duration_ms,
        "fps": fps,
        "include_audio": include_audio,
        "out_path": out_path,
      },
    )
    self._emit_action("screen", "record", target=f"display={screen_index}", value=f"{duration_ms}ms", result=result.get("path", ""))
    return result

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
    files = result if isinstance(result, list) else result.get("files", [])
    self._emit_action("file", "list_files", target=path, result=f"{len(files)} files")
    return files

  async def read_file(self, path: str) -> str:
    """Read a file as text.

    Args:
      path: Absolute file path.

    Returns:
      File content as a string.
    """
    result = await self._post("/files/read", {"path": path})
    content = str(result.get("content", ""))
    self._emit_action("file", "read_file", target=path, result=f"{len(content)} chars")
    return content

  async def write_file(self, path: str, content: str) -> None:
    """Write text content to a file.

    Args:
      path: Absolute file path.
      content: Text content to write.
    """
    await self._post("/files/write", {"path": path, "content": content})
    self._emit_action("file", "write_file", target=path, result=f"{len(content)} chars")

  async def move_file(self, from_path: str, to_path: str) -> None:
    """Move or rename a file.

    Args:
      from_path: Source path.
      to_path: Destination path.
    """
    await self._post("/files/move", {"from": from_path, "to": to_path})
    self._emit_action("file", "move_file", target=from_path, value=to_path)

  async def delete_file(self, path: str, to_trash: bool = True) -> None:
    """Delete a file.

    Args:
      path: Absolute file path.
      to_trash: Move to Trash (default) instead of permanent deletion.
    """
    await self._post("/files/delete", {"path": path, "toTrash": to_trash})
    self._emit_action("file", "delete_file", target=path, value="trash" if to_trash else "permanent")

  async def file_info(self, path: str) -> Dict[str, Any]:
    """Get metadata for a file.

    Args:
      path: Absolute file path.

    Returns:
      Dict with ``size``, ``created``, ``modified``, ``kind``.
    """
    result = await self._post("/files/info", {"path": path})
    self._emit_action("file", "file_info", target=path)
    return result

  # --- Clipboard ---

  async def get_clipboard(self) -> str:
    """Get the clipboard text content.

    Returns:
      Current clipboard text, or empty string if not text.
    """
    result = await self._post("/clipboard/get")
    text = str(result.get("text", ""))
    self._emit_action("clipboard", "get", result=f"{len(text)} chars")
    return text

  async def set_clipboard(self, text: str) -> None:
    """Set the clipboard text content.

    Args:
      text: Text to put on the clipboard.
    """
    await self._post("/clipboard/set", {"text": text})
    self._emit_action("clipboard", "set", value=text[:100])

  # --- System ---

  async def system_info(self) -> Dict[str, Any]:
    """Get system information.

    Returns:
      Dict with ``hostname``, ``os_version``, ``cpu``, ``memory_gb``.
    """
    result = await self._post("/system/info")
    self._emit_action("system", "info", result=result.get("hostname", ""))
    return result

  async def get_volume(self) -> int:
    """Get system output volume level (0-100).

    Returns:
      Current volume as integer 0-100.
    """
    result = await self._post("/system/volume")
    vol = int(result.get("volume", 0))
    self._emit_action("system", "get_volume", result=str(vol))
    return vol

  async def set_volume(self, volume: int) -> None:
    """Set system output volume level.

    Args:
      volume: Volume level 0-100.
    """
    await self._post("/system/set_volume", {"volume": max(0, min(100, volume))})
    self._emit_action("system", "set_volume", value=str(volume))

  async def get_battery(self) -> Dict[str, Any]:
    """Get battery status.

    Returns:
      Dict with ``level`` (int 0-100), ``charging`` (bool), ``time_remaining`` (int minutes or -1).
    """
    result = await self._post("/system/battery")
    self._emit_action("system", "get_battery", result=f"{result.get('level', '?')}%")
    return result

  async def send_notification(self, title: str, message: str) -> None:
    """Send a macOS notification.

    Args:
      title: Notification title.
      message: Notification body.
    """
    await self._post("/notifications/send", {"title": title, "message": message})
    self._emit_action("system", "send_notification", target=title, value=message[:100])

  async def set_dark_mode(self, enabled: bool) -> None:
    """Enable or disable dark mode.

    Args:
      enabled: ``True`` for dark mode, ``False`` for light mode.
    """
    await self._post("/system/set_dark_mode", {"enabled": enabled})
    self._emit_action("system", "set_dark_mode", value="on" if enabled else "off")

  async def lock_screen(self) -> None:
    """Lock the screen."""
    await self._post("/system/lock")
    self._emit_action("system", "lock_screen")


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
