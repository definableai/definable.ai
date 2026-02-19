"""MacOS skill — control a Mac like a human through the Definable Desktop Bridge.

Gives any agent screen-reading, input simulation, app management, window
control, file operations, clipboard access, and system actions. Requires
the Definable Desktop Bridge app to be running on ``localhost:7777``.

⚠️  SECURITY: This skill executes real macOS actions. Always use
``allowed_apps`` or ``blocked_apps`` in production to limit exposure.

Example::

    from definable.skill.builtin.macos import MacOS

    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        skills=[MacOS(allowed_apps={"Safari", "TextEdit"})],
    )
    output = await agent.arun("Open Safari and navigate to apple.com")
"""

import base64
from typing import Any, List, Optional, Set

from definable.skill.base import Skill
from definable.tool.decorator import tool


class MacOS(Skill):
  """Skill for controlling macOS through the Definable Desktop Bridge.

  Provides tools for screen reading, mouse/keyboard input, app and window
  management, accessibility inspection, file operations, clipboard, and
  system information. All bridge communication is handled by a lazily-
  initialized :class:`~definable.agent.interface.desktop.bridge_client.BridgeClient`.

  Args:
    bridge_host: Bridge hostname (default: ``"127.0.0.1"``).
    bridge_port: Bridge port (default: ``7777``).
    bridge_token: Bearer token. If ``None``, reads from ``~/.definable/bridge-token``.
    allowed_apps: If set, only these app names can be targeted by tools.
    blocked_apps: App names that can never be targeted by tools.
    enable_applescript: Whether to expose the AppleScript execution tool.
    enable_file_write: Whether to expose file-write and file-move tools.
    enable_input: Whether to expose mouse/keyboard input tools.

  Example::

    # Read-only agent (no input simulation, no file writes)
    MacOS(enable_input=False, enable_file_write=False)

    # Restricted to specific apps
    MacOS(allowed_apps={"Safari", "TextEdit", "Terminal"})
  """

  name = "macos"

  def __init__(
    self,
    *,
    bridge_host: str = "127.0.0.1",
    bridge_port: int = 7777,
    bridge_token: Optional[str] = None,
    allowed_apps: Optional[Set[str]] = None,
    blocked_apps: Optional[Set[str]] = None,
    enable_applescript: bool = True,
    enable_file_write: bool = True,
    enable_input: bool = True,
  ) -> None:
    super().__init__()
    self._bridge_host = bridge_host
    self._bridge_port = bridge_port
    self._bridge_token = bridge_token
    self._allowed_apps = allowed_apps
    self._blocked_apps = blocked_apps or set()
    self._enable_applescript = enable_applescript
    self._enable_file_write = enable_file_write
    self._enable_input = enable_input
    self._client: Any = None  # BridgeClient, lazy

  @property
  def instructions(self) -> str:  # type: ignore[override]
    parts = [
      "You can control this Mac through the macOS Desktop Bridge.",
      "Workflow: take a screenshot → identify what you see → interact with the target → verify the result.",
      "Always screenshot before and after actions to confirm they succeeded.",
    ]
    if self._allowed_apps:
      parts.append(f"You may only target these apps: {', '.join(sorted(self._allowed_apps))}.")
    if not self._enable_input:
      parts.append("Input simulation (click, type, key) is disabled — you are in read-only mode.")
    if not self._enable_file_write:
      parts.append("File write and move operations are disabled.")
    if not self._enable_applescript:
      parts.append("AppleScript execution is disabled.")
    parts.append("If the bridge is not running, tools will return a friendly error — tell the user to start the Definable Desktop Bridge app.")
    return " ".join(parts)

  def _get_client(self) -> Any:
    """Lazily create and return the BridgeClient."""
    if self._client is None:
      from definable.agent.interface.desktop.bridge_client import BridgeClient

      self._client = BridgeClient(
        host=self._bridge_host,
        port=self._bridge_port,
        token=self._bridge_token,
      )
    return self._client

  def _check_app_allowed(self, app_name: str) -> Optional[str]:
    """Return an error string if ``app_name`` is not permitted, else ``None``."""
    if self._allowed_apps is not None and app_name not in self._allowed_apps:
      return f"App '{app_name}' is not in the allowed list: {sorted(self._allowed_apps)}. Ask the user to update the MacOS skill configuration."
    if app_name in self._blocked_apps:
      return f"App '{app_name}' is blocked. Ask the user to update the MacOS skill configuration."
    return None

  def _bridge_error_msg(self, host: str, port: int) -> str:
    return f"Bridge is not running. Please start the Definable Desktop Bridge app at http://{host}:{port} and try again."

  @property
  def tools(self) -> list:  # noqa: C901
    skill = self

    all_tools: List[Any] = []

    # ── Screen ─────────────────────────────────────────────────────────────

    @tool
    async def screenshot() -> str:
      """Capture a screenshot of the current screen.

      The image is downscaled to 1280px wide and encoded as JPEG to stay
      within LLM context budgets. Full retina PNG would exceed token limits.

      Returns:
        A base64-encoded JPEG data URI (data:image/jpeg;base64,...) that vision
        models can interpret directly, or an error string.
      """
      try:
        client = skill._get_client()
        jpeg_bytes = await client.capture_screen(max_width=512)
        b64 = base64.b64encode(jpeg_bytes).decode()
        return f"data:image/jpeg;base64,{b64}"
      except Exception as exc:
        if "connect" in str(exc).lower() or "connection" in str(exc).lower():
          return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
        return f"Screenshot failed: {exc}"

    all_tools.append(screenshot)

    @tool
    async def read_screen(
      region_x: Optional[int] = None,
      region_y: Optional[int] = None,
      region_w: Optional[int] = None,
      region_h: Optional[int] = None,
    ) -> str:
      """Read text visible on screen using OCR.

      Args:
        region_x: Optional crop region left edge (pixels).
        region_y: Optional crop region top edge (pixels).
        region_w: Optional crop region width (pixels).
        region_h: Optional crop region height (pixels).

      Returns:
        Text found on screen, or an error string.
      """
      try:
        region = None
        if all(v is not None for v in [region_x, region_y, region_w, region_h]):
          region = {"x": region_x, "y": region_y, "width": region_w, "height": region_h}
        client = skill._get_client()
        result = await client.ocr_screen(region=region)
        return result.get("text", "(no text found)")
      except Exception as exc:
        if "connect" in str(exc).lower() or "connection" in str(exc).lower():
          return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
        return f"Screen read failed: {exc}"

    all_tools.append(read_screen)

    @tool
    async def find_text_on_screen(text: str) -> str:
      """Find where specific text appears on screen.

      Args:
        text: The text to locate (case-insensitive).

      Returns:
        Coordinates of the text (center x, y) or "not found".
      """
      try:
        client = skill._get_client()
        bounds = await client.find_text_on_screen(text)
        if bounds is None:
          return f"Text '{text}' not found on screen."
        cx, cy = bounds.center
        b = bounds
        return f"Found '{text}' at center ({cx:.0f}, {cy:.0f}), bounds: x={b.x:.0f}, y={b.y:.0f}, w={b.width:.0f}, h={b.height:.0f}"
      except Exception as exc:
        if "connect" in str(exc).lower() or "connection" in str(exc).lower():
          return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
        return f"Find text failed: {exc}"

    all_tools.append(find_text_on_screen)

    # ── Input (gated by enable_input) ──────────────────────────────────────

    if skill._enable_input:

      @tool
      async def click(
        x: float,
        y: float,
        button: str = "left",
        clicks: int = 1,
      ) -> str:
        """Click at screen coordinates.

        Args:
          x: X coordinate in screen pixels.
          y: Y coordinate in screen pixels.
          button: Mouse button — "left", "right", or "middle".
          clicks: Number of clicks (2 for double-click).

        Returns:
          "ok" on success or an error string.
        """
        try:
          client = skill._get_client()
          await client.click(x=x, y=y, button=button, clicks=clicks)
          return "ok"
        except Exception as exc:
          if "connect" in str(exc).lower() or "connection" in str(exc).lower():
            return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
          return f"Click failed: {exc}"

      all_tools.append(click)

      @tool
      async def type_text(text: str) -> str:
        """Type text at the current cursor position.

        Args:
          text: Text to type. Supports Unicode characters.

        Returns:
          "ok" on success or an error string.
        """
        try:
          client = skill._get_client()
          await client.type_text(text)
          return "ok"
        except Exception as exc:
          if "connect" in str(exc).lower() or "connection" in str(exc).lower():
            return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
          return f"Type text failed: {exc}"

      all_tools.append(type_text)

      @tool
      async def press_key(key: str, modifiers: str = "") -> str:
        """Press a key combination.

        Args:
          key: Key name (e.g. "return", "escape", "a", "tab", "space").
          modifiers: Comma-separated modifier keys (e.g. "cmd,shift" or "cmd").

        Returns:
          "ok" on success or an error string.
        """
        try:
          mod_list = [m.strip() for m in modifiers.split(",") if m.strip()]
          client = skill._get_client()
          await client.press_key(key=key, modifiers=mod_list)
          return "ok"
        except Exception as exc:
          if "connect" in str(exc).lower() or "connection" in str(exc).lower():
            return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
          return f"Key press failed: {exc}"

      all_tools.append(press_key)

      @tool
      async def scroll(
        x: float,
        y: float,
        direction: str = "down",
        amount: float = 3,
      ) -> str:
        """Scroll at the given screen position.

        Args:
          x: X coordinate to scroll at.
          y: Y coordinate to scroll at.
          direction: "up", "down", "left", or "right".
          amount: Number of scroll units (default: 3).

        Returns:
          "ok" on success or an error string.
        """
        try:
          client = skill._get_client()
          dy = -amount if direction == "down" else (amount if direction == "up" else 0)
          dx = -amount if direction == "right" else (amount if direction == "left" else 0)
          await client.scroll(x=x, y=y, dx=dx, dy=dy)
          return "ok"
        except Exception as exc:
          if "connect" in str(exc).lower() or "connection" in str(exc).lower():
            return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
          return f"Scroll failed: {exc}"

      all_tools.append(scroll)

      @tool
      async def drag(from_x: float, from_y: float, to_x: float, to_y: float) -> str:
        """Click and drag from one screen position to another.

        Args:
          from_x: Starting X coordinate.
          from_y: Starting Y coordinate.
          to_x: Ending X coordinate.
          to_y: Ending Y coordinate.

        Returns:
          "ok" on success or an error string.
        """
        try:
          client = skill._get_client()
          await client.drag(from_x=from_x, from_y=from_y, to_x=to_x, to_y=to_y)
          return "ok"
        except Exception as exc:
          if "connect" in str(exc).lower() or "connection" in str(exc).lower():
            return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
          return f"Drag failed: {exc}"

      all_tools.append(drag)

    # ── Apps ───────────────────────────────────────────────────────────────

    @tool
    async def list_running_apps() -> str:
      """List all currently running applications.

      Returns:
        A newline-separated list of running app names, or an error string.
      """
      try:
        client = skill._get_client()
        apps = await client.list_apps()
        if not apps:
          return "No apps running."
        lines = [f"{'* ' if a.active else '  '}{a.name} (pid={a.pid})" for a in apps]
        return "\n".join(lines)
      except Exception as exc:
        if "connect" in str(exc).lower() or "connection" in str(exc).lower():
          return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
        return f"List apps failed: {exc}"

    all_tools.append(list_running_apps)

    @tool
    async def open_app(name: str) -> str:
      """Launch or activate an application.

      Args:
        name: App name (e.g. "Safari"), bundle ID (e.g. "com.apple.Safari"),
          or full path.

      Returns:
        PID of the launched process on success, or an error string.
      """
      err = skill._check_app_allowed(name)
      if err:
        return err
      try:
        client = skill._get_client()
        pid = await client.open_app(name)
        return f"Opened '{name}' (pid={pid})"
      except Exception as exc:
        if "connect" in str(exc).lower() or "connection" in str(exc).lower():
          return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
        return f"Open app failed: {exc}"

    all_tools.append(open_app)

    @tool
    async def quit_app(name: str, force: bool = False) -> str:
      """Quit an application.

      Args:
        name: Application name.
        force: Force-quit without saving (default: False).

      Returns:
        "ok" on success or an error string.
      """
      err = skill._check_app_allowed(name)
      if err:
        return err
      try:
        client = skill._get_client()
        await client.quit_app(name=name, force=force)
        return "ok"
      except Exception as exc:
        if "connect" in str(exc).lower() or "connection" in str(exc).lower():
          return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
        return f"Quit app failed: {exc}"

    all_tools.append(quit_app)

    @tool
    async def activate_app(name: str) -> str:
      """Bring an application to the foreground.

      Args:
        name: Application name.

      Returns:
        "ok" on success or an error string.
      """
      err = skill._check_app_allowed(name)
      if err:
        return err
      try:
        client = skill._get_client()
        await client.activate_app(name)
        return "ok"
      except Exception as exc:
        if "connect" in str(exc).lower() or "connection" in str(exc).lower():
          return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
        return f"Activate app failed: {exc}"

    all_tools.append(activate_app)

    @tool
    async def open_url(url: str) -> str:
      """Open a URL in the default browser.

      Args:
        url: URL to open (e.g. "https://apple.com").

      Returns:
        "ok" on success or an error string.
      """
      try:
        client = skill._get_client()
        await client.open_url(url)
        return "ok"
      except Exception as exc:
        if "connect" in str(exc).lower() or "connection" in str(exc).lower():
          return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
        return f"Open URL failed: {exc}"

    all_tools.append(open_url)

    # ── Windows ────────────────────────────────────────────────────────────

    @tool
    async def list_windows() -> str:
      """List all open windows across all applications.

      Returns:
        A formatted list of windows with their app, title, and bounds.
      """
      try:
        client = skill._get_client()
        windows = await client.list_windows()
        if not windows:
          return "No windows found."
        lines = [f"[{w.id}] {w.app} — '{w.title}' ({w.bounds.width:.0f}x{w.bounds.height:.0f} at {w.bounds.x:.0f},{w.bounds.y:.0f})" for w in windows]
        return "\n".join(lines)
      except Exception as exc:
        if "connect" in str(exc).lower() or "connection" in str(exc).lower():
          return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
        return f"List windows failed: {exc}"

    all_tools.append(list_windows)

    @tool
    async def focus_window(title: str) -> str:
      """Bring a window to focus by matching its title.

      Args:
        title: Window title or a substring to match.

      Returns:
        "ok" on success or an error string.
      """
      try:
        client = skill._get_client()
        await client.focus_window(title=title)
        return "ok"
      except Exception as exc:
        if "connect" in str(exc).lower() or "connection" in str(exc).lower():
          return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
        return f"Focus window failed: {exc}"

    all_tools.append(focus_window)

    # ── Accessibility ──────────────────────────────────────────────────────

    @tool
    async def find_element(app: str, role: str = "", title: str = "") -> str:
      """Find a UI element in an application using accessibility APIs.

      Args:
        app: Application name (e.g. "Safari").
        role: Accessibility role (e.g. "AXButton", "AXTextField"). Optional.
        title: Element label or title. Optional.

      Returns:
        Element info with role, title, value, and center coordinates, or an error.
      """
      err = skill._check_app_allowed(app)
      if err:
        return err
      try:
        client = skill._get_client()
        elem = await client.find_ui_element(app=app, role=role or None, title=title or None)
        if elem is None:
          return f"No element found in '{app}' with role='{role}' title='{title}'."
        cx, cy = elem.bounds.center
        return f"Found {elem.role} '{elem.title}' value='{elem.value}' center=({cx:.0f},{cy:.0f})"
      except Exception as exc:
        if "connect" in str(exc).lower() or "connection" in str(exc).lower():
          return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
        return f"Find element failed: {exc}"

    all_tools.append(find_element)

    @tool
    async def get_ui_tree(app: str, depth: int = 3) -> str:
      """Get the accessibility UI tree for an application.

      Args:
        app: Application name.
        depth: Tree depth to traverse (default: 3, max: 8).

      Returns:
        JSON representation of the UI tree, or an error string.
      """
      err = skill._check_app_allowed(app)
      if err:
        return err
      try:
        import json as _json

        client = skill._get_client()
        tree = await client.get_ui_tree(app=app, depth=min(depth, 8))
        return _json.dumps(tree, indent=2)
      except Exception as exc:
        if "connect" in str(exc).lower() or "connection" in str(exc).lower():
          return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
        return f"Get UI tree failed: {exc}"

    all_tools.append(get_ui_tree)

    if skill._enable_input:

      @tool
      async def click_element(app: str, role: str = "", title: str = "") -> str:
        """Click a UI element found by accessibility attributes.

        Preferred over coordinate-based clicking when elements are identifiable
        by role or title — more reliable across screen resolutions.

        Args:
          app: Application name.
          role: Accessibility role (e.g. "AXButton").
          title: Element label or title.

        Returns:
          "ok" on success or an error string.
        """
        err = skill._check_app_allowed(app)
        if err:
          return err
        try:
          client = skill._get_client()
          await client.click_ui_element(app=app, role=role or None, title=title or None)
          return "ok"
        except Exception as exc:
          if "connect" in str(exc).lower() or "connection" in str(exc).lower():
            return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
          return f"Click element failed: {exc}"

      all_tools.append(click_element)

      @tool
      async def set_element_value(app: str, role: str, title: str, value: str) -> str:
        """Set the value of a UI element (e.g. a text field or slider).

        Args:
          app: Application name.
          role: Accessibility role (e.g. "AXTextField", "AXSlider").
          title: Element label or title.
          value: New value to set.

        Returns:
          "ok" on success or an error string.
        """
        err = skill._check_app_allowed(app)
        if err:
          return err
        try:
          client = skill._get_client()
          await client.set_ui_value(app=app, role=role, title=title, value=value)
          return "ok"
        except Exception as exc:
          if "connect" in str(exc).lower() or "connection" in str(exc).lower():
            return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
          return f"Set element value failed: {exc}"

      all_tools.append(set_element_value)

    # ── AppleScript (gated) ────────────────────────────────────────────────

    if skill._enable_applescript:

      @tool
      async def run_applescript(script: str) -> str:
        """Execute an AppleScript and return its output.

        Use for complex automation that cannot be achieved with other tools
        (e.g. controlling apps that have poor accessibility support).

        Args:
          script: AppleScript source code to execute.

        Returns:
          Script output string, or an error string.
        """
        try:
          client = skill._get_client()
          result = await client.run_applescript(script)
          if result.get("error"):
            return f"AppleScript error: {result['error']}"
          return result.get("output", "(no output)")
        except Exception as exc:
          if "connect" in str(exc).lower() or "connection" in str(exc).lower():
            return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
          return f"AppleScript failed: {exc}"

      all_tools.append(run_applescript)

    # ── Files ──────────────────────────────────────────────────────────────

    @tool
    async def list_files(path: str, recursive: bool = False) -> str:
      """List files in a directory.

      Args:
        path: Absolute directory path (e.g. "/Users/you/Documents").
        recursive: Whether to list subdirectories recursively.

      Returns:
        Newline-separated file list or an error string.
      """
      try:
        client = skill._get_client()
        files = await client.list_files(path=path, recursive=recursive)
        if not files:
          return f"No files found in '{path}'."
        return "\n".join(f.get("path", f.get("name", "?")) for f in files)
      except Exception as exc:
        if "connect" in str(exc).lower() or "connection" in str(exc).lower():
          return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
        return f"List files failed: {exc}"

    all_tools.append(list_files)

    @tool
    async def read_file(path: str) -> str:
      """Read a file and return its text content.

      Args:
        path: Absolute file path.

      Returns:
        File content as a string, or an error string.
      """
      try:
        client = skill._get_client()
        return await client.read_file(path)
      except Exception as exc:
        if "connect" in str(exc).lower() or "connection" in str(exc).lower():
          return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
        return f"Read file failed: {exc}"

    all_tools.append(read_file)

    if skill._enable_file_write:

      @tool
      async def write_file(path: str, content: str) -> str:
        """Write text content to a file (creates or overwrites).

        Args:
          path: Absolute file path.
          content: Text content to write.

        Returns:
          "ok" on success or an error string.
        """
        try:
          client = skill._get_client()
          await client.write_file(path=path, content=content)
          return "ok"
        except Exception as exc:
          if "connect" in str(exc).lower() or "connection" in str(exc).lower():
            return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
          return f"Write file failed: {exc}"

      all_tools.append(write_file)

      @tool
      async def move_file(from_path: str, to_path: str) -> str:
        """Move or rename a file.

        Args:
          from_path: Source absolute path.
          to_path: Destination absolute path.

        Returns:
          "ok" on success or an error string.
        """
        try:
          client = skill._get_client()
          await client.move_file(from_path=from_path, to_path=to_path)
          return "ok"
        except Exception as exc:
          if "connect" in str(exc).lower() or "connection" in str(exc).lower():
            return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
          return f"Move file failed: {exc}"

      all_tools.append(move_file)

    # ── Clipboard ──────────────────────────────────────────────────────────

    @tool
    async def get_clipboard() -> str:
      """Get the current clipboard text content.

      Returns:
        Clipboard text or empty string if clipboard is not text.
      """
      try:
        client = skill._get_client()
        return await client.get_clipboard()
      except Exception as exc:
        if "connect" in str(exc).lower() or "connection" in str(exc).lower():
          return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
        return f"Get clipboard failed: {exc}"

    all_tools.append(get_clipboard)

    if skill._enable_input:

      @tool
      async def set_clipboard(text: str) -> str:
        """Set the clipboard text content.

        Args:
          text: Text to place on the clipboard.

        Returns:
          "ok" on success or an error string.
        """
        try:
          client = skill._get_client()
          await client.set_clipboard(text)
          return "ok"
        except Exception as exc:
          if "connect" in str(exc).lower() or "connection" in str(exc).lower():
            return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
          return f"Set clipboard failed: {exc}"

      all_tools.append(set_clipboard)

    # ── System ─────────────────────────────────────────────────────────────

    @tool
    async def system_info() -> str:
      """Get system information (hostname, OS version, CPU, memory).

      Returns:
        Formatted system info string or an error string.
      """
      try:
        client = skill._get_client()
        info = await client.system_info()
        return (
          f"Hostname: {info.get('hostname', '?')}\n"
          f"OS: {info.get('os_version', '?')}\n"
          f"CPU: {info.get('cpu', '?')}\n"
          f"Memory: {info.get('memory_gb', '?')} GB"
        )
      except Exception as exc:
        if "connect" in str(exc).lower() or "connection" in str(exc).lower():
          return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
        return f"System info failed: {exc}"

    all_tools.append(system_info)

    @tool
    async def get_battery() -> str:
      """Get battery level and charging status.

      Returns:
        Battery status string, or "No battery" on desktop Macs.
      """
      try:
        client = skill._get_client()
        info = await client.get_battery()
        level = info.get("level", -1)
        if level < 0:
          return "No battery (desktop Mac or bridge error)."
        charging = info.get("charging", False)
        time_rem = info.get("time_remaining", -1)
        status = "charging" if charging else "discharging"
        time_str = f", {time_rem} min remaining" if time_rem > 0 else ""
        return f"Battery: {level}% ({status}{time_str})"
      except Exception as exc:
        if "connect" in str(exc).lower() or "connection" in str(exc).lower():
          return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
        return f"Get battery failed: {exc}"

    all_tools.append(get_battery)

    @tool
    async def set_volume(volume: int) -> str:
      """Set the system output volume level.

      Args:
        volume: Volume level 0-100.

      Returns:
        "ok" on success or an error string.
      """
      try:
        client = skill._get_client()
        await client.set_volume(volume)
        return "ok"
      except Exception as exc:
        if "connect" in str(exc).lower() or "connection" in str(exc).lower():
          return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
        return f"Set volume failed: {exc}"

    all_tools.append(set_volume)

    @tool
    async def send_notification(title: str, message: str) -> str:
      """Send a macOS notification banner.

      Args:
        title: Notification title.
        message: Notification body text.

      Returns:
        "ok" on success or an error string.
      """
      try:
        client = skill._get_client()
        await client.send_notification(title=title, message=message)
        return "ok"
      except Exception as exc:
        if "connect" in str(exc).lower() or "connection" in str(exc).lower():
          return skill._bridge_error_msg(skill._bridge_host, skill._bridge_port)
        return f"Send notification failed: {exc}"

    all_tools.append(send_notification)

    return all_tools

  def teardown(self) -> None:
    """Close the bridge client if it was created."""
    if self._client is not None:
      try:
        import asyncio

        loop = asyncio.get_running_loop()
        loop.create_task(self._client.close())
      except RuntimeError:
        pass  # No running loop — leave for GC
