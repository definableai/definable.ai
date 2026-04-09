"""Persistent agent settings stored in ``.definable/settings.json``.

Currently stores:
  - ``tool_permissions``: per-tool permission rules from "always allow" / "always deny".

The file is human-readable JSON, created lazily on first write.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from definable.agent.hitl.types import PermissionAction
from definable.utils.log import log_debug, log_warning
from definable.utils.workspace import workspace_path


_SETTINGS_FILE = "settings.json"


@dataclass
class Settings:
  """Load, query, and persist agent settings."""

  tool_permissions: Dict[str, str] = field(default_factory=dict)

  # -- Load / Save ----------------------------------------------------------

  @classmethod
  def load(cls, path: Optional[Path] = None) -> "Settings":
    """Load from ``.definable/settings.json``. Returns defaults if missing."""
    path = path or cls._default_path()
    if not path.exists():
      return cls()
    try:
      data = json.loads(path.read_text(encoding="utf-8"))
      return cls(tool_permissions=data.get("tool_permissions", {}))
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
      log_warning(f"Failed to load settings from {path}: {exc}")
      return cls()

  def save(self, path: Optional[Path] = None) -> None:
    """Write settings to ``.definable/settings.json`` (human-readable)."""
    path = path or self._default_path()
    data = {"tool_permissions": self.tool_permissions}
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log_debug(f"Settings saved to {path}")

  # -- Permission helpers ---------------------------------------------------

  def get_tool_permission(self, tool_name: str) -> Optional[PermissionAction]:
    """Get persisted permission for a tool, or ``None`` if not set."""
    raw = self.tool_permissions.get(tool_name)
    if raw is None:
      return None
    try:
      return PermissionAction(raw)
    except ValueError:
      return None

  def set_tool_permission(self, tool_name: str, action: PermissionAction, *, path: Optional[Path] = None) -> None:
    """Set a persistent permission rule and save immediately."""
    self.tool_permissions[tool_name] = action.value
    self.save(path)

  # -- Internal -------------------------------------------------------------

  @staticmethod
  def _default_path() -> Path:
    return workspace_path(_SETTINGS_FILE)
