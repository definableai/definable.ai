"""Tests for HITL settings persistence."""

import json
from pathlib import Path


from definable.agent.hitl.settings import Settings
from definable.agent.hitl.types import PermissionAction


class TestSettings:
  def test_load_missing_file(self, tmp_path: Path):
    path = tmp_path / "settings.json"
    settings = Settings.load(path=path)
    assert settings.tool_permissions == {}

  def test_load_and_save_roundtrip(self, tmp_path: Path):
    path = tmp_path / "settings.json"
    settings = Settings(tool_permissions={"bash": "allow", "delete_file": "deny"})
    settings.save(path=path)

    loaded = Settings.load(path=path)
    assert loaded.tool_permissions == {"bash": "allow", "delete_file": "deny"}

  def test_save_creates_readable_json(self, tmp_path: Path):
    path = tmp_path / "settings.json"
    settings = Settings(tool_permissions={"bash": "allow"})
    settings.save(path=path)

    raw = json.loads(path.read_text())
    assert raw == {"tool_permissions": {"bash": "allow"}}

  def test_get_tool_permission_found(self):
    settings = Settings(tool_permissions={"bash": "allow"})
    assert settings.get_tool_permission("bash") == PermissionAction.allow

  def test_get_tool_permission_not_found(self):
    settings = Settings()
    assert settings.get_tool_permission("bash") is None

  def test_get_tool_permission_invalid_value(self):
    settings = Settings(tool_permissions={"bash": "garbage"})
    assert settings.get_tool_permission("bash") is None

  def test_set_tool_permission(self, tmp_path: Path):
    path = tmp_path / "settings.json"
    settings = Settings()
    settings.set_tool_permission("bash", PermissionAction.allow, path=path)

    assert settings.tool_permissions["bash"] == "allow"
    # Verify it was persisted
    loaded = Settings.load(path=path)
    assert loaded.tool_permissions["bash"] == "allow"

  def test_load_corrupted_file(self, tmp_path: Path):
    path = tmp_path / "settings.json"
    path.write_text("not json", encoding="utf-8")
    settings = Settings.load(path=path)
    assert settings.tool_permissions == {}
