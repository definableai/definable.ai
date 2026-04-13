"""
Unit tests: MacOS skill configuration and pure-logic behaviour.

No bridge running, no API calls, no real macOS interaction.
All tests are pure logic: defaults, allow/block lists, instructions text.

Covers:
  - MacOS __init__ defaults
  - _check_app_allowed: allowed list enforced
  - _check_app_allowed: blocked list enforced
  - _check_app_allowed: both None -> all apps pass
  - _check_app_allowed: allowed overrides an otherwise-blocked name when in allow list
  - instructions property reflects config flags
  - tools count varies by enable_input / enable_file_write / enable_applescript
  - DesktopConfig defaults and frozen-dataclass immutability
  - BridgeClient token file resolution

Migrated from tests_e2e/unit/test_macos_skill_config.py -- all original tests preserved.
"""

from __future__ import annotations

import pytest

from definable.agent.interface.desktop.config import DesktopConfig
from definable.skill.builtin.macos import MacOS


# ---------------------------------------------------------------------------
# MacOS.__init__ defaults
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMacOSDefaults:
  def test_default_bridge_host(self):
    skill = MacOS()
    assert skill._bridge_host == "127.0.0.1"

  def test_default_bridge_port(self):
    skill = MacOS()
    assert skill._bridge_port == 7777

  def test_default_bridge_token_is_none(self):
    skill = MacOS()
    assert skill._bridge_token is None

  def test_default_allowed_apps_is_none(self):
    skill = MacOS()
    assert skill._allowed_apps is None

  def test_default_blocked_apps_is_empty_set(self):
    skill = MacOS()
    assert skill._blocked_apps == set()

  def test_default_enable_applescript_is_true(self):
    skill = MacOS()
    assert skill._enable_applescript is True

  def test_default_enable_file_write_is_true(self):
    skill = MacOS()
    assert skill._enable_file_write is True

  def test_default_enable_input_is_true(self):
    skill = MacOS()
    assert skill._enable_input is True

  def test_client_is_none_before_any_call(self):
    """BridgeClient must not be created at init time."""
    skill = MacOS()
    assert skill._client is None

  def test_name_attribute(self):
    assert MacOS.name == "macos"


# ---------------------------------------------------------------------------
# _check_app_allowed
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCheckAppAllowed:
  def test_no_lists_all_apps_pass(self):
    """When neither allowed_apps nor blocked_apps is set, everything is allowed."""
    skill = MacOS()
    assert skill._check_app_allowed("Safari") is None
    assert skill._check_app_allowed("Finder") is None
    assert skill._check_app_allowed("anything") is None

  def test_allowed_list_blocks_unlisted_app(self):
    skill = MacOS(allowed_apps={"Safari", "TextEdit"})
    result = skill._check_app_allowed("Terminal")
    assert result is not None
    assert "Terminal" in result
    assert "allowed list" in result.lower()

  def test_allowed_list_permits_listed_app(self):
    skill = MacOS(allowed_apps={"Safari", "TextEdit"})
    assert skill._check_app_allowed("Safari") is None
    assert skill._check_app_allowed("TextEdit") is None

  def test_blocked_list_blocks_named_app(self):
    skill = MacOS(blocked_apps={"System Preferences", "Terminal"})
    result = skill._check_app_allowed("Terminal")
    assert result is not None
    assert "blocked" in result.lower()

  def test_blocked_list_allows_unlisted_app(self):
    skill = MacOS(blocked_apps={"Terminal"})
    assert skill._check_app_allowed("Safari") is None

  def test_blocked_wins_when_in_both_lists(self):
    """If app is in both allowed_apps and blocked_apps, blocked takes precedence (security-first)."""
    skill = MacOS(allowed_apps={"Safari"}, blocked_apps={"Safari"})
    # allowed_apps check passes (Safari is in the list), then blocked check fires -> blocked wins
    result = skill._check_app_allowed("Safari")
    assert result is not None
    assert "blocked" in result.lower()

  def test_empty_allowed_set_blocks_everything(self):
    skill = MacOS(allowed_apps=set())
    result = skill._check_app_allowed("Safari")
    assert result is not None

  def test_error_message_includes_app_name(self):
    skill = MacOS(allowed_apps={"TextEdit"})
    result = skill._check_app_allowed("Finder")
    assert result is not None
    assert "Finder" in result


# ---------------------------------------------------------------------------
# instructions property
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMacOSInstructions:
  def test_default_instructions_mention_bridge(self):
    skill = MacOS()
    instructions = skill.instructions
    assert "bridge" in instructions.lower() or "mac" in instructions.lower()

  def test_instructions_mention_screenshot(self):
    skill = MacOS()
    assert "screenshot" in skill.instructions.lower()

  def test_instructions_with_allowed_apps(self):
    skill = MacOS(allowed_apps={"Safari", "TextEdit"})
    assert "Safari" in skill.instructions or "TextEdit" in skill.instructions

  def test_instructions_read_only_mode(self):
    skill = MacOS(enable_input=False)
    assert "read-only" in skill.instructions.lower() or "input" in skill.instructions.lower()

  def test_instructions_mention_file_write_disabled(self):
    skill = MacOS(enable_file_write=False)
    assert "write" in skill.instructions.lower() or "file" in skill.instructions.lower()

  def test_instructions_mention_applescript_disabled(self):
    skill = MacOS(enable_applescript=False)
    assert "applescript" in skill.instructions.lower()

  def test_instructions_not_empty(self):
    skill = MacOS()
    assert len(skill.instructions) > 20

  def test_instructions_returns_string(self):
    assert isinstance(MacOS().instructions, str)


# ---------------------------------------------------------------------------
# tools count vs config flags
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMacOSToolsCount:
  """Tool count changes when feature flags are toggled."""

  def _tool_names(self, skill: MacOS) -> set[str]:
    return {t.name for t in skill.tools}

  def test_default_tools_non_empty(self):
    skill = MacOS()
    assert len(skill.tools) > 0

  def test_enable_input_false_removes_click_type_key_scroll_drag_set_clipboard(self):
    self._tool_names(MacOS(enable_input=True))
    names_off = self._tool_names(MacOS(enable_input=False))
    # Input tools removed
    assert "click" not in names_off
    assert "type_text" not in names_off
    assert "press_key" not in names_off
    assert "scroll" not in names_off
    assert "drag" not in names_off
    assert "set_clipboard" not in names_off
    assert "click_element" not in names_off
    assert "set_element_value" not in names_off
    # Non-input tools remain
    assert "screenshot" in names_off
    assert "list_running_apps" in names_off

  def test_enable_input_true_adds_click_tools(self):
    names = self._tool_names(MacOS(enable_input=True))
    assert "click" in names
    assert "type_text" in names
    assert "press_key" in names

  def test_enable_applescript_false_removes_run_applescript(self):
    names = self._tool_names(MacOS(enable_applescript=False))
    assert "run_applescript" not in names

  def test_enable_applescript_true_adds_run_applescript(self):
    names = self._tool_names(MacOS(enable_applescript=True))
    assert "run_applescript" in names

  def test_enable_file_write_false_removes_write_and_move(self):
    names = self._tool_names(MacOS(enable_file_write=False))
    assert "write_file" not in names
    assert "move_file" not in names

  def test_enable_file_write_true_adds_write_and_move(self):
    names = self._tool_names(MacOS(enable_file_write=True))
    assert "write_file" in names
    assert "move_file" in names

  def test_read_file_always_present(self):
    """read_file is always available regardless of enable_file_write."""
    assert "read_file" in self._tool_names(MacOS(enable_file_write=False))
    assert "read_file" in self._tool_names(MacOS(enable_file_write=True))

  def test_read_only_mode_has_fewer_tools_than_default(self):
    default_count = len(MacOS().tools)
    read_only_count = len(MacOS(enable_input=False, enable_file_write=False, enable_applescript=False).tools)
    assert read_only_count < default_count

  def test_all_tools_return_list(self):
    tools = MacOS().tools
    assert isinstance(tools, list)

  def test_tools_called_twice_returns_consistent_count(self):
    """tools property is deterministic -- same count on repeated calls."""
    skill = MacOS()
    assert len(skill.tools) == len(skill.tools)

  def test_always_present_screen_tools(self):
    """Screen-read tools present in every config."""
    for flag_combo in [
      {},
      {"enable_input": False},
      {"enable_file_write": False},
      {"enable_applescript": False},
      {"enable_input": False, "enable_file_write": False, "enable_applescript": False},
    ]:
      names = self._tool_names(MacOS(**flag_combo))  # type: ignore[arg-type]
      assert "screenshot" in names, f"screenshot missing in {flag_combo}"
      assert "read_screen" in names, f"read_screen missing in {flag_combo}"
      assert "find_text_on_screen" in names, f"find_text_on_screen missing in {flag_combo}"

  def test_always_present_app_tools(self):
    """App management tools always present."""
    names = self._tool_names(MacOS(enable_input=False))
    assert "list_running_apps" in names
    assert "open_app" in names
    assert "quit_app" in names
    assert "activate_app" in names
    assert "open_url" in names


# ---------------------------------------------------------------------------
# DesktopConfig
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDesktopConfig:
  def test_default_platform(self):
    cfg = DesktopConfig()
    assert cfg.platform == "desktop"

  def test_default_bridge_host(self):
    cfg = DesktopConfig()
    assert cfg.bridge_host == "127.0.0.1"

  def test_default_bridge_port(self):
    cfg = DesktopConfig()
    assert cfg.bridge_port == 7777

  def test_default_bridge_token_is_none(self):
    cfg = DesktopConfig()
    assert cfg.bridge_token is None

  def test_default_auto_screenshot_is_false(self):
    cfg = DesktopConfig()
    assert cfg.auto_screenshot is False

  def test_default_screenshot_on_error_is_true(self):
    cfg = DesktopConfig()
    assert cfg.screenshot_on_error is True

  def test_default_websocket_port_is_zero(self):
    cfg = DesktopConfig()
    assert cfg.websocket_port == 0

  def test_custom_bridge_port(self):
    cfg = DesktopConfig(bridge_port=9000)
    assert cfg.bridge_port == 9000

  def test_custom_token(self):
    cfg = DesktopConfig(bridge_token="my-secret-token")
    assert cfg.bridge_token == "my-secret-token"

  def test_frozen_dataclass_immutable(self):
    cfg = DesktopConfig()
    with pytest.raises((TypeError, AttributeError)):
      cfg.bridge_port = 9999  # type: ignore[misc]

  def test_with_updates_creates_new_instance(self):
    cfg = DesktopConfig()
    cfg2 = cfg.with_updates(bridge_port=9999)
    assert cfg.bridge_port == 7777
    assert cfg2.bridge_port == 9999  # type: ignore[attr-defined]

  def test_with_updates_preserves_other_fields(self):
    cfg = DesktopConfig(bridge_host="192.168.1.1")
    cfg2 = cfg.with_updates(bridge_port=9999)
    assert cfg2.bridge_host == "192.168.1.1"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# BridgeClient token file resolution (no network)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBridgeClientTokenResolution:
  def test_explicit_token_used_directly(self, tmp_path):
    from definable.agent.interface.desktop.bridge_client import BridgeClient

    client = BridgeClient(token="explicit-token")
    assert client._resolve_token() == "explicit-token"

  def test_token_read_from_file(self, tmp_path):
    from definable.agent.interface.desktop.bridge_client import _read_token_file

    token_file = tmp_path / "bridge-token"
    token_file.write_text("file-token-abc123")

    token = _read_token_file(token_file)
    assert token == "file-token-abc123"

  def test_missing_token_file_returns_none(self, tmp_path):
    from definable.agent.interface.desktop.bridge_client import _read_token_file

    result = _read_token_file(tmp_path / "nonexistent-token")
    assert result is None

  def test_empty_token_file_returns_none(self, tmp_path):
    from definable.agent.interface.desktop.bridge_client import _read_token_file

    token_file = tmp_path / "bridge-token"
    token_file.write_text("")

    assert _read_token_file(token_file) is None

  def test_whitespace_only_token_file_returns_none(self, tmp_path):
    from definable.agent.interface.desktop.bridge_client import _read_token_file

    token_file = tmp_path / "bridge-token"
    token_file.write_text("   \n  ")

    assert _read_token_file(token_file) is None

  def test_client_lazy_init_no_client_at_startup(self):
    from definable.agent.interface.desktop.bridge_client import BridgeClient

    client = BridgeClient()
    assert client._client is None

  def test_bridge_client_base_url_formation(self):
    from definable.agent.interface.desktop.bridge_client import BridgeClient

    client = BridgeClient(host="192.168.1.1", port=9000)
    assert client._base_url == "http://192.168.1.1:9000"
