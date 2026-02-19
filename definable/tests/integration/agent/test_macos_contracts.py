"""
Contract tests: MacOS skill ABC compliance.

No bridge running, no API calls, no real macOS interaction.

Covers:
  - MacOS is an instance of Skill
  - MacOS().tools returns a non-empty list
  - All tools are Function instances
  - All tools have a non-empty name and description
  - Tool names are unique within a config
  - enable_input=False: click, type_text, press_key, scroll, drag, set_clipboard, click_element, set_element_value absent
  - enable_applescript=False: run_applescript absent
  - enable_file_write=False: write_file and move_file absent
  - MacOS is importable from definable.skill
  - DesktopInterface is a subclass of BaseInterface
  - DesktopConfig is a subclass of InterfaceConfig
  - BridgeClient public API has expected methods
"""

from __future__ import annotations

import inspect

import pytest

from definable.skill.base import Skill
from definable.skill.builtin.macos import MacOS
from definable.tool.function import Function


# ---------------------------------------------------------------------------
# MacOS is a Skill subclass
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestMacOSIsSkill:
  def test_macos_is_subclass_of_skill(self):
    assert issubclass(MacOS, Skill)

  def test_macos_instance_is_skill(self):
    assert isinstance(MacOS(), Skill)

  def test_macos_has_name_attribute(self):
    skill = MacOS()
    assert isinstance(skill.name, str) and skill.name

  def test_macos_has_instructions_property(self):
    skill = MacOS()
    assert isinstance(skill.instructions, str)

  def test_macos_has_tools_property(self):
    skill = MacOS()
    assert hasattr(skill, "tools")

  def test_macos_has_teardown_method(self):
    assert callable(MacOS().teardown)

  def test_macos_importable_from_skills_package(self):
    from definable.skill import MacOS as MacOSFromPackage

    assert MacOSFromPackage is MacOS

  def test_macos_repr_shows_tool_count(self):
    skill = MacOS()
    r = repr(skill)
    assert "macos" in r


# ---------------------------------------------------------------------------
# All tools are Function instances
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestMacOSToolsAreFunction:
  def test_all_tools_are_function_instances(self):
    for tool in MacOS().tools:
      assert isinstance(tool, Function), f"Tool {tool!r} is not a Function"

  def test_all_tools_have_name(self):
    for tool in MacOS().tools:
      assert isinstance(tool.name, str) and tool.name, f"Tool has empty name: {tool!r}"

  def test_all_tools_have_description(self):
    for tool in MacOS().tools:
      desc = tool.description
      assert isinstance(desc, str) and desc, f"Tool '{tool.name}' has empty description"

  def test_all_tool_names_are_unique(self):
    tools = MacOS().tools
    names = [t.name for t in tools]
    assert len(names) == len(set(names)), f"Duplicate tool names: {[n for n in names if names.count(n) > 1]}"

  def test_tools_list_is_non_empty(self):
    assert len(MacOS().tools) > 0

  def test_read_only_tools_list_is_non_empty(self):
    skill = MacOS(enable_input=False, enable_file_write=False, enable_applescript=False)
    assert len(skill.tools) > 0

  def test_tools_returns_list_not_generator(self):
    tools = MacOS().tools
    assert isinstance(tools, list)


# ---------------------------------------------------------------------------
# Input tools absent when enable_input=False
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestMacOSInputGating:
  """When enable_input=False, all input-simulation tools must be absent."""

  def _names(self, **kwargs: object) -> set[str]:
    return {t.name for t in MacOS(**kwargs).tools}  # type: ignore[arg-type]

  _INPUT_TOOLS = {"click", "type_text", "press_key", "scroll", "drag", "set_clipboard", "click_element", "set_element_value"}
  _NON_INPUT_TOOLS = {"screenshot", "read_screen", "find_text_on_screen", "list_running_apps", "read_file"}

  def test_click_absent_when_input_disabled(self):
    assert "click" not in self._names(enable_input=False)

  def test_type_text_absent_when_input_disabled(self):
    assert "type_text" not in self._names(enable_input=False)

  def test_press_key_absent_when_input_disabled(self):
    assert "press_key" not in self._names(enable_input=False)

  def test_scroll_absent_when_input_disabled(self):
    assert "scroll" not in self._names(enable_input=False)

  def test_drag_absent_when_input_disabled(self):
    assert "drag" not in self._names(enable_input=False)

  def test_set_clipboard_absent_when_input_disabled(self):
    assert "set_clipboard" not in self._names(enable_input=False)

  def test_click_element_absent_when_input_disabled(self):
    assert "click_element" not in self._names(enable_input=False)

  def test_set_element_value_absent_when_input_disabled(self):
    assert "set_element_value" not in self._names(enable_input=False)

  def test_screenshot_present_when_input_disabled(self):
    """Screen-read tools are always available."""
    assert "screenshot" in self._names(enable_input=False)

  def test_read_file_present_when_input_disabled(self):
    assert "read_file" in self._names(enable_input=False)

  def test_list_running_apps_present_when_input_disabled(self):
    assert "list_running_apps" in self._names(enable_input=False)

  def test_all_input_tools_present_by_default(self):
    names = self._names()
    for tool_name in self._INPUT_TOOLS:
      assert tool_name in names, f"Input tool '{tool_name}' missing from default config"


# ---------------------------------------------------------------------------
# AppleScript gating
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestMacOSAppleScriptGating:
  def test_run_applescript_absent_when_disabled(self):
    names = {t.name for t in MacOS(enable_applescript=False).tools}
    assert "run_applescript" not in names

  def test_run_applescript_present_by_default(self):
    names = {t.name for t in MacOS().tools}
    assert "run_applescript" in names


# ---------------------------------------------------------------------------
# File write gating
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestMacOSFileWriteGating:
  def test_write_file_absent_when_disabled(self):
    names = {t.name for t in MacOS(enable_file_write=False).tools}
    assert "write_file" not in names

  def test_move_file_absent_when_disabled(self):
    names = {t.name for t in MacOS(enable_file_write=False).tools}
    assert "move_file" not in names

  def test_read_file_always_present(self):
    for flag in [True, False]:
      names = {t.name for t in MacOS(enable_file_write=flag).tools}
      assert "read_file" in names

  def test_list_files_always_present(self):
    names = {t.name for t in MacOS(enable_file_write=False).tools}
    assert "list_files" in names

  def test_write_file_present_by_default(self):
    names = {t.name for t in MacOS().tools}
    assert "write_file" in names

  def test_move_file_present_by_default(self):
    names = {t.name for t in MacOS().tools}
    assert "move_file" in names


# ---------------------------------------------------------------------------
# DesktopInterface is a BaseInterface subclass
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestDesktopInterfaceContract:
  def test_desktop_interface_is_subclass_of_base_interface(self):
    from definable.agent.interface.base import BaseInterface
    from definable.agent.interface.desktop.interface import DesktopInterface

    assert issubclass(DesktopInterface, BaseInterface)

  def test_desktop_interface_importable_from_package(self):
    from definable.agent.interface import DesktopInterface

    assert DesktopInterface is not None

  def test_desktop_config_importable_from_package(self):
    from definable.agent.interface import DesktopConfig

    assert DesktopConfig is not None

  def test_desktop_config_is_subclass_of_interface_config(self):
    from definable.agent.interface.config import InterfaceConfig
    from definable.agent.interface.desktop.config import DesktopConfig

    assert issubclass(DesktopConfig, InterfaceConfig)

  def test_desktop_interface_has_required_abstract_methods(self):
    from definable.agent.interface.desktop.interface import DesktopInterface

    assert hasattr(DesktopInterface, "_start_receiver")
    assert hasattr(DesktopInterface, "_stop_receiver")
    assert hasattr(DesktopInterface, "_convert_inbound")
    assert hasattr(DesktopInterface, "_send_response")

  def test_desktop_interface_is_not_abstract(self):
    """DesktopInterface must be concrete (all abstract methods implemented)."""
    from definable.agent.interface.desktop.config import DesktopConfig
    from definable.agent.interface.desktop.interface import DesktopInterface

    cfg = DesktopConfig()
    iface = DesktopInterface(config=cfg)
    assert iface is not None


# ---------------------------------------------------------------------------
# BridgeClient public API
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestBridgeClientAPI:
  """BridgeClient exposes all expected public methods."""

  _EXPECTED_METHODS = [
    "health",
    "capture_screen",
    "ocr_screen",
    "find_text_on_screen",
    "click",
    "type_text",
    "press_key",
    "mouse_move",
    "scroll",
    "drag",
    "list_apps",
    "open_app",
    "quit_app",
    "activate_app",
    "open_url",
    "open_file",
    "list_windows",
    "focus_window",
    "resize_window",
    "close_window",
    "get_focused_element",
    "get_ui_tree",
    "find_ui_element",
    "click_ui_element",
    "set_ui_value",
    "run_applescript",
    "list_files",
    "read_file",
    "write_file",
    "move_file",
    "delete_file",
    "file_info",
    "get_clipboard",
    "set_clipboard",
    "system_info",
    "get_volume",
    "set_volume",
    "get_battery",
    "send_notification",
    "set_dark_mode",
    "lock_screen",
    "close",
  ]

  def test_all_expected_methods_exist(self):
    from definable.agent.interface.desktop.bridge_client import BridgeClient

    client = BridgeClient()
    for method_name in self._EXPECTED_METHODS:
      assert hasattr(client, method_name), f"BridgeClient missing method: {method_name}"

  def test_all_public_methods_are_coroutines(self):
    from definable.agent.interface.desktop.bridge_client import BridgeClient

    for method_name in self._EXPECTED_METHODS:
      method = getattr(BridgeClient, method_name)
      assert inspect.iscoroutinefunction(method), f"BridgeClient.{method_name} is not async"

  def test_bridge_client_is_async_context_manager(self):
    from definable.agent.interface.desktop.bridge_client import BridgeClient

    client = BridgeClient()
    assert hasattr(client, "__aenter__")
    assert hasattr(client, "__aexit__")
