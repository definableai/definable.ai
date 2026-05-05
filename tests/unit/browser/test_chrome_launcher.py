"""Tests for Chrome launcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from definable.browser.chrome_launcher import (
  RunningChrome,
  find_chrome_executable,
  stop_chrome,
)


class TestFindChromeExecutable:
  @pytest.mark.asyncio
  @patch("definable.browser.chrome_launcher.platform")
  @patch("os.path.isfile", return_value=True)
  async def test_macos_finds_chrome(self, mock_isfile, mock_platform):
    mock_platform.system.return_value = "Darwin"
    result = await find_chrome_executable()
    assert "Chrome" in result or "chrome" in result.lower()

  @pytest.mark.asyncio
  @patch("definable.browser.chrome_launcher.platform")
  @patch("os.path.isfile", return_value=False)
  @patch("shutil.which", return_value=None)
  async def test_not_found_raises(self, mock_which, mock_isfile, mock_platform):
    mock_platform.system.return_value = "Darwin"
    with pytest.raises(FileNotFoundError, match="No Chrome"):
      await find_chrome_executable()

  @pytest.mark.asyncio
  @patch("definable.browser.chrome_launcher.platform")
  @patch("shutil.which", return_value="/usr/bin/google-chrome")
  async def test_linux_which_fallback(self, mock_which, mock_platform):
    mock_platform.system.return_value = "Linux"
    result = await find_chrome_executable()
    assert result == "/usr/bin/google-chrome"


class TestRunningChrome:
  def test_dataclass(self):
    proc = MagicMock()
    rc = RunningChrome(pid=123, process=proc, cdp_port=9222, cdp_url="http://127.0.0.1:9222", user_data_dir="/tmp/test")
    assert rc.pid == 123
    assert rc.cdp_port == 9222
    assert rc.cdp_url == "http://127.0.0.1:9222"


class TestStopChrome:
  @pytest.mark.asyncio
  async def test_already_exited(self):
    proc = MagicMock()
    proc.poll.return_value = 0  # already exited
    rc = RunningChrome(pid=123, process=proc, cdp_port=9222, cdp_url="http://127.0.0.1:9222", user_data_dir="/tmp/test")
    await stop_chrome(rc)
    proc.terminate.assert_not_called()

  @pytest.mark.asyncio
  async def test_terminates_gracefully(self):
    proc = MagicMock()
    poll_count = 0

    def poll_side_effect():
      nonlocal poll_count
      poll_count += 1
      if poll_count >= 2:
        return 0
      return None

    proc.poll.side_effect = poll_side_effect
    rc = RunningChrome(pid=123, process=proc, cdp_port=9222, cdp_url="http://127.0.0.1:9222", user_data_dir="/tmp/test")
    await stop_chrome(rc, timeout=1.0)
    proc.terminate.assert_called_once()

  @pytest.mark.asyncio
  async def test_force_kills_on_timeout(self):
    proc = MagicMock()
    proc.poll.return_value = None  # never exits
    rc = RunningChrome(pid=123, process=proc, cdp_port=9222, cdp_url="http://127.0.0.1:9222", user_data_dir="/tmp/test")
    await stop_chrome(rc, timeout=0.1)
    proc.kill.assert_called_once()
