"""
Unit tests for BrowserConfig (SeleniumBase CDP version).

No browser launched, no seleniumbase import, no API calls.
Pure Python dataclass logic only.
"""

import pytest
from dataclasses import FrozenInstanceError

from definable.browser.config import BrowserConfig


@pytest.mark.unit
class TestBrowserConfigDefaults:
  def test_default_headless_is_false(self):
    assert BrowserConfig().headless is False

  def test_default_user_data_dir_is_none(self):
    assert BrowserConfig().user_data_dir is None

  def test_default_proxy_is_none(self):
    assert BrowserConfig().proxy is None

  def test_default_user_agent_is_none(self):
    assert BrowserConfig().user_agent is None

  def test_default_lang_is_en(self):
    assert BrowserConfig().lang == "en"

  def test_default_sandbox_is_true(self):
    assert BrowserConfig().sandbox is True

  def test_default_browser_executable_path_is_none(self):
    assert BrowserConfig().browser_executable_path is None

  def test_default_browser_args_is_none(self):
    assert BrowserConfig().browser_args is None

  def test_default_host_is_none(self):
    assert BrowserConfig().host is None

  def test_default_port_is_none(self):
    assert BrowserConfig().port is None

  def test_default_incognito_is_false(self):
    assert BrowserConfig().incognito is False

  def test_default_mobile_is_false(self):
    assert BrowserConfig().mobile is False

  def test_default_ad_block_is_false(self):
    assert BrowserConfig().ad_block is False

  def test_default_timeout_is_30(self):
    assert BrowserConfig().timeout == 30.0


@pytest.mark.unit
class TestBrowserConfigCustomValues:
  def test_headless_true(self):
    assert BrowserConfig(headless=True).headless is True

  def test_user_data_dir(self):
    cfg = BrowserConfig(user_data_dir="/tmp/profile")
    assert cfg.user_data_dir == "/tmp/profile"

  def test_proxy(self):
    cfg = BrowserConfig(proxy="user:pass@proxy.example.com:8080")
    assert cfg.proxy == "user:pass@proxy.example.com:8080"

  def test_proxy_simple_format(self):
    cfg = BrowserConfig(proxy="1.2.3.4:8080")
    assert cfg.proxy == "1.2.3.4:8080"

  def test_user_agent(self):
    ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X)"
    cfg = BrowserConfig(user_agent=ua)
    assert cfg.user_agent == ua

  def test_lang(self):
    cfg = BrowserConfig(lang="fr")
    assert cfg.lang == "fr"

  def test_sandbox_false(self):
    cfg = BrowserConfig(sandbox=False)
    assert cfg.sandbox is False

  def test_browser_executable_path(self):
    cfg = BrowserConfig(browser_executable_path="/usr/bin/chromium")
    assert cfg.browser_executable_path == "/usr/bin/chromium"

  def test_browser_args_tuple(self):
    args = ("--disable-gpu", "--no-first-run")
    cfg = BrowserConfig(browser_args=args)
    assert cfg.browser_args == args

  def test_host_and_port_for_cdp_attach(self):
    cfg = BrowserConfig(host="127.0.0.1", port=9222)
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 9222

  def test_incognito(self):
    assert BrowserConfig(incognito=True).incognito is True

  def test_mobile(self):
    assert BrowserConfig(mobile=True).mobile is True

  def test_ad_block(self):
    assert BrowserConfig(ad_block=True).ad_block is True

  def test_custom_timeout(self):
    cfg = BrowserConfig(timeout=60.0)
    assert cfg.timeout == 60.0


@pytest.mark.unit
class TestBrowserConfigImmutability:
  def test_is_frozen(self):
    cfg = BrowserConfig()
    with pytest.raises((FrozenInstanceError, AttributeError)):
      cfg.headless = True  # type: ignore[misc]

  def test_equality(self):
    a = BrowserConfig(headless=True, lang="fr")
    b = BrowserConfig(headless=True, lang="fr")
    assert a == b

  def test_inequality(self):
    a = BrowserConfig(headless=True)
    b = BrowserConfig(headless=False)
    assert a != b

  def test_hashable(self):
    cfg = BrowserConfig(headless=True)
    assert hash(cfg) is not None
    s = {cfg}
    assert len(s) == 1


@pytest.mark.unit
class TestBrowserConfigConnectionModes:
  def test_mode_a_fresh_chrome(self):
    """Mode A: headless=False, no user_data_dir, no host/port."""
    cfg = BrowserConfig(headless=False)
    assert cfg.host is None
    assert cfg.port is None
    assert cfg.user_data_dir is None

  def test_mode_b_persistent_profile(self):
    """Mode B: user_data_dir set, no host/port."""
    cfg = BrowserConfig(user_data_dir="/tmp/profile")
    assert cfg.user_data_dir == "/tmp/profile"
    assert cfg.host is None
    assert cfg.port is None

  def test_mode_c_cdp_attach(self):
    """Mode C: host + port set."""
    cfg = BrowserConfig(host="127.0.0.1", port=9222)
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 9222
    assert cfg.user_data_dir is None
