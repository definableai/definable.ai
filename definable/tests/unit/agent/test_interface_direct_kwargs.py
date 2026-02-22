"""Tests for direct-kwargs interface construction (config class elimination from public API)."""

import warnings

import pytest

from definable.agent.interface.cli.config import CLIConfig
from definable.agent.interface.desktop.config import DesktopConfig
from definable.agent.interface.discord.config import DiscordConfig
from definable.agent.interface.errors import InterfaceError
from definable.agent.interface.telegram.config import TelegramConfig


# ---------------------------------------------------------------------------
# Telegram — direct kwargs
# ---------------------------------------------------------------------------


class TestTelegramDirectKwargs:
  """TelegramInterface accepts all config fields as direct kwargs."""

  def test_direct_kwargs_basic(self):
    from definable.agent.interface.telegram import TelegramInterface

    iface = TelegramInterface(bot_token="test-token-123")
    assert iface.config.platform == "telegram"
    assert iface._tg_config.bot_token == "test-token-123"
    assert iface._tg_config.mode == "polling"

  def test_direct_kwargs_all_telegram_fields(self):
    from definable.agent.interface.telegram import TelegramInterface

    iface = TelegramInterface(
      bot_token="tok",
      mode="polling",
      webhook_path="/hook",
      webhook_port=9999,
      webhook_secret="s3cr3t",
      allowed_user_ids=[1, 2, 3],
      allowed_chat_ids=[100],
      parse_mode="MarkdownV2",
      polling_interval=1.0,
      polling_timeout=60,
      connect_timeout=5.0,
      request_timeout=30.0,
    )
    assert iface._tg_config.webhook_path == "/hook"
    assert iface._tg_config.webhook_port == 9999
    assert iface._tg_config.webhook_secret == "s3cr3t"
    assert iface._tg_config.allowed_user_ids == [1, 2, 3]
    assert iface._tg_config.allowed_chat_ids == [100]
    assert iface._tg_config.parse_mode == "MarkdownV2"
    assert iface._tg_config.polling_interval == 1.0
    assert iface._tg_config.polling_timeout == 60
    assert iface._tg_config.connect_timeout == 5.0
    assert iface._tg_config.request_timeout == 30.0

  def test_base_config_kwargs_propagate(self):
    from definable.agent.interface.telegram import TelegramInterface

    iface = TelegramInterface(
      bot_token="tok",
      max_session_history=100,
      session_ttl_seconds=7200,
      max_concurrent_requests=5,
      error_message="Oops!",
      typing_indicator=False,
      max_message_length=2000,
      rate_limit_messages_per_minute=60,
    )
    assert iface.config.max_session_history == 100
    assert iface.config.session_ttl_seconds == 7200
    assert iface.config.max_concurrent_requests == 5
    assert iface.config.error_message == "Oops!"
    assert iface.config.typing_indicator is False
    assert iface.config.max_message_length == 2000
    assert iface.config.rate_limit_messages_per_minute == 60

  def test_backwards_compat_config_kwarg(self):
    from definable.agent.interface.telegram import TelegramInterface

    cfg = TelegramConfig(bot_token="legacy-token")
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      iface = TelegramInterface(config=cfg)
      assert len(w) == 1
      assert issubclass(w[0].category, DeprecationWarning)
      assert "config=" in str(w[0].message)
    assert iface._tg_config.bot_token == "legacy-token"

  def test_missing_bot_token_raises(self):
    from definable.agent.interface.telegram import TelegramInterface

    with pytest.raises(InterfaceError, match="bot_token is required"):
      TelegramInterface()

  def test_webhook_mode_without_url_raises(self):
    from definable.agent.interface.telegram import TelegramInterface

    with pytest.raises(InterfaceError, match="webhook_url is required"):
      TelegramInterface(bot_token="tok", mode="webhook")

  def test_webhook_mode_with_url_succeeds(self):
    from definable.agent.interface.telegram import TelegramInterface

    iface = TelegramInterface(
      bot_token="tok",
      mode="webhook",
      webhook_url="https://example.com",
    )
    assert iface._tg_config.mode == "webhook"
    assert iface._tg_config.webhook_url == "https://example.com"


# ---------------------------------------------------------------------------
# Discord — direct kwargs
# ---------------------------------------------------------------------------


class TestDiscordDirectKwargs:
  """DiscordInterface accepts all config fields as direct kwargs."""

  def test_direct_kwargs_basic(self):
    from definable.agent.interface.discord import DiscordInterface

    iface = DiscordInterface(bot_token="discord-tok")
    assert iface.config.platform == "discord"
    assert iface._dc_config.bot_token == "discord-tok"
    assert iface._dc_config.max_message_length == 2000

  def test_all_discord_fields(self):
    from definable.agent.interface.discord import DiscordInterface

    iface = DiscordInterface(
      bot_token="tok",
      intents_message_content=False,
      allowed_guild_ids=[111],
      allowed_channel_ids=[222],
      respond_to_bots=True,
      command_prefix="!",
      connect_timeout=15.0,
    )
    assert iface._dc_config.intents_message_content is False
    assert iface._dc_config.allowed_guild_ids == [111]
    assert iface._dc_config.allowed_channel_ids == [222]
    assert iface._dc_config.respond_to_bots is True
    assert iface._dc_config.command_prefix == "!"
    assert iface._dc_config.connect_timeout == 15.0

  def test_backwards_compat_config_kwarg(self):
    from definable.agent.interface.discord import DiscordInterface

    cfg = DiscordConfig(bot_token="legacy")
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      iface = DiscordInterface(config=cfg)
      assert len(w) == 1
      assert issubclass(w[0].category, DeprecationWarning)
    assert iface._dc_config.bot_token == "legacy"

  def test_missing_bot_token_raises(self):
    from definable.agent.interface.discord import DiscordInterface

    with pytest.raises(InterfaceError, match="bot_token is required"):
      DiscordInterface()


# ---------------------------------------------------------------------------
# Desktop — direct kwargs
# ---------------------------------------------------------------------------


class TestDesktopDirectKwargs:
  """DesktopInterface accepts all config fields as direct kwargs."""

  def test_direct_kwargs_basic(self):
    from definable.agent.interface.desktop import DesktopInterface

    iface = DesktopInterface(websocket_port=9999)
    assert iface.config.platform == "desktop"
    assert iface._config.websocket_port == 9999
    assert iface._config.bridge_host == "127.0.0.1"

  def test_all_desktop_fields(self):
    from definable.agent.interface.desktop import DesktopInterface

    iface = DesktopInterface(
      bridge_host="0.0.0.0",
      bridge_port=8888,
      bridge_token="tok123",
      auto_screenshot=True,
      screenshot_on_error=False,
      websocket_port=7777,
    )
    assert iface._config.bridge_host == "0.0.0.0"
    assert iface._config.bridge_port == 8888
    assert iface._config.bridge_token == "tok123"
    assert iface._config.auto_screenshot is True
    assert iface._config.screenshot_on_error is False
    assert iface._config.websocket_port == 7777

  def test_backwards_compat_config_kwarg(self):
    from definable.agent.interface.desktop import DesktopInterface

    cfg = DesktopConfig(websocket_port=5555)
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      iface = DesktopInterface(config=cfg)
      assert len(w) == 1
      assert issubclass(w[0].category, DeprecationWarning)
    assert iface._config.websocket_port == 5555

  def test_default_max_message_length(self):
    from definable.agent.interface.desktop import DesktopInterface

    iface = DesktopInterface()
    assert iface.config.max_message_length == 100_000


# ---------------------------------------------------------------------------
# CLI — direct kwargs
# ---------------------------------------------------------------------------


class TestCLIDirectKwargs:
  """CLIInterface accepts all config fields as direct kwargs."""

  def test_direct_kwargs_basic(self):
    from definable.agent.interface.cli import CLIInterface

    iface = CLIInterface(show_banner=False, markdown_output=False)
    assert iface._cli_config.show_banner is False
    assert iface._cli_config.markdown_output is False
    assert iface.config.platform == "cli"

  def test_all_cli_fields(self):
    from definable.agent.interface.cli import CLIInterface

    iface = CLIInterface(
      prompt="$ ",
      show_banner=False,
      show_metrics=False,
      show_tool_args=False,
      show_tool_results=False,
      show_thinking=False,
      show_timestamps=True,
      max_content_display=1000,
      markdown_output=False,
      command_prefix="!",
      enable_completions=False,
      user_id="test-user",
    )
    assert iface._cli_config.prompt == "$ "
    assert iface._cli_config.show_metrics is False
    assert iface._cli_config.show_tool_args is False
    assert iface._cli_config.show_tool_results is False
    assert iface._cli_config.show_thinking is False
    assert iface._cli_config.show_timestamps is True
    assert iface._cli_config.max_content_display == 1000
    assert iface._cli_config.command_prefix == "!"
    assert iface._cli_config.enable_completions is False
    assert iface._cli_config.user_id == "test-user"

  def test_backwards_compat_config_kwarg(self):
    from definable.agent.interface.cli import CLIInterface

    cfg = CLIConfig(prompt=">> ", show_banner=False)
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      iface = CLIInterface(config=cfg)
      assert len(w) == 1
      assert issubclass(w[0].category, DeprecationWarning)
    assert iface._cli_config.prompt == ">> "
    assert iface._cli_config.show_banner is False

  def test_default_values_match_cli_config(self):
    from definable.agent.interface.cli import CLIInterface

    iface = CLIInterface()
    default_cfg = CLIConfig()
    assert iface._cli_config.prompt == default_cfg.prompt
    assert iface._cli_config.show_banner == default_cfg.show_banner
    assert iface._cli_config.max_concurrent_requests == default_cfg.max_concurrent_requests
    assert iface._cli_config.max_message_length == default_cfg.max_message_length

  def test_base_config_kwargs(self):
    from definable.agent.interface.cli import CLIInterface

    iface = CLIInterface(
      session_ttl_seconds=9999,
      max_session_history=200,
    )
    assert iface.config.session_ttl_seconds == 9999
    assert iface.config.max_session_history == 200
