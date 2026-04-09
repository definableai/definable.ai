from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from definable.agent import Agent
from definable.agent.interface import (
  DiscordInterface,
  EmailInterface,
  SlackInterface,
  SQLiteIdentityResolver,
  TelegramInterface,
  WebSocketInterface,
  WhatsAppInterface,
)
from definable.agent.interface.gateway import InterfaceGateway
from definable.agent.interface.session import SessionManager
from definable.agent.testing import MockModel


async def _resolve_identity(db_path: Path) -> dict[str, object]:
  resolver = SQLiteIdentityResolver(db_path=str(db_path))
  async with resolver:
    await resolver.link("telegram", "42", "user-1", username="alice")
    await resolver.link("slack", "U123", "user-1", username="alice-work")

    resolved = await resolver.resolve("telegram", "42")
    identities = await resolver.get_identities("user-1")

  return {
    "resolved_user": resolved,
    "linked_platforms": sorted(identity.platform for identity in identities),
  }


def main() -> dict[str, object]:
  gateway = InterfaceGateway(shared_sessions=True)

  telegram = TelegramInterface(bot_token="123:test", commands={"start": "Start the bot"})
  discord = DiscordInterface(bot_token="discord-token", command_prefix="!")
  slack = SlackInterface(
    bot_token="xoxb-test",
    app_token="xapp-test",
    slash_commands={"/ask": "Ask the assistant"},
  )

  agent = Agent(
    model=MockModel(responses=["ready"]),
    gateway=gateway,
    interfaces=[telegram, discord, slack],
  )

  email = EmailInterface(
    imap_host="imap.example.com",
    smtp_host="smtp.example.com",
    email_address="agent@example.com",
    email_password="secret",
  )
  websocket = WebSocketInterface(path="/agent/ws", heartbeat_interval=10.0)
  whatsapp_twilio = WhatsAppInterface(
    provider="twilio",
    account_sid="AC123",
    auth_token="token",
    from_number="whatsapp:+14155550123",
  )
  whatsapp_baileys = WhatsAppInterface(provider="baileys", auth_dir="./tmp-whatsapp-auth")

  sessions = SessionManager(session_ttl_seconds=30)
  first_session = sessions.get_or_create("telegram", "u-1", "chat-1")
  second_session = sessions.get_or_create("telegram", "u-1", "chat-1")

  with TemporaryDirectory() as tmp_dir:
    identity = asyncio.run(_resolve_identity(Path(tmp_dir) / "identity.db"))

  summary = {
    "attached_platforms": [interface.config.platform for interface in agent.interfaces],
    "shared_sessions": gateway.interfaces[0].session_manager is gateway.interfaces[1].session_manager,
    "session_reused": first_session.session_id == second_session.session_id,
    "email_address": email.config.email_address,
    "websocket_path": websocket.config.path,
    "whatsapp_server_modes": [whatsapp_twilio.needs_server(), whatsapp_baileys.needs_server()],
    **identity,
  }

  assert summary["attached_platforms"] == ["telegram", "discord", "slack"]
  assert summary["shared_sessions"] is True
  assert summary["session_reused"] is True
  assert summary["email_address"] == "agent@example.com"
  assert summary["websocket_path"] == "/agent/ws"
  assert summary["whatsapp_server_modes"] == [True, False]
  assert summary["resolved_user"] == "user-1"
  assert summary["linked_platforms"] == ["slack", "telegram"]

  return summary


if __name__ == "__main__":
  print(main())
