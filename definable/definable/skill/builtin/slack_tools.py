"""Slack skill — send messages, search, manage channels and files."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from definable.skill.base import Skill
from definable.tool.decorator import tool


class SlackTools(Skill):
  """Interact with Slack: send messages, search, list channels, manage threads.

  Requires ``slack-sdk``: ``pip install slack-sdk``

  Args:
      token: Slack Bot token (xoxb-...). Falls back to SLACK_TOKEN env var.
      enable_send: Enable message sending. Default True.
      enable_search: Enable message search. Default False.
      enable_files: Enable file upload/download. Default False.
      enable_users: Enable user listing. Default False.

  Example::

      from definable.skill.builtin import SlackTools
      agent = Agent(model=model, skills=[SlackTools(token="xoxb-...")])
  """

  name = "slack_tools"
  instructions = (
    "You have access to Slack tools for messaging and collaboration. "
    "Use send_message to post to channels. Use list_channels to find channels. "
    "Use get_channel_history to read recent messages. Channel can be a name or ID."
  )

  def __init__(
    self,
    *,
    token: Optional[str] = None,
    enable_send: bool = True,
    enable_search: bool = False,
    enable_files: bool = False,
    enable_users: bool = False,
  ):
    super().__init__()
    self._token = token or os.getenv("SLACK_TOKEN")
    self._enable_send = enable_send
    self._enable_search = enable_search
    self._enable_files = enable_files
    self._enable_users = enable_users
    self._client: Any = None

  @property
  def client(self) -> Any:
    if self._client is not None:
      return self._client
    try:
      from slack_sdk import WebClient
    except ImportError:
      raise ImportError("`slack-sdk` not installed. Run: pip install slack-sdk")
    if not self._token:
      raise ValueError("Slack token required. Set token or SLACK_TOKEN env var.")
    self._client = WebClient(token=self._token)
    return self._client

  @property
  def tools(self) -> list:
    skill = self
    result: list = []

    @tool
    def list_channels(max_results: int = 50) -> str:
      """List Slack channels the bot has access to."""
      try:
        resp = skill.client.conversations_list(limit=max_results, types="public_channel,private_channel")
        channels = [{"id": c["id"], "name": c["name"], "topic": c.get("topic", {}).get("value", "")} for c in resp["channels"]]
        return json.dumps(channels, indent=2)
      except Exception as e:
        return json.dumps({"error": str(e)})

    @tool
    def get_channel_history(channel: str, limit: int = 20) -> str:
      """Get recent messages from a Slack channel."""
      try:
        resp = skill.client.conversations_history(channel=channel, limit=limit)
        messages = []
        for m in resp["messages"]:
          messages.append({"user": m.get("user", ""), "text": m.get("text", ""), "ts": m.get("ts", ""), "thread_ts": m.get("thread_ts")})
        return json.dumps(messages, indent=2)
      except Exception as e:
        return json.dumps({"error": str(e)})

    result.extend([list_channels, get_channel_history])

    if self._enable_send:

      @tool
      def send_message(channel: str, text: str) -> str:
        """Send a message to a Slack channel."""
        try:
          resp = skill.client.chat_postMessage(channel=channel, text=text, mrkdwn=True)
          return json.dumps({"ok": True, "channel": resp["channel"], "ts": resp["ts"]})
        except Exception as e:
          return json.dumps({"error": str(e)})

      @tool
      def reply_in_thread(channel: str, thread_ts: str, text: str) -> str:
        """Reply to a message thread in Slack."""
        try:
          resp = skill.client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=text, mrkdwn=True)
          return json.dumps({"ok": True, "channel": resp["channel"], "ts": resp["ts"]})
        except Exception as e:
          return json.dumps({"error": str(e)})

      result.extend([send_message, reply_in_thread])

    if self._enable_search:

      @tool
      def search_messages(query: str, max_results: int = 20) -> str:
        """Search Slack messages by keyword."""
        try:
          resp = skill.client.search_messages(query=query, count=max_results)
          messages = []
          for m in resp["messages"]["matches"]:
            messages.append({
              "text": m.get("text", ""),
              "user": m.get("username", ""),
              "channel": m.get("channel", {}).get("name", ""),
              "ts": m.get("ts", ""),
              "permalink": m.get("permalink", ""),
            })
          return json.dumps(messages, indent=2)
        except Exception as e:
          return json.dumps({"error": str(e)})

      @tool
      def get_thread(channel: str, thread_ts: str, limit: int = 50) -> str:
        """Get all messages in a thread."""
        try:
          resp = skill.client.conversations_replies(channel=channel, ts=thread_ts, limit=limit)
          messages = [{"user": m.get("user", ""), "text": m.get("text", ""), "ts": m.get("ts", "")} for m in resp["messages"]]
          return json.dumps(messages, indent=2)
        except Exception as e:
          return json.dumps({"error": str(e)})

      result.extend([search_messages, get_thread])

    if self._enable_files:

      @tool
      def upload_file(channel: str, content: str, filename: str, title: str = "") -> str:
        """Upload a text file to a Slack channel."""
        try:
          resp = skill.client.files_upload_v2(channel=channel, content=content, filename=filename, title=title or filename)
          return json.dumps({"ok": True, "file_id": resp.get("file", {}).get("id", "")})
        except Exception as e:
          return json.dumps({"error": str(e)})

      result.append(upload_file)

    if self._enable_users:

      @tool
      def list_users(limit: int = 50) -> str:
        """List users in the Slack workspace."""
        try:
          resp = skill.client.users_list(limit=limit)
          users = []
          for u in resp["members"]:
            if not u.get("deleted") and not u.get("is_bot"):
              display = u.get("profile", {}).get("display_name", "")
              users.append({"id": u["id"], "name": u.get("real_name", u.get("name", "")), "display_name": display})
          return json.dumps(users, indent=2)
        except Exception as e:
          return json.dumps({"error": str(e)})

      result.append(list_users)

    return result
