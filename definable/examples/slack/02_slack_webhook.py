"""Slack bot using HTTP Events API with AgentRuntime.

This mode is for production deployments where the bot is behind a
public URL. Slack sends events via HTTP POST to your server.

Setup:
  1. Create a Slack app at https://api.slack.com/apps
  2. Enable Events API (NOT Socket Mode)
  3. Set Request URL to: https://your-domain.com/slack/events
  4. Add Bot Token Scopes: chat:write, app_mentions:read, channels:history,
     im:history, files:read, files:write, users:read, reactions:write
  5. Subscribe to events: message.im, app_mention
  6. Install the app to your workspace
  7. Set environment variables:
     - SLACK_BOT_TOKEN=xoxb-...
     - SLACK_SIGNING_SECRET=... (from app's Basic Information page)

Usage:
  pip install 'definable[slack,runtime]'
  python 02_slack_webhook.py
"""

import asyncio
import os

from definable.agent import Agent
from definable.agent.interface.slack import SlackInterface
from definable.agent.runtime import AgentRuntime


agent = Agent(
  model="openai/gpt-4o-mini",
  instructions="You are a helpful Slack assistant.",
)

interface = SlackInterface(
  agent=agent,
  bot_token=os.environ["SLACK_BOT_TOKEN"],
  signing_secret=os.environ["SLACK_SIGNING_SECRET"],
  mode="http",
)

runtime = AgentRuntime(agent, interfaces=[interface], port=3000)
asyncio.run(runtime.start())
