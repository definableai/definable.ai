"""Minimal Slack bot using Socket Mode.

Setup:
  1. Create a Slack app at https://api.slack.com/apps
  2. Enable Socket Mode in your app settings
  3. Add an App-Level Token with 'connections:write' scope
  4. Add Bot Token Scopes: chat:write, app_mentions:read, channels:history,
     im:history, files:read, files:write, users:read, reactions:write
  5. Install the app to your workspace
  6. Set environment variables:
     - SLACK_BOT_TOKEN=xoxb-...
     - SLACK_APP_TOKEN=xapp-...

Usage:
  pip install 'definable[slack]'
  python 01_slack_bot.py
"""

import asyncio
import os

from definable.agent import Agent
from definable.agent.interface.slack import SlackInterface


agent = Agent(
  model="openai/gpt-4o-mini",
  instructions="You are a helpful Slack assistant. Keep responses concise.",
)

interface = SlackInterface(
  agent=agent,
  bot_token=os.environ["SLACK_BOT_TOKEN"],
  app_token=os.environ["SLACK_APP_TOKEN"],
)

asyncio.run(interface.serve_forever())
