"""Telegram output skill for agent-controlled inline keyboards."""

import json
from typing import List

from definable.skill.base import Skill
from definable.tool.decorator import tool


@tool
def telegram_reply_buttons(buttons: str) -> str:
  """Queue Telegram inline buttons for the next text response.

  Args:
    buttons: JSON string representing rows of buttons.
      Shape: [[{"text":"A","callback_data":"a"}], [{"text":"Docs","url":"https://..."}]]

  Returns:
    Human-readable status message for the agent.
  """
  from definable.agent.interface.telegram.interface import _pending_buttons_var
  from definable.agent.interface.telegram.keyboards import InlineButton

  try:
    parsed = json.loads(buttons)
  except Exception:
    return "Failed to parse buttons JSON. Use a JSON array of rows."

  if not isinstance(parsed, list):
    return "Invalid buttons format. Expected a JSON array of rows."

  rows: List[List[InlineButton]] = []
  for row in parsed:
    if not isinstance(row, list):
      return "Invalid row format. Each row must be a JSON array of buttons."
    btn_row: List[InlineButton] = []
    for item in row:
      if not isinstance(item, dict):
        return "Invalid button format. Each button must be a JSON object."
      text = item.get("text")
      callback_data = item.get("callback_data")
      url = item.get("url")
      if not isinstance(text, str) or not text.strip():
        return "Each button requires a non-empty 'text' field."
      try:
        btn_row.append(InlineButton(text=text, callback_data=callback_data, url=url))
      except Exception as e:
        return f"Invalid button: {e}"
    rows.append(btn_row)

  _pending_buttons_var.set(rows)
  return f"Queued {sum(len(r) for r in rows)} button(s) for the next Telegram reply."


class TelegramOutputSkill(Skill):
  """Skill that lets agents prepare Telegram-specific UI output."""

  def __init__(self, include_buttons: bool = True, custom_instructions: str = "") -> None:
    instructions = (
      "When the user needs a choice, use telegram_reply_buttons to prepare inline buttons before your next textual response.\n"
      "Keep button labels short and use callback_data values that are concise and stable."
    )
    if custom_instructions:
      instructions = f"{instructions}\n{custom_instructions}"

    tools = [telegram_reply_buttons] if include_buttons else []
    super().__init__(name="telegram_output", instructions=instructions, tools=tools)
