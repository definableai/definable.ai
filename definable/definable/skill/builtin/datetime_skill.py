"""DateTime skill — current time, date formatting, and timezone awareness.

Gives the agent accurate awareness of the current date and time,
eliminating the common LLM problem of hallucinating dates.

Example:
    from definable.skill.builtin import DateTime

    agent = Agent(
        model=model,
        skills=[DateTime()],
    )
    output = agent.run("What day of the week is it?")
"""

from datetime import datetime, timezone
from typing import Optional

from definable.skill.base import Skill
from definable.tool.decorator import tool


@tool
def get_current_time(timezone_name: Optional[str] = None) -> str:
  """Get the current date and time.

  Args:
    timezone_name: Optional timezone offset like "+05:30", "-08:00",
        or "UTC". Defaults to UTC if not specified.

  Returns:
    Current date and time as a formatted string including
    day of week, date, time, and timezone.
  """
  import re
  from datetime import timedelta

  now = datetime.now(timezone.utc)

  if timezone_name and timezone_name.upper() != "UTC":
    # Parse offset like "+05:30" or "-08:00"
    match = re.match(r"^([+-])(\d{1,2}):?(\d{2})?$", timezone_name.strip())
    if match:
      sign = 1 if match.group(1) == "+" else -1
      hours = int(match.group(2))
      minutes = int(match.group(3) or 0)
      offset = timedelta(hours=hours, minutes=minutes) * sign
      tz = timezone(offset)
      now = now.astimezone(tz)
      tz_label = f"UTC{timezone_name}"
    else:
      tz_label = "UTC"
  else:
    tz_label = "UTC"

  return (
    f"{now.strftime('%A, %B %d, %Y')} | "
    f"{now.strftime('%I:%M:%S %p')} {tz_label} | "
    f"Week {now.isocalendar()[1]}, Day {now.timetuple().tm_yday} of {now.year}"
  )


@tool
def date_difference(date1: str, date2: str) -> str:
  """Calculate the difference between two dates.

  Args:
    date1: First date in YYYY-MM-DD format.
    date2: Second date in YYYY-MM-DD format.

  Returns:
    The difference in days, weeks, months, and years.
  """
  try:
    d1 = datetime.strptime(date1.strip(), "%Y-%m-%d")
    d2 = datetime.strptime(date2.strip(), "%Y-%m-%d")
  except ValueError:
    return "Error: dates must be in YYYY-MM-DD format."

  delta = abs((d2 - d1).days)
  weeks = delta // 7
  months_approx = round(delta / 30.44, 1)
  years_approx = round(delta / 365.25, 2)

  return (
    f"{delta} days ({weeks} weeks, ~{months_approx} months, ~{years_approx} years). From {d1.strftime('%B %d, %Y')} to {d2.strftime('%B %d, %Y')}."
  )


class DateTime(Skill):
  """Skill for accurate date/time awareness and calculations.

  Adds tools for getting the current time and computing date differences.
  Prevents the common LLM problem of hallucinating or guessing dates.

  Example:
    agent = Agent(model=model, skills=[DateTime()])
    output = agent.run("How many days until December 25, 2025?")
  """

  name = "datetime"
  instructions = (
    "You have access to date and time tools. Always use get_current_time "
    "when asked about the current date, time, or day of the week — never "
    "guess or assume. Use date_difference for computing spans between dates."
  )

  def __init__(self):
    super().__init__()

  @property
  def tools(self) -> list:
    return [get_current_time, date_difference]
