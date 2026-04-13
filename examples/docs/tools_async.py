import asyncio

from definable.tool import tool
from definable.tool.function import FunctionCall


@tool
async def fetch_record(record_id: str) -> str:
  """Fetch a record asynchronously."""
  await asyncio.sleep(0)
  return f"record:{record_id}"


async def main() -> None:
  result = await FunctionCall(function=fetch_record, arguments={"record_id": "42"}).aexecute()
  assert result.result == "record:42"


asyncio.run(main())
