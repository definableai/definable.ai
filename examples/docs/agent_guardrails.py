import asyncio

from definable.agent import Agent
from definable.agent.guardrail import Guardrails, max_tokens
from definable.agent.run import RunStatus
from definable.agent.testing import MockModel


async def main() -> None:
  agent = Agent(
    model=MockModel(responses=["ok"]),
    guardrails=Guardrails(
      input=[max_tokens(2)],
      on_block="return_message",
    ),
  )

  output = await agent.arun("This input should be blocked before the model runs.")

  assert output.status == RunStatus.blocked


asyncio.run(main())
