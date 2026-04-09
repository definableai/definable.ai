import asyncio

from definable.agent import Agent, MockModel
from definable.agent.scheduler import JobStatus, Scheduler
from definable.agent.trigger import OneShot, TriggerExecutor


async def main() -> None:
  agent = Agent(model=MockModel(responses=["scheduled run"]))
  scheduler = Scheduler(tick_interval=0.01, max_concurrent=1)

  async def handler(event):
    return "Run from schedule"

  trigger = OneShot(delay=0.01)
  trigger.handler = handler
  job = scheduler.add(trigger, name="one-shot-demo", max_runs=1)

  started: list[str] = []
  completed: list[tuple[str, int]] = []
  scheduler.on_job_started = lambda scheduled_job: started.append(scheduled_job.name)
  scheduler.on_job_completed = lambda scheduled_job: completed.append((scheduled_job.name, scheduled_job.run_count))

  task = asyncio.create_task(scheduler.start(TriggerExecutor(agent)))
  await asyncio.sleep(0.05)
  scheduler.stop()
  await task

  assert job.status == JobStatus.COMPLETED
  assert started == ["one-shot-demo"]
  assert completed == [("one-shot-demo", 1)]
  assert agent.model.call_count == 1


asyncio.run(main())
