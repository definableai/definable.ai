from definable.agent import Agent, MetricsMiddleware, MockModel, RetryMiddleware


class CaptureMiddleware:
  def __init__(self) -> None:
    self.session_ids: list[str | None] = []

  async def __call__(self, context, next_handler):
    self.session_ids.append(context.session_id)
    result = await next_handler(context)
    result.metadata = {"captured": True}
    return result


capture = CaptureMiddleware()
metrics = MetricsMiddleware()

agent = Agent(model=MockModel(responses=["hello"]), session_id="docs-session").use(capture).use(metrics)
output = agent.run("Hi")

assert output.content == "hello"
assert output.metadata == {"captured": True}
assert capture.session_ids == ["docs-session"]
assert metrics.run_count == 1


attempts = {"count": 0}


class FlakyMiddleware:
  async def __call__(self, context, next_handler):
    attempts["count"] += 1
    if attempts["count"] == 1:
      raise ConnectionError("temporary")
    return await next_handler(context)


retrying_agent = (
  Agent(model=MockModel(responses=["recovered"])).use(RetryMiddleware(max_retries=1, backoff_base=0.0, backoff_max=0.0)).use(FlakyMiddleware())
)
retry_output = retrying_agent.run("retry")

assert retry_output.content == "recovered"
assert attempts["count"] == 2
