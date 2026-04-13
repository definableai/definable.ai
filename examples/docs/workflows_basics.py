import asyncio

from definable.agent.workflow import Condition, Loop, Parallel, Router, Step, Workflow


async def main() -> None:
  async def research(step_input):
    return "Research notes"

  def draft(step_input):
    return f"Draft from {step_input.get_last_step_content()}"

  workflow = Workflow(
    name="publish",
    steps=[
      Step(name="research", executor=research),
      Step(name="draft", executor=draft),
    ],
  )
  output = await workflow.arun("Write an overview.")
  assert output.content == "Draft from Research notes"
  assert output.get_step_content("research") == "Research notes"

  parallel = Workflow(
    name="analysis",
    steps=Parallel(
      name="review",
      steps=[
        Step(name="product", executor=lambda step_input: "Product analysis"),
        Step(name="risk", executor=lambda step_input: "Risk analysis"),
      ],
    ),
  )
  parallel_output = await parallel.arun("Review the launch.")
  assert "[product]: Product analysis" in (parallel_output.content or "")
  assert "[risk]: Risk analysis" in (parallel_output.content or "")

  router = Workflow(
    name="support",
    steps=Router(
      name="dispatch",
      selector=lambda step_input: "technical" if "bug" in (step_input.input or "") else "general",
      routes={
        "technical": Step(name="technical", executor=lambda step_input: "Technical queue"),
        "general": Step(name="general", executor=lambda step_input: "General queue"),
      },
    ),
  )
  routed = await router.arun("There is a bug in checkout.")
  assert routed.content == "Technical queue"

  revision_state = {"attempts": 0}

  def revise(step_input):
    revision_state["attempts"] += 1
    return f"Revision {revision_state['attempts']}"

  def review(step_input):
    if revision_state["attempts"] >= 2:
      return "APPROVED"
    return "CHANGES"

  looped = Workflow(
    name="revision",
    steps=Loop(
      name="revise_until_approved",
      steps=[
        Step(name="revise", executor=revise),
        Step(name="review", executor=review),
      ],
      end_condition=lambda outputs: outputs[-1].content == "APPROVED",
      max_iterations=3,
    ),
  )
  loop_output = await looped.arun("Polish the draft.")
  assert loop_output.content == "APPROVED"

  conditional = Workflow(
    name="gate",
    steps=Condition(
      name="quality_gate",
      condition=lambda step_input: "pass" in (step_input.input or "").lower(),
      true_steps=Step(name="publish", executor=lambda step_input: "Published"),
      false_steps=Step(name="revise", executor=lambda step_input: "Needs revision"),
    ),
  )
  conditional_output = await conditional.arun("pass this draft")
  assert conditional_output.content == "Published"


asyncio.run(main())
