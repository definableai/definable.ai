"""
Structured output with Pydantic models.

This example shows how to:
- Define Pydantic models for response schemas
- Get typed, validated responses from the model
- Access parsed response data

Requirements:
    export OPENAI_API_KEY=sk-...
"""

from typing import List, Optional

from definable.models.message import Message
from definable.models.openai import OpenAIChat
from pydantic import BaseModel, Field


# Define response schemas using Pydantic
class TaskItem(BaseModel):
  """A single task item."""

  title: str = Field(description="The task title")
  priority: str = Field(description="Priority level: high, medium, or low")
  estimated_hours: Optional[float] = Field(default=None, description="Estimated hours to complete")


class TaskList(BaseModel):
  """A list of tasks."""

  tasks: List[TaskItem] = Field(description="List of tasks")
  total_estimated_hours: Optional[float] = Field(default=None, description="Total estimated time")


class MathSolution(BaseModel):
  """A math problem solution."""

  problem: str = Field(description="The original problem")
  steps: List[str] = Field(description="Step-by-step solution")
  answer: float = Field(description="The final numerical answer")


def structured_task_list():
  """Generate a structured task list."""
  model = OpenAIChat(id="gpt-4o-mini")

  messages = [
    Message(
      role="user",
      content="Create a task list for building a simple web application. Include 3-4 tasks.",
    ),
  ]

  response = model.invoke(
    messages=messages,
    assistant_message=Message(role="assistant", content=""),
    response_format=TaskList,  # Pass the Pydantic model
  )

  print("Structured Task List:")
  print("-" * 40)

  # Access the parsed response
  if response.parsed:
    task_list: TaskList = response.parsed
    for i, task in enumerate(task_list.tasks, 1):
      print(f"{i}. {task.title}")
      print(f"   Priority: {task.priority}")
      if task.estimated_hours:
        print(f"   Estimated: {task.estimated_hours} hours")
      print()

    if task_list.total_estimated_hours:
      print(f"Total estimated time: {task_list.total_estimated_hours} hours")
  else:
    # Fallback to raw content if parsing fails
    print(response.content)


def structured_math_solution():
  """Get a structured math solution."""
  model = OpenAIChat(id="gpt-4o-mini")

  messages = [
    Message(
      role="user",
      content="Solve this problem: If a train travels at 60 mph for 2.5 hours, how far does it go?",
    ),
  ]

  response = model.invoke(
    messages=messages,
    assistant_message=Message(role="assistant", content=""),
    response_format=MathSolution,
  )

  print("\nMath Solution:")
  print("-" * 40)

  if response.parsed:
    solution: MathSolution = response.parsed
    print(f"Problem: {solution.problem}")
    print("\nSteps:")
    for i, step in enumerate(solution.steps, 1):
      print(f"  {i}. {step}")
    print(f"\nAnswer: {solution.answer}")


class SentimentAnalysis(BaseModel):
  """Sentiment analysis result."""

  text: str
  sentiment: str = Field(description="positive, negative, or neutral")
  confidence: float = Field(description="Confidence score between 0 and 1")
  key_phrases: List[str] = Field(description="Key phrases that influenced the sentiment")


def structured_sentiment():
  """Analyze sentiment with structured output."""
  model = OpenAIChat(id="gpt-4o-mini")

  texts = [
    "I absolutely love this product! It exceeded all my expectations.",
    "The service was okay, nothing special but not bad either.",
    "This is the worst experience I've ever had. Total waste of money.",
  ]

  print("\nSentiment Analysis:")
  print("-" * 40)

  for text in texts:
    response = model.invoke(
      messages=[Message(role="user", content=f"Analyze the sentiment: {text}")],
      assistant_message=Message(role="assistant", content=""),
      response_format=SentimentAnalysis,
    )

    if response.parsed:
      result: SentimentAnalysis = response.parsed
      print(f"\nText: {result.text[:50]}...")
      print(f"Sentiment: {result.sentiment} (confidence: {result.confidence:.2f})")
      print(f"Key phrases: {', '.join(result.key_phrases)}")


if __name__ == "__main__":
  structured_task_list()
  structured_math_solution()
  structured_sentiment()
