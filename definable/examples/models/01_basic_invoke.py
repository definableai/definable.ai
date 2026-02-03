"""
Basic synchronous model invocation.

This example shows how to:
- Initialize an OpenAI chat model
- Create messages for the model
- Invoke the model and get a response
- Access response content and metrics

Requirements:
    export OPENAI_API_KEY=sk-...
"""

from definable.models.message import Message
from definable.models.openai import OpenAIChat


def main():
  # Initialize the model
  # API key is read from OPENAI_API_KEY environment variable by default
  model = OpenAIChat(
    id="gpt-4o-mini",  # Model ID
    # api_key=os.getenv("OPENAI_API_KEY"),  # Optional: explicit API key
    # temperature=0.7,  # Optional: control randomness
    # max_tokens=1000,  # Optional: limit response length
  )

  # Create the conversation messages
  messages = [
    Message(role="system", content="You are a helpful assistant."),
    Message(role="user", content="What is the capital of France?"),
  ]

  # Invoke the model (synchronous)
  response = model.invoke(
    messages=messages,
    assistant_message=Message(role="assistant", content=""),
  )

  # Access the response content
  print("Response:")
  print(response.content)
  print()

  # Access response metrics (token usage)
  if response.response_usage:
    print("Metrics:")
    print(f"  Input tokens: {response.response_usage.input_tokens}")
    print(f"  Output tokens: {response.response_usage.output_tokens}")
    print(f"  Total tokens: {response.response_usage.total_tokens}")


if __name__ == "__main__":
  main()
