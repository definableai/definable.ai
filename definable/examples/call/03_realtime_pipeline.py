"""Realtime Voice Pipeline — OpenAI Realtime API (Speech-to-Speech).

This example creates a voice agent using the realtime pipeline:
audio flows directly between the caller and OpenAI's speech-to-speech
model with no separate STT/TTS roundtrip.

Architecture:
  Caller speaks → Twilio Media Streams → OpenAI Realtime (STT+LLM+TTS) → audio → Twilio → Caller hears
  Tool calls:   OpenAI → tool_call event → Pipeline invokes tool → result → OpenAI continues

This is the lowest-latency pipeline mode (~200-300ms TTFB) since
the model processes speech natively. Function calling is handled
by the pipeline: the provider emits tool_call events and the
pipeline invokes the matching tool from the agent's registry.

Requirements:
  1. Install dependencies:
     pip install definable[call]

  2. Set environment variables:
     export OPENAI_API_KEY=sk-...
     export TWILIO_ACCOUNT_SID=AC...
     export TWILIO_AUTH_TOKEN=...

  3. Configure your Twilio phone number webhook:
     For local development, use ngrok:
       ngrok http 8000
     Set the webhook URL to:
       https://<your-ngrok-url>/call/incoming

  4. Run this script and call your Twilio number!
"""

import asyncio

from definable.agent import Agent
from definable.agent.interface.call import CallInterface
from definable.agent.interface.call.realtime.openai import OpenAIRealtimeProvider
from definable.agent.runtime.runner import AgentRuntime
from definable.tool.decorator import tool


# --- Define tools ---


@tool
def check_order_status(order_id: str) -> str:
  """Check the status of a customer order.

  Args:
    order_id: The order ID to look up.
  """
  return f"Order {order_id}: Shipped on Feb 23, estimated delivery Feb 27."


@tool
def schedule_callback(phone_number: str, preferred_time: str) -> str:
  """Schedule a callback for the customer.

  Args:
    phone_number: Customer's phone number.
    preferred_time: When the customer would like to be called back.
  """
  return f"Callback scheduled for {preferred_time} at {phone_number}."


# --- Create the realtime provider ---

realtime = OpenAIRealtimeProvider(
  # api_key= falls back to OPENAI_API_KEY env var
  model="gpt-4o-realtime-preview",
  voice="alloy",
  temperature=0.8,
  turn_detection={
    "type": "server_vad",
    "threshold": 0.5,
    "prefix_padding_ms": 300,
    "silence_duration_ms": 500,
  },
)


# --- Create the agent ---

agent = Agent(
  model="openai/gpt-4o-mini",  # Fallback model (not used in realtime mode)
  instructions=(
    "You are a helpful order support agent for ShipFast Inc. "
    "You can check order status and schedule callbacks. "
    "Keep responses concise — you're on a phone call. "
    "Be warm and professional."
  ),
  tools=[check_order_status, schedule_callback],
)


# --- Create the call interface with realtime pipeline ---

call = CallInterface(
  agent=agent,
  provider="twilio",
  phone_number="+15551234567",
  pipeline="realtime",  # Speech-to-speech via OpenAI Realtime API
  realtime=realtime,
  welcome_message="Hello! Thank you for calling ShipFast. How can I help you?",
)


# --- Run with the runtime ---


async def main():
  runtime = AgentRuntime(
    agent,
    interfaces=[call],
    host="0.0.0.0",
    port=8000,
  )
  await runtime.start()


if __name__ == "__main__":
  asyncio.run(main())
