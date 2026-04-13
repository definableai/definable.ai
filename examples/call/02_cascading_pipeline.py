"""Cascading Voice Pipeline — Deepgram STT + Cartesia TTS.

This example creates a voice agent using the cascading pipeline:
raw audio flows through our own STT and TTS providers instead of
relying on Twilio's built-in speech processing.

Architecture:
  Caller speaks → Twilio Media Streams → Deepgram STT → text
  → Agent.arun() → text → Cartesia TTS → audio → Twilio → Caller hears

This gives full control over transcription and synthesis — you can
swap in any STT/TTS provider without changing the agent code.

Requirements:
  1. Install dependencies:
     pip install definable[call]

  2. Set environment variables:
     export OPENAI_API_KEY=sk-...
     export TWILIO_ACCOUNT_SID=AC...
     export TWILIO_AUTH_TOKEN=...
     export DEEPGRAM_API_KEY=...
     export CARTESIA_API_KEY=...

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
from definable.agent.interface.call.stt.deepgram import DeepgramSTT
from definable.agent.interface.call.tts.cartesia import CartesiaTTS
from definable.agent.runtime.runner import AgentRuntime
from definable.tool.decorator import tool


# --- Define tools ---


@tool
def lookup_account(account_number: str) -> str:
  """Look up a customer account by account number.

  Args:
    account_number: The customer's account number.
  """
  return f"Account {account_number}: Active, balance $142.50, next payment due March 15."


@tool
def transfer_to_agent(department: str) -> str:
  """Transfer the call to a human agent in the specified department.

  Args:
    department: The department to transfer to (billing, support, sales).
  """
  return f"Transferring to {department}. Please hold while I connect you."


# --- Create STT and TTS providers ---

stt = DeepgramSTT(
  # api_key= falls back to DEEPGRAM_API_KEY env var
  model="nova-3",
  language="en-US",
  endpointing=300,  # ms of silence to trigger final transcript
  smart_format=True,
)

tts = CartesiaTTS(
  # api_key= falls back to CARTESIA_API_KEY env var
  model="sonic-2",
  voice_id="a0e99841-438c-4a64-b679-ae501e7d6091",  # "Barbershop Man" voice
  language="en",
  speed="normal",
)


# --- Create the agent ---

agent = Agent(
  model="openai/gpt-4o-mini",
  instructions=(
    "You are a helpful customer service agent for Acme Corp. "
    "You can look up account information and transfer calls. "
    "Keep responses concise — you're on a phone call, so short "
    "sentences work best. Be warm and conversational."
  ),
  tools=[lookup_account, transfer_to_agent],
)


# --- Create the call interface with cascading pipeline ---

call = CallInterface(
  agent=agent,
  provider="twilio",
  phone_number="+15551234567",
  pipeline="cascading",  # Use our own STT/TTS
  stt=stt,
  tts=tts,
  welcome_message="Hello! Thank you for calling Acme Corp. How can I help you?",
  language="en-US",
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
