"""Managed Voice Agent — Twilio ConversationRelay.

This example creates a voice-enabled agent that can receive phone calls
via Twilio. Twilio handles speech-to-text and text-to-speech natively
using ConversationRelay, so the agent works entirely in text — no audio
processing needed on our side.

Architecture:
  Caller speaks → Twilio (STT) → text → Agent.arun() → text → Twilio (TTS) → Caller hears

Requirements:
  1. Install dependencies:
     pip install definable[call]

  2. Set environment variables:
     export OPENAI_API_KEY=sk-...
     export TWILIO_ACCOUNT_SID=AC...
     export TWILIO_AUTH_TOKEN=...

  3. You need a Twilio phone number configured to send incoming
     call webhooks to your server's /call/incoming endpoint.
     For local development, use ngrok:
       ngrok http 8000
     Then set the Twilio phone number's webhook URL to:
       https://<your-ngrok-url>/call/incoming

  4. Run this script and call your Twilio number!
"""

from definable.agent import Agent
from definable.agent.interface.call import CallInterface
from definable.tool.decorator import tool

# --- Define tools the agent can use during calls ---


@tool
def check_order_status(order_id: str) -> str:
  """Check the status of a customer order.

  Args:
    order_id: The order ID to look up.
  """
  # In production, this would query your database
  return f"Order {order_id} is currently being processed and will ship within 2 business days."


@tool
def schedule_callback(phone_number: str, preferred_time: str) -> str:
  """Schedule a callback for the customer.

  Args:
    phone_number: Customer's phone number.
    preferred_time: When the customer prefers to be called back.
  """
  return f"Callback scheduled for {preferred_time} at {phone_number}."


# --- Create the agent with call interface ---

agent = Agent(
  model="openai/gpt-4o-mini",
  instructions=(
    "You are a friendly and professional customer service agent for Acme Corp. "
    "You help customers check order statuses, schedule callbacks, and answer "
    "general questions. Keep responses concise — you're speaking on the phone, "
    "so short sentences work best. Be warm and conversational."
  ),
  tools=[check_order_status, schedule_callback],
  interfaces=CallInterface(
    provider="twilio",
    phone_number="+15551234567",  # Your Twilio phone number
    pipeline="managed",  # Twilio handles STT/TTS
    welcome_message="Hello! Thank you for calling Acme Corp. How can I help you today?",
    tts_provider="google",
    stt_provider="deepgram",
    voice="en-US-Standard-A",
    language="en-US",
    interruptible="any",
    interrupt_sensitivity="medium",
  ),
)

if __name__ == "__main__":
  agent.serve(port=8000)
