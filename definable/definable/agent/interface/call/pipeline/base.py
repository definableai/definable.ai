"""Abstract base for voice pipeline strategies."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.agent.interface.call.call import CallSession
  from definable.agent.interface.call.telephony.base import TelephonyProvider


class CallPipeline(ABC):
  """Abstract base for voice pipeline strategies.

  A pipeline defines how audio/text flows between the telephony
  provider and the agent during a call. Three strategies exist:

    - **ManagedPipeline**: Provider handles STT/TTS (e.g. ConversationRelay).
      Text flows both directions. Maps directly to agent.arun().
    - **CascadingPipeline**: Raw audio → STT → agent.arun() → TTS → audio.
      Full control over each component.
    - **RealtimePipeline**: Audio proxied to a speech-to-speech model
      (e.g. OpenAI Realtime API). Lowest latency.

  Subclasses implement ``handle_call`` which runs for the full
  duration of a single call's WebSocket connection.
  """

  @abstractmethod
  async def handle_call(
    self,
    websocket: Any,
    call_session: "CallSession",
    agent: "Agent",
    telephony: "TelephonyProvider",
  ) -> None:
    """Handle the full lifecycle of a single call over WebSocket.

    This method runs for the duration of the call. It receives
    events from the telephony provider's WebSocket, processes them
    (transcription, agent invocation, synthesis), and sends responses
    back through the WebSocket.

    Args:
      websocket: The WebSocket connection (FastAPI WebSocket).
      call_session: The CallSession tracking this call's state.
      agent: The Agent instance to invoke for each utterance.
      telephony: The telephony provider for encoding responses.
    """
    ...
