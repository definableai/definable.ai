"""
Test Definable UI Integration - IPC-Style Communication (Like Electron!)

This demonstrates the zero-config integration of UI with agents using
Electron-style IPC (Inter-Process Communication) via window.postMessage API!

Architecture:
- Agent auto-registers with ChatWindow (no decorators needed)
- IPC Bridge: window.definableChat.send() / .on() (Electron-style!)
- Streaming responses via arun_stream() for real-time UX
- No direct fetch() calls - everything through IPC bridge
- Python backend with FastAPI + Jinja2 templates (Django-style)
- Web Component (CDN-ready) for UI

IPC API (like Electron):
    // Frontend
    window.definableChat.send('chat:message', { content: 'Hello!' })
    window.definableChat.on('chat:chunk', (data) => console.log(data))

Usage:
    export OPENAI_API_KEY="sk-..."
    python -m definable.test
"""

import os
from definable.agents import Agent
from definable.models.openai import OpenAIChat
from definable.tools.decorator import tool
from definable.ui import ChatWindow


# Define example tools
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny and 72°F"


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


def main():
    """Main test function demonstrating IPC-style UI integration."""

    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Set OPENAI_API_KEY environment variable")
        return

    # Create UI
    chat = ChatWindow(title="Definable AI Assistant", port=8000)

    # Create agent - auto-registers with UI
    Agent(
        model=OpenAIChat(id="gpt-4o-mini", api_key=api_key),
        tools=[get_weather, calculate],
        instructions="You are a helpful assistant.",
        ui=chat,
    )

    print("✅ Agent ready | 📡 IPC Bridge | 🌐 http://localhost:8000\n")
    chat.show()

if __name__ == "__main__":
    main()
