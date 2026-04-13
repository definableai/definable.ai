from fastapi.testclient import TestClient

from definable.agent import Agent
from definable.agent.runtime import AgentServer
from definable.agent.testing import MockModel


agent = Agent(model=MockModel(responses=["pong"]), instructions="Reply briefly.")
client = TestClient(AgentServer(agent).create_app())

response = client.post("/run", json={"input": "ping", "session_id": "docs-session"})

assert response.status_code == 200
assert response.json()["content"] == "pong"
assert response.json()["session_id"] == "docs-session"
