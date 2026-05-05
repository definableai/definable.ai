from fastapi.testclient import TestClient

from definable.agent import Agent
from definable.agent.auth import APIKeyAuth
from definable.agent.runtime import AgentServer
from definable.agent.testing import MockModel


agent = Agent(model=MockModel(responses=["pong"]), instructions="Reply briefly.")
agent.auth = APIKeyAuth(keys={"secret-key"})

client = TestClient(AgentServer(agent).create_app())

unauthorized = client.post("/run", json={"input": "ping"})
authorized = client.post(
  "/run",
  headers={"X-API-Key": "secret-key"},
  json={"input": "ping"},
)

assert unauthorized.status_code == 401
assert authorized.status_code == 200
assert authorized.json()["content"] == "pong"
