from definable.agent.run import RunContext
from definable.tool import tool
from definable.tool.function import FunctionCall


@tool
def create_ticket(
  title: str,
  dependencies=None,
  session_state=None,
  run_context=None,
) -> str:
  """Create a ticket using injected runtime objects."""
  dependencies["tickets"].append((title, run_context.run_id, session_state["user"]))
  return f"created:{title}"


function = create_ticket.model_copy(deep=True)
function._dependencies = {"tickets": []}
function._session_state = {"user": "alice"}
function._run_context = RunContext(
  run_id="run-1",
  session_id="session-1",
  user_id="alice",
  session_state=function._session_state,
)

result = FunctionCall(function=function, arguments={"title": "Fix login"}).execute()

assert "dependencies" not in function.parameters["properties"]
assert result.result == "created:Fix login"
assert function._dependencies["tickets"] == [("Fix login", "run-1", "alice")]
