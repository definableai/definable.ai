"""
Unit tests for ClaudeCodeAgent -- all offline, no CLI needed.

Covers:
  - Message type parsing (assistant, result, system, stream, control)
  - ToolBridge registration, execution, error handling
  - Parser: CLI messages -> RunOutput
  - Event conversion: messages -> EventBus events
  - System prompt construction
  - CLI args construction
  - Control handler (guardrails, MCP tool execution)
  - Agent config and memory resolution

Migrated from tests_e2e/unit/test_claude_code_agent.py -- all original tests preserved.
"""

import asyncio

import pytest

from definable.claude_code.bridge import ToolBridge
from definable.claude_code.parser import message_to_event, parse_to_run_output
from definable.claude_code.types import (
  AssistantMessage,
  ControlRequest,
  ResultMessage,
  StreamEvent,
  SystemMessage,
  TextBlock,
  ThinkingBlock,
  ToolResultBlock,
  ToolUseBlock,
  parse_message,
)
from definable.agent.events import (
  ReasoningContentDeltaEvent,
  RunCompletedEvent,
  RunContentEvent,
  RunContext,
  RunStatus,
  ToolCallStartedEvent,
)
from definable.tool.function import Function


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def run_context():
  return RunContext(run_id="run-1", session_id="sess-1", user_id="user-1")


@pytest.fixture
def sample_tool():
  def deploy(branch: str = "main") -> str:
    """Deploy the application."""
    return f"Deployed {branch}"

  return Function.from_callable(deploy)


@pytest.fixture
def async_tool():
  async def fetch_data(url: str) -> str:
    """Fetch data from URL."""
    return f"Data from {url}"

  return Function.from_callable(fetch_data)


# ===========================================================================
# Message type parsing tests
# ===========================================================================


@pytest.mark.unit
class TestParseMessage:
  def test_parse_assistant_text(self):
    data = {
      "type": "assistant",
      "message": {
        "content": [{"type": "text", "text": "Hello world"}],
        "model": "claude-sonnet-4-6",
      },
    }
    msg = parse_message(data)
    assert isinstance(msg, AssistantMessage)
    assert len(msg.content) == 1
    assert isinstance(msg.content[0], TextBlock)
    assert msg.content[0].text == "Hello world"
    assert msg.model == "claude-sonnet-4-6"

  def test_parse_assistant_thinking(self):
    data = {
      "type": "assistant",
      "message": {
        "content": [{"type": "thinking", "thinking": "Let me analyze...", "signature": "sig-abc"}],
        "model": "claude-sonnet-4-6",
      },
    }
    msg = parse_message(data)
    assert isinstance(msg, AssistantMessage)
    block = msg.content[0]
    assert isinstance(block, ThinkingBlock)
    assert block.thinking == "Let me analyze..."
    assert block.signature == "sig-abc"

  def test_parse_assistant_tool_use(self):
    data = {
      "type": "assistant",
      "message": {
        "content": [
          {
            "type": "tool_use",
            "id": "tool_1",
            "name": "Read",
            "input": {"file_path": "/auth.py"},
          }
        ],
        "model": "claude-sonnet-4-6",
      },
    }
    msg = parse_message(data)
    assert isinstance(msg, AssistantMessage)
    block = msg.content[0]
    assert isinstance(block, ToolUseBlock)
    assert block.id == "tool_1"
    assert block.name == "Read"
    assert block.input == {"file_path": "/auth.py"}

  def test_parse_assistant_tool_result(self):
    data = {
      "type": "assistant",
      "message": {
        "content": [
          {
            "type": "tool_result",
            "tool_use_id": "tool_1",
            "content": "file content here",
            "is_error": False,
          }
        ],
        "model": "claude-sonnet-4-6",
      },
    }
    msg = parse_message(data)
    block = msg.content[0]  # type: ignore[union-attr]
    assert isinstance(block, ToolResultBlock)
    assert block.tool_use_id == "tool_1"
    assert block.content == "file content here"
    assert block.is_error is False

  def test_parse_result_message(self):
    data = {
      "type": "result",
      "subtype": "success",
      "session_id": "sess-123",
      "duration_ms": 15000,
      "duration_api_ms": 8000,
      "total_cost_usd": 0.045,
      "input_tokens": 2000,
      "output_tokens": 500,
      "is_error": False,
      "turn_count": 5,
    }
    msg = parse_message(data)
    assert isinstance(msg, ResultMessage)
    assert msg.subtype == "success"
    assert msg.session_id == "sess-123"
    assert msg.duration_ms == 15000
    assert msg.total_cost_usd == 0.045
    assert msg.input_tokens == 2000
    assert msg.output_tokens == 500
    assert msg.is_error is False
    assert msg.turn_count == 5

  def test_parse_system_message(self):
    data = {"type": "system", "subtype": "init", "version": "1.0"}
    msg = parse_message(data)
    assert isinstance(msg, SystemMessage)
    assert msg.subtype == "init"
    assert msg.data["version"] == "1.0"

  def test_parse_stream_event(self):
    data = {
      "type": "stream_event",
      "uuid": "msg-1",
      "session_id": "sess-123",
      "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}},
    }
    msg = parse_message(data)
    assert isinstance(msg, StreamEvent)
    assert msg.uuid == "msg-1"
    assert msg.event["delta"]["text"] == "Hello"

  def test_parse_control_request(self):
    data = {
      "type": "control_request",
      "id": "req-1",
      "subtype": "can_use_tool",
      "tool_name": "Bash",
      "input": {"command": "ls"},
    }
    msg = parse_message(data)
    assert isinstance(msg, ControlRequest)
    assert msg.id == "req-1"
    assert msg.subtype == "can_use_tool"
    assert msg.tool_name == "Bash"
    assert msg.input == {"command": "ls"}

  def test_parse_unknown_type(self):
    with pytest.raises(ValueError, match="Unknown message type"):
      parse_message({"type": "bogus"})

  def test_parse_mixed_content(self):
    """AssistantMessage with text + thinking + tool_use blocks."""
    data = {
      "type": "assistant",
      "message": {
        "content": [
          {"type": "thinking", "thinking": "Step 1...", "signature": "s1"},
          {"type": "text", "text": "I'll fix the bug."},
          {"type": "tool_use", "id": "t1", "name": "Edit", "input": {"file_path": "/f.py"}},
        ],
        "model": "claude-sonnet-4-6",
      },
    }
    msg = parse_message(data)
    assert isinstance(msg, AssistantMessage)
    assert len(msg.content) == 3
    assert isinstance(msg.content[0], ThinkingBlock)
    assert isinstance(msg.content[1], TextBlock)
    assert isinstance(msg.content[2], ToolUseBlock)


# ===========================================================================
# ToolBridge tests
# ===========================================================================


@pytest.mark.unit
class TestToolBridge:
  def test_bridge_tool_registration(self, sample_tool):
    bridge = ToolBridge(tools=[sample_tool])
    assert bridge.tool_count == 1

  def test_bridge_mcp_config(self, sample_tool):
    bridge = ToolBridge(tools=[sample_tool])
    config = bridge.get_mcp_config()
    assert "definable" in config
    tools = config["definable"]["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == "deploy"
    assert "inputSchema" in tools[0]

  def test_bridge_tool_names(self, sample_tool):
    bridge = ToolBridge(tools=[sample_tool])
    names = bridge.get_tool_names()
    assert names == ["mcp__definable__deploy"]

  def test_bridge_custom_server_name(self, sample_tool):
    bridge = ToolBridge(tools=[sample_tool])
    names = bridge.get_tool_names(server_name="myapp")
    assert names == ["mcp__myapp__deploy"]

  def test_bridge_tool_execution(self, sample_tool):
    bridge = ToolBridge(tools=[sample_tool])
    result = asyncio.get_event_loop().run_until_complete(bridge.execute("mcp__definable__deploy", {"branch": "staging"}))
    assert result["content"][0]["text"] == "Deployed staging"
    assert "isError" not in result

  def test_bridge_async_tool(self, async_tool):
    bridge = ToolBridge(tools=[async_tool])
    result = asyncio.get_event_loop().run_until_complete(bridge.execute("mcp__definable__fetch_data", {"url": "https://example.com"}))
    assert "Data from https://example.com" in result["content"][0]["text"]

  def test_bridge_unknown_tool(self):
    bridge = ToolBridge(tools=[])
    result = asyncio.get_event_loop().run_until_complete(bridge.execute("nonexistent", {}))
    assert result["isError"] is True
    assert "Unknown tool" in result["content"][0]["text"]

  def test_bridge_tool_error(self):
    def bad_tool() -> str:
      """A tool that raises."""
      raise ValueError("intentional error")

    fn = Function.from_callable(bad_tool)
    bridge = ToolBridge(tools=[fn])
    result = asyncio.get_event_loop().run_until_complete(bridge.execute("bad_tool", {}))
    assert result["isError"] is True
    assert "intentional error" in result["content"][0]["text"]

  def test_bridge_empty(self):
    bridge = ToolBridge()
    assert bridge.tool_count == 0
    config = bridge.get_mcp_config()
    assert config == {}


# ===========================================================================
# Parser tests (CLI messages -> RunOutput)
# ===========================================================================


@pytest.mark.unit
class TestParseToRunOutput:
  def test_basic_content(self, run_context):
    messages = [
      AssistantMessage(content=[TextBlock(text="Hello world")], model="claude-sonnet-4-6"),
      ResultMessage(subtype="success", session_id="sess-1", input_tokens=100, output_tokens=50),
    ]
    result = parse_to_run_output(messages, run_context, "claude-sonnet-4-6")  # type: ignore[arg-type]
    assert result.content == "Hello world"
    assert result.status == RunStatus.completed
    assert result.model == "claude-sonnet-4-6"

  def test_metrics_from_result(self, run_context):
    messages = [
      ResultMessage(
        subtype="success",
        session_id="sess-1",
        input_tokens=2000,
        output_tokens=500,
        total_cost_usd=0.045,
        duration_ms=15000,
      ),
    ]
    result = parse_to_run_output(messages, run_context, "claude-sonnet-4-6")  # type: ignore[arg-type]
    assert result.metrics.input_tokens == 2000  # type: ignore[union-attr]
    assert result.metrics.output_tokens == 500  # type: ignore[union-attr]
    assert result.metrics.total_tokens == 2500  # type: ignore[union-attr]
    assert result.metrics.cost == 0.045  # type: ignore[union-attr]
    assert result.metrics.duration == 15.0  # type: ignore[union-attr]

  def test_tool_executions(self, run_context):
    messages = [
      AssistantMessage(
        content=[
          ToolUseBlock(id="t1", name="Read", input={"file_path": "/a.py"}),
          ToolResultBlock(tool_use_id="t1", content="file content"),
        ],
        model="claude-sonnet-4-6",
      ),
      ResultMessage(subtype="success", session_id="sess-1"),
    ]
    result = parse_to_run_output(messages, run_context, "claude-sonnet-4-6")  # type: ignore[arg-type]
    assert result.tools is not None
    assert len(result.tools) == 1
    assert result.tools[0].tool_name == "Read"
    assert result.tools[0].result == "file content"

  def test_thinking_content(self, run_context):
    messages = [
      AssistantMessage(
        content=[
          ThinkingBlock(thinking="Step 1: analyze..."),
          TextBlock(text="Here's my answer."),
        ],
        model="claude-sonnet-4-6",
      ),
      ResultMessage(subtype="success", session_id="sess-1"),
    ]
    result = parse_to_run_output(messages, run_context, "claude-sonnet-4-6")  # type: ignore[arg-type]
    assert result.reasoning_content == "Step 1: analyze..."
    assert result.content == "Here's my answer."

  def test_structured_output(self, run_context):
    messages = [
      ResultMessage(
        subtype="success",
        session_id="sess-1",
        structured_output={"name": "Alice", "age": 30},
      ),
    ]
    result = parse_to_run_output(messages, run_context, "claude-sonnet-4-6")  # type: ignore[arg-type]
    assert result.content == {"name": "Alice", "age": 30}

  def test_error_status(self, run_context):
    messages = [
      ResultMessage(subtype="error", session_id="sess-1", is_error=True),
    ]
    result = parse_to_run_output(messages, run_context, "claude-sonnet-4-6")  # type: ignore[arg-type]
    assert result.status == RunStatus.error

  def test_session_id_from_result(self, run_context):
    messages = [
      ResultMessage(subtype="success", session_id="new-sess-42"),
    ]
    result = parse_to_run_output(messages, run_context, "claude-sonnet-4-6")  # type: ignore[arg-type]
    assert result.session_id == "new-sess-42"

  def test_empty_messages(self, run_context):
    result = parse_to_run_output([], run_context, "claude-sonnet-4-6")  # type: ignore[arg-type]
    assert result.content is None
    assert result.status == RunStatus.completed

  def test_stream_event_text_delta(self, run_context):
    messages = [
      StreamEvent(
        uuid="msg-1",
        session_id="sess-1",
        event={"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello "}},
      ),
      StreamEvent(
        uuid="msg-1",
        session_id="sess-1",
        event={"type": "content_block_delta", "delta": {"type": "text_delta", "text": "world"}},
      ),
      ResultMessage(subtype="success", session_id="sess-1"),
    ]
    result = parse_to_run_output(messages, run_context, "claude-sonnet-4-6")  # type: ignore[arg-type]
    assert result.content == "Hello world"


# ===========================================================================
# Event conversion tests
# ===========================================================================


@pytest.mark.unit
class TestMessageToEvent:
  def test_text_block_to_content_event(self, run_context):
    msg = AssistantMessage(content=[TextBlock(text="hello")], model="claude-sonnet-4-6")
    event = message_to_event(msg, run_context)
    assert isinstance(event, RunContentEvent)
    assert event.content == "hello"

  def test_thinking_block_to_reasoning_event(self, run_context):
    msg = AssistantMessage(content=[ThinkingBlock(thinking="analyzing...")], model="claude-sonnet-4-6")
    event = message_to_event(msg, run_context)
    assert isinstance(event, ReasoningContentDeltaEvent)
    assert event.reasoning_content == "analyzing..."

  def test_tool_use_block_to_tool_event(self, run_context):
    msg = AssistantMessage(content=[ToolUseBlock(id="t1", name="Bash", input={"command": "ls"})], model="claude-sonnet-4-6")
    event = message_to_event(msg, run_context)
    assert isinstance(event, ToolCallStartedEvent)
    assert event.tool.tool_name == "Bash"  # type: ignore[union-attr]

  def test_result_to_completed_event(self, run_context):
    msg = ResultMessage(subtype="success", session_id="s", input_tokens=100, output_tokens=50, duration_ms=5000, total_cost_usd=0.01)
    event = message_to_event(msg, run_context)
    assert isinstance(event, RunCompletedEvent)
    assert event.metrics.input_tokens == 100  # type: ignore[union-attr]
    assert event.metrics.cost == 0.01  # type: ignore[union-attr]

  def test_system_message_no_event(self, run_context):
    msg = SystemMessage(subtype="init", data={})
    event = message_to_event(msg, run_context)
    assert event is None

  def test_stream_text_delta_event(self, run_context):
    msg = StreamEvent(
      uuid="m1",
      session_id="s1",
      event={"type": "content_block_delta", "delta": {"type": "text_delta", "text": "chunk"}},
    )
    event = message_to_event(msg, run_context)
    assert isinstance(event, RunContentEvent)
    assert event.content == "chunk"


# ===========================================================================
# System prompt construction tests
# ===========================================================================


@pytest.mark.unit
class TestBuildSystemPrompt:
  def test_instructions_only(self):
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent(instructions="You are a helpful assistant.")
    prompt = agent._build_system_prompt()
    assert "You are a helpful assistant." in prompt

  def test_with_knowledge_context(self):
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent(instructions="Help the user.")
    prompt = agent._build_system_prompt(knowledge_ctx="[1] Auth uses JWT tokens")
    assert "<knowledge_context>" in prompt
    assert "Auth uses JWT tokens" in prompt

  def test_with_memory_context(self):
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent(instructions="Help the user.")
    prompt = agent._build_system_prompt(memory_ctx="- Prefers Python over JS")
    assert "<user_memory>" in prompt
    assert "Prefers Python over JS" in prompt

  def test_composition_order(self):
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent(instructions="Base instructions.")
    prompt = agent._build_system_prompt(knowledge_ctx="KB data", memory_ctx="Memory data")
    # Instructions should come first
    instr_pos = prompt.index("Base instructions")
    kb_pos = prompt.index("knowledge_context")
    mem_pos = prompt.index("user_memory")
    assert instr_pos < kb_pos < mem_pos

  def test_empty_instructions(self):
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent()
    prompt = agent._build_system_prompt()
    assert prompt == ""


# ===========================================================================
# CLI args construction tests
# ===========================================================================


@pytest.mark.unit
class TestBuildCliArgs:
  def test_default_args(self):
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent(model="claude-sonnet-4-6")
    args = agent._build_cli_args("system prompt here")
    assert "--output-format" in args
    assert "stream-json" in args
    assert "--model" in args
    assert "claude-sonnet-4-6" in args
    assert "--system-prompt" in args

  def test_max_turns(self):
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent(max_turns=10)
    args = agent._build_cli_args("prompt")
    assert "--max-turns" in args
    assert "10" in args

  def test_budget(self):
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent(max_budget_usd=5.0)
    args = agent._build_cli_args("prompt")
    assert "--max-budget-usd" in args
    assert "5.0" in args

  def test_thinking_tokens(self):
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent(thinking_budget_tokens=4096)
    args = agent._build_cli_args("prompt")
    assert "--max-thinking-tokens" in args
    assert "4096" in args

  def test_session_resume(self):
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent(continue_conversation=True)
    args = agent._build_cli_args("prompt", session_id="sess-abc")
    assert "--resume" in args
    assert "sess-abc" in args

  def test_allowed_tools_with_custom(self, sample_tool):
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent(allowed_tools=["Read", "Write"], tools=[sample_tool])
    agent._ensure_initialized()
    args = agent._build_cli_args("prompt")
    assert "--allowedTools" in args
    tools_str = args[args.index("--allowedTools") + 1]
    assert "Read" in tools_str
    assert "Write" in tools_str
    assert "mcp__definable__deploy" in tools_str


# ===========================================================================
# Control handler tests
# ===========================================================================


@pytest.mark.unit
class TestControlHandler:
  @pytest.mark.asyncio
  async def test_allow_tool_no_guardrails(self, run_context):
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent()
    request = ControlRequest(id="req-1", subtype="can_use_tool", tool_name="Read", input={})
    response = await agent._handle_control(request, run_context)
    assert response["behavior"] == "allow"

  @pytest.mark.asyncio
  async def test_deny_tool_with_guardrail(self, run_context):
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.agent.guardrail import Guardrails, tool_blocklist

    agent = ClaudeCodeAgent(guardrails=Guardrails(tool=[tool_blocklist({"Bash"})]))
    request = ControlRequest(id="req-1", subtype="can_use_tool", tool_name="Bash", input={"command": "rm -rf /"})
    response = await agent._handle_control(request, run_context)
    assert response["behavior"] == "deny"

  @pytest.mark.asyncio
  async def test_execute_mcp_tool(self, run_context, sample_tool):
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent(tools=[sample_tool])
    agent._ensure_initialized()
    request = ControlRequest(id="req-2", subtype="tool_call", tool_name="mcp__definable__deploy", input={"branch": "main"})
    response = await agent._handle_control(request, run_context)
    assert "result" in response
    assert "Deployed main" in response["result"]["content"][0]["text"]

  @pytest.mark.asyncio
  async def test_unknown_control_subtype(self, run_context):
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent()
    request = ControlRequest(id="req-3", subtype="unknown_subtype")
    response = await agent._handle_control(request, run_context)
    assert response["behavior"] == "allow"


# ===========================================================================
# Agent config tests
# ===========================================================================


@pytest.mark.unit
class TestAgentConfig:
  def test_default_config(self):
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent()
    assert agent.model == "claude-sonnet-4-6"
    assert agent.permission_mode == "bypassPermissions"
    assert agent.agent_name == "ClaudeCodeAgent"
    assert agent.agent_id is not None

  def test_custom_config(self):
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent(
      model="claude-opus-4-6",
      instructions="Expert coder",
      cwd="/tmp/project",
      max_turns=20,
      max_budget_usd=10.0,
      agent_name="MyCoder",
    )
    assert agent.model == "claude-opus-4-6"
    assert agent.instructions == "Expert coder"
    assert agent.cwd == "/tmp/project"
    assert agent.max_turns == 20
    assert agent.max_budget_usd == 10.0
    assert agent.agent_name == "MyCoder"

  def test_memory_resolution_bool(self):
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent(memory=True)
    agent._ensure_initialized()
    assert agent._memory_manager is not None

  def test_memory_resolution_false(self):
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent(memory=False)
    agent._ensure_initialized()
    assert agent._memory_manager is None

  def test_memory_resolution_config(self):
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.memory.manager import Memory
    from definable.memory.store.in_memory import InMemoryStore

    agent = ClaudeCodeAgent(memory=Memory(store=InMemoryStore()))
    agent._ensure_initialized()
    assert agent._memory_manager is not None

  def test_memory_resolution_manager(self):
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.memory.manager import MemoryManager
    from definable.memory.store.in_memory import InMemoryStore

    mm = MemoryManager(store=InMemoryStore())
    agent = ClaudeCodeAgent(memory=mm)
    agent._ensure_initialized()
    assert agent._memory_manager is mm
