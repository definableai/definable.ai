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
  RunErrorEvent,
  RunOutput,
  RunPausedEvent,
  RunStartedEvent,
  RunStatus,
  ToolCallCompletedEvent,
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
    result = asyncio.run(bridge.execute("mcp__definable__deploy", {"branch": "staging"}))
    assert result["content"][0]["text"] == "Deployed staging"
    assert "isError" not in result

  def test_bridge_async_tool(self, async_tool):
    bridge = ToolBridge(tools=[async_tool])
    result = asyncio.run(bridge.execute("mcp__definable__fetch_data", {"url": "https://example.com"}))
    assert "Data from https://example.com" in result["content"][0]["text"]

  def test_bridge_unknown_tool(self):
    bridge = ToolBridge(tools=[])
    result = asyncio.run(bridge.execute("nonexistent", {}))
    assert result["isError"] is True
    assert "Unknown tool" in result["content"][0]["text"]

  def test_bridge_tool_error(self):
    def bad_tool() -> str:
      """A tool that raises."""
      raise ValueError("intentional error")

    fn = Function.from_callable(bad_tool)
    bridge = ToolBridge(tools=[fn])
    result = asyncio.run(bridge.execute("bad_tool", {}))
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

  def test_framework_tools_injected_when_allowed_tools_set(self):
    """AskUserQuestion + task tools must be injected when --allowedTools is used."""
    from definable.claude_code.agent import ClaudeCodeAgent, _FRAMEWORK_TOOLS

    agent = ClaudeCodeAgent(allowed_tools=["Read", "Write", "Edit", "Bash"])
    agent._ensure_initialized()
    args = agent._build_cli_args("prompt")
    tools_str = args[args.index("--allowedTools") + 1]
    tools_list = tools_str.split(",")
    for fw_tool in _FRAMEWORK_TOOLS:
      assert fw_tool in tools_list, f"{fw_tool} missing from --allowedTools"

  def test_framework_tools_not_duplicated(self):
    """No duplicates when user already lists framework tools."""
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent(allowed_tools=["Read", "AskUserQuestion", "TaskCreate"])
    agent._ensure_initialized()
    args = agent._build_cli_args("prompt")
    tools_str = args[args.index("--allowedTools") + 1]
    tools_list = tools_str.split(",")
    assert tools_list.count("AskUserQuestion") == 1
    assert tools_list.count("TaskCreate") == 1

  def test_no_injection_when_no_allowed_tools(self):
    """No --allowedTools flag at all when user doesn't set allowed_tools."""
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent()
    agent._ensure_initialized()
    args = agent._build_cli_args("prompt")
    assert "--allowedTools" not in args

  def test_framework_tools_injected_with_custom_tools_only(self, sample_tool):
    """Framework tools injected even when only custom @tool functions are registered."""
    from definable.claude_code.agent import ClaudeCodeAgent, _FRAMEWORK_TOOLS

    agent = ClaudeCodeAgent(tools=[sample_tool])
    agent._ensure_initialized()
    args = agent._build_cli_args("prompt")
    assert "--allowedTools" in args
    tools_str = args[args.index("--allowedTools") + 1]
    tools_list = tools_str.split(",")
    assert "mcp__definable__deploy" in tools_list
    for fw_tool in _FRAMEWORK_TOOLS:
      assert fw_tool in tools_list, f"{fw_tool} missing when only custom tools set"


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

  def test_memory_resolution_instance(self):
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.memory.manager import Memory
    from definable.memory.store.in_memory import InMemoryStore

    mem = Memory(store=InMemoryStore())
    agent = ClaudeCodeAgent(memory=mem)
    agent._ensure_initialized()
    assert agent._memory_manager is mem

  def test_session_id_persists_across_instance(self):
    """Instance session_id is stable across multiple _build_cli_args calls."""
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent()
    sid = agent.session_id
    assert sid is not None

    # Multiple accesses should return the same value
    assert agent.session_id == sid
    assert agent.session_id == sid

  def test_session_id_override_respected(self):
    """Explicit session_id passed to constructor is preserved."""
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent(session_id="custom-session-42")
    assert agent.session_id == "custom-session-42"


# ===========================================================================
# Skill integration tests
# ===========================================================================


@pytest.mark.unit
class TestSkillIntegration:
  def test_skill_setup_called_on_init(self):
    """Skill.setup() is called during _ensure_initialized()."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import MagicMock

    skill = MagicMock()
    skill.name = "test_skill"
    skill.tools = []
    skill.setup = MagicMock()
    skill.get_instructions.return_value = "Do stuff"

    agent = ClaudeCodeAgent(skills=[skill])
    agent._ensure_initialized()

    skill.setup.assert_called_once()

  def test_skill_setup_error_does_not_block(self):
    """If skill.setup() raises, initialization continues."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import MagicMock

    skill = MagicMock()
    skill.name = "broken_skill"
    skill.tools = []
    skill.setup.side_effect = RuntimeError("setup boom")

    agent = ClaudeCodeAgent(skills=[skill])
    agent._ensure_initialized()
    # Should still be initialized despite the error
    assert agent._initialized is True

  def test_skill_get_instructions_used(self):
    """System prompt uses skill.get_instructions() not raw .instructions."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import MagicMock

    skill = MagicMock()
    skill.name = "dynamic_skill"
    skill.instructions = "static instructions"
    skill.get_instructions.return_value = "dynamic instructions from get_instructions()"
    skill.tools = []
    skill.setup = MagicMock()

    agent = ClaudeCodeAgent(instructions="Base.", skills=[skill])
    agent._ensure_initialized()
    prompt = agent._build_system_prompt()
    assert "dynamic instructions from get_instructions()" in prompt
    assert "static instructions" not in prompt

  def test_skill_get_instructions_fallback(self):
    """Falls back to getattr(skill, 'instructions') if get_instructions() raises."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import MagicMock

    skill = MagicMock()
    skill.name = "fallback_skill"
    skill.instructions = "fallback instructions"
    skill.get_instructions.side_effect = AttributeError("no method")
    skill.tools = []
    skill.setup = MagicMock()

    agent = ClaudeCodeAgent(instructions="Base.", skills=[skill])
    agent._ensure_initialized()
    prompt = agent._build_system_prompt()
    assert "fallback instructions" in prompt


# ===========================================================================
# Memory session_id / user_id fix tests
# ===========================================================================


@pytest.mark.unit
class TestMemorySessionIdFix:
  @pytest.mark.asyncio
  async def test_memory_recall_uses_context_session_id(self):
    """_memory_recall() uses context.session_id, not hardcoded 'default'."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import AsyncMock

    agent = ClaudeCodeAgent(memory=True)
    agent._ensure_initialized()

    # Mock the memory manager
    mock_manager = AsyncMock()
    mock_manager.get_entries = AsyncMock(return_value=[])
    agent._memory_manager = mock_manager

    context = RunContext(run_id="r1", session_id="my-session", user_id="alice")
    await agent._memory_recall(context)

    mock_manager.get_entries.assert_called_once_with("my-session", "alice", limit=50)

  @pytest.mark.asyncio
  async def test_memory_recall_defaults_user_id(self):
    """_memory_recall() defaults user_id to 'default' when not provided."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import AsyncMock

    agent = ClaudeCodeAgent(memory=True)
    agent._ensure_initialized()

    mock_manager = AsyncMock()
    mock_manager.get_entries = AsyncMock(return_value=[])
    agent._memory_manager = mock_manager

    context = RunContext(run_id="r1", session_id="sess-1", user_id=None)
    await agent._memory_recall(context)

    mock_manager.get_entries.assert_called_once_with("sess-1", "default", limit=50)

  @pytest.mark.asyncio
  async def test_memory_store_uses_context_session_id(self):
    """_memory_store() uses context.session_id, not hardcoded 'default'."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import AsyncMock

    agent = ClaudeCodeAgent(memory=True)
    agent._ensure_initialized()

    mock_manager = AsyncMock()
    mock_manager._ensure_initialized = AsyncMock()
    mock_manager.add = AsyncMock()
    agent._memory_manager = mock_manager

    context = RunContext(run_id="r1", session_id="custom-sess", user_id="bob")
    await agent._memory_store(context, "hello", "world")

    # Both calls should use custom-sess and bob
    assert mock_manager.add.call_count == 2
    for call in mock_manager.add.call_args_list:
      assert call.kwargs["session_id"] == "custom-sess"
      assert call.kwargs["user_id"] == "bob"

  @pytest.mark.asyncio
  async def test_memory_store_defaults_user_id(self):
    """_memory_store() defaults user_id to 'default' when context.user_id is None."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import AsyncMock

    agent = ClaudeCodeAgent(memory=True)
    agent._ensure_initialized()

    mock_manager = AsyncMock()
    mock_manager._ensure_initialized = AsyncMock()
    mock_manager.add = AsyncMock()
    agent._memory_manager = mock_manager

    context = RunContext(run_id="r1", session_id="sess-1", user_id=None)
    await agent._memory_store(context, "hello", "world")

    for call in mock_manager.add.call_args_list:
      assert call.kwargs["user_id"] == "default"

  @pytest.mark.asyncio
  async def test_memory_store_no_user_id_still_works(self):
    """_memory_store() works even without user_id (no longer returns early)."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import AsyncMock

    agent = ClaudeCodeAgent(memory=True)
    agent._ensure_initialized()

    mock_manager = AsyncMock()
    mock_manager._ensure_initialized = AsyncMock()
    mock_manager.add = AsyncMock()
    agent._memory_manager = mock_manager

    context = RunContext(run_id="r1", session_id="sess-1")
    await agent._memory_store(context, "hello", "world")

    # Should still store (with default user_id)
    assert mock_manager.add.call_count == 2


# ===========================================================================
# Knowledge async + format_context tests
# ===========================================================================


@pytest.mark.unit
class TestKnowledgeAsyncFix:
  @pytest.mark.asyncio
  async def test_knowledge_uses_asearch(self):
    """_knowledge_retrieve() calls asearch() (async), not search() (sync)."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import AsyncMock, MagicMock

    mock_doc = MagicMock()
    mock_doc.content = "some knowledge"

    mock_knowledge = MagicMock()
    mock_knowledge.top_k = 3
    mock_knowledge.asearch = AsyncMock(return_value=[mock_doc])
    mock_knowledge.format_context.return_value = "formatted context"

    agent = ClaudeCodeAgent()
    agent._knowledge_instance = mock_knowledge

    context = RunContext(run_id="r1", session_id="s1")
    result = await agent._knowledge_retrieve(context, "query")

    mock_knowledge.asearch.assert_called_once_with("query", top_k=3)
    assert result == "formatted context"

  @pytest.mark.asyncio
  async def test_knowledge_uses_format_context(self):
    """_knowledge_retrieve() delegates formatting to Knowledge.format_context()."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import AsyncMock, MagicMock

    mock_doc = MagicMock()
    mock_doc.content = "doc content"

    mock_knowledge = MagicMock()
    mock_knowledge.top_k = 5
    mock_knowledge.asearch = AsyncMock(return_value=[mock_doc])
    mock_knowledge.format_context.return_value = "<xml>doc content</xml>"

    agent = ClaudeCodeAgent()
    agent._knowledge_instance = mock_knowledge

    context = RunContext(run_id="r1", session_id="s1")
    result = await agent._knowledge_retrieve(context, "test")

    mock_knowledge.format_context.assert_called_once_with([mock_doc])
    assert result == "<xml>doc content</xml>"

  @pytest.mark.asyncio
  async def test_knowledge_fallback_without_format_context(self):
    """Falls back to manual [i] formatting if format_context() is missing."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import AsyncMock, MagicMock

    mock_doc = MagicMock()
    mock_doc.content = "fallback doc"

    mock_knowledge = MagicMock(spec=["top_k", "asearch"])
    mock_knowledge.top_k = 5
    mock_knowledge.asearch = AsyncMock(return_value=[mock_doc])
    # No format_context method on this spec

    agent = ClaudeCodeAgent()
    agent._knowledge_instance = mock_knowledge

    context = RunContext(run_id="r1", session_id="s1")
    result = await agent._knowledge_retrieve(context, "test")

    assert "[1] fallback doc" in result


# ===========================================================================
# Toolkit initialization tests
# ===========================================================================


@pytest.mark.unit
class TestToolkitInitialization:
  @pytest.mark.asyncio
  async def test_toolkit_initialize_called(self):
    """Toolkits with AsyncLifecycleToolkit protocol are initialized."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import AsyncMock, MagicMock

    mock_toolkit = MagicMock()
    mock_toolkit._initialized = False
    mock_toolkit.initialize = AsyncMock()
    mock_toolkit.shutdown = AsyncMock()
    mock_fn = MagicMock(spec=Function)
    mock_fn.name = "toolkit_tool"
    mock_toolkit.tools = [mock_fn]

    agent = ClaudeCodeAgent(toolkits=[mock_toolkit])
    agent._ensure_initialized()
    await agent._ensure_toolkits_initialized()

    mock_toolkit.initialize.assert_called_once()
    assert mock_toolkit in agent._agent_owned_toolkits

  @pytest.mark.asyncio
  async def test_toolkit_tools_merged_into_bridge(self):
    """Toolkit tools are registered in the ToolBridge after initialization."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import AsyncMock, MagicMock

    def my_tool() -> str:
      """A toolkit tool."""
      return "result"

    fn = Function.from_callable(my_tool)

    mock_toolkit = MagicMock()
    mock_toolkit._initialized = False
    mock_toolkit.initialize = AsyncMock()
    mock_toolkit.shutdown = AsyncMock()
    mock_toolkit.tools = [fn]

    agent = ClaudeCodeAgent(toolkits=[mock_toolkit])
    agent._ensure_initialized()
    assert agent._tool_bridge.tool_count == 0  # No direct tools

    await agent._ensure_toolkits_initialized()
    assert agent._tool_bridge.tool_count == 1  # Toolkit tool added
    assert "my_tool" in agent._tool_bridge._tools

  @pytest.mark.asyncio
  async def test_toolkit_already_initialized_skipped(self):
    """Toolkits that are already initialized are not re-initialized."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import AsyncMock, MagicMock

    mock_toolkit = MagicMock()
    mock_toolkit._initialized = True  # Already initialized
    mock_toolkit.initialize = AsyncMock()
    mock_toolkit.tools = []

    agent = ClaudeCodeAgent(toolkits=[mock_toolkit])
    agent._ensure_initialized()
    await agent._ensure_toolkits_initialized()

    mock_toolkit.initialize.assert_not_called()
    assert mock_toolkit not in agent._agent_owned_toolkits


# ===========================================================================
# Lifecycle cleanup tests
# ===========================================================================


@pytest.mark.unit
class TestLifecycleCleanup:
  @pytest.mark.asyncio
  async def test_aexit_calls_skill_teardown(self):
    """__aexit__ calls teardown() on all skills."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import MagicMock

    skill = MagicMock()
    skill.name = "test_skill"
    skill.tools = []
    skill.setup = MagicMock()
    skill.teardown = MagicMock()
    skill.get_instructions.return_value = ""

    agent = ClaudeCodeAgent(skills=[skill])
    async with agent:
      pass

    skill.teardown.assert_called_once()

  @pytest.mark.asyncio
  async def test_aexit_closes_memory(self):
    """__aexit__ calls memory.close()."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import AsyncMock

    agent = ClaudeCodeAgent(memory=True)
    async with agent:
      mock_close = AsyncMock()
      agent._memory_manager.close = mock_close

    mock_close.assert_called_once()

  @pytest.mark.asyncio
  async def test_aexit_shuts_down_toolkits(self):
    """__aexit__ calls shutdown() on agent-owned toolkits."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import AsyncMock, MagicMock

    mock_toolkit = MagicMock()
    mock_toolkit._initialized = False
    mock_toolkit.initialize = AsyncMock()
    mock_toolkit.shutdown = AsyncMock()
    mock_toolkit.tools = []

    agent = ClaudeCodeAgent(toolkits=[mock_toolkit])
    async with agent:
      await agent._ensure_toolkits_initialized()

    mock_toolkit.shutdown.assert_called_once()

  @pytest.mark.asyncio
  async def test_aexit_drains_pending_tasks(self):
    """__aexit__ waits for all pending fire-and-forget tasks."""
    from definable.claude_code.agent import ClaudeCodeAgent

    completed = []

    async def slow_task():
      await asyncio.sleep(0.01)
      completed.append(True)

    agent = ClaudeCodeAgent()
    async with agent:
      task = asyncio.create_task(slow_task())
      agent._pending_tasks.append(task)

    assert len(completed) == 1

  def test_agent_id_set_on_init(self):
    """Agent assigns a unique agent_id on __post_init__."""
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent(agent_id="my-agent")
    assert agent.agent_id == "my-agent"
    # Default generates a uuid-based id
    agent2 = ClaudeCodeAgent()
    assert agent2.agent_id.startswith("claude-code-")


# ===========================================================================
# Bridge dependency injection tests
# ===========================================================================


@pytest.mark.unit
class TestBridgeDependencyInjection:
  def test_skill_deps_injected_into_tools(self):
    """Skill dependencies are merged into each tool's _dependencies."""
    from definable.claude_code.bridge import ToolBridge
    from unittest.mock import MagicMock

    def my_tool() -> str:
      """A tool."""
      return "ok"

    fn = Function.from_callable(my_tool)
    fn._dependencies = {"existing": "dep"}

    skill = MagicMock()
    skill.name = "dep_skill"
    skill.tools = [fn]
    skill.dependencies = {"db_url": "sqlite:///test.db", "api_key": "xxx"}

    bridge = ToolBridge(skills=[skill])
    registered_fn = bridge._tools.get("my_tool")
    assert registered_fn is not None
    assert registered_fn._dependencies["db_url"] == "sqlite:///test.db"
    assert registered_fn._dependencies["api_key"] == "xxx"
    assert registered_fn._dependencies["existing"] == "dep"

  def test_skill_without_deps_still_registers(self):
    """Skills without dependencies still register their tools normally."""
    from definable.claude_code.bridge import ToolBridge
    from unittest.mock import MagicMock

    def my_tool() -> str:
      """A tool."""
      return "ok"

    fn = Function.from_callable(my_tool)

    skill = MagicMock(spec=["name", "tools"])
    skill.name = "no_deps_skill"
    skill.tools = [fn]

    bridge = ToolBridge(skills=[skill])
    assert bridge.tool_count == 1


# ===========================================================================
# ToolServer decorator syntax test
# ===========================================================================


@pytest.mark.unit
class TestToolServerDecoratorSyntax:
  def test_generated_function_uses_decorator_syntax(self):
    """Generated tool function uses @mcp.tool() decorator, not mcp.tool()({name})."""
    from definable.claude_code.tool_server import ToolServer
    from definable.claude_code.bridge import ToolBridge

    def deploy(branch: str = "main") -> str:
      """Deploy the app."""
      return f"Deployed {branch}"

    fn = Function.from_callable(deploy)
    bridge = ToolBridge(tools=[fn])
    server = ToolServer(bridge)

    func_src = server._generate_tool_function("deploy", "Deploy the app.", fn.parameters)
    assert func_src.startswith("@mcp.tool()")
    assert "async def deploy(" in func_src
    # Should NOT contain the old programmatic style
    assert "mcp.tool()(deploy)" not in func_src


# ===========================================================================
# MCP passthrough tests
# ===========================================================================


@pytest.mark.unit
class TestMCPPassthrough:
  def test_mcp_toolkit_skipped_in_toolkit_init(self):
    """MCPToolkit.initialize() is NOT called — it goes through passthrough."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.mcp.toolkit import MCPToolkit
    from definable.mcp.config import MCPConfig, MCPServerConfig
    from unittest.mock import AsyncMock

    toolkit = MCPToolkit(
      config=MCPConfig(
        servers=[
          MCPServerConfig(name="test", transport="http", url="https://example.com/mcp"),
        ]
      )
    )
    toolkit.initialize = AsyncMock()

    agent = ClaudeCodeAgent(toolkits=[toolkit])
    agent._ensure_initialized()
    asyncio.run(agent._ensure_toolkits_initialized())

    toolkit.initialize.assert_not_called()

  def test_collect_mcp_configs_http(self):
    """HTTP transport produces correct JSON config."""
    import json
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.mcp.toolkit import MCPToolkit
    from definable.mcp.config import MCPConfig, MCPServerConfig

    toolkit = MCPToolkit(
      config=MCPConfig(
        servers=[
          MCPServerConfig(
            name="gmail",
            transport="http",
            url="https://backend.composio.dev/mcp",
            headers={"x-api-key": "test-key"},
          ),
        ]
      )
    )

    agent = ClaudeCodeAgent(toolkits=[toolkit])
    paths = agent._collect_mcp_configs()

    assert len(paths) == 1
    with open(paths[0]) as f:
      data = json.load(f)

    assert "mcpServers" in data
    assert "gmail" in data["mcpServers"]
    server = data["mcpServers"]["gmail"]
    assert server["type"] == "http"
    assert server["url"] == "https://backend.composio.dev/mcp"
    assert server["headers"] == {"x-api-key": "test-key"}

    # Cleanup
    agent._cleanup_mcp_temps()

  def test_collect_mcp_configs_stdio(self):
    """stdio transport produces correct JSON config."""
    import json
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.mcp.toolkit import MCPToolkit
    from definable.mcp.config import MCPConfig, MCPServerConfig

    toolkit = MCPToolkit(
      config=MCPConfig(
        servers=[
          MCPServerConfig(
            name="filesystem",
            transport="stdio",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            env={"NODE_PATH": "/usr/local"},
          ),
        ]
      )
    )

    agent = ClaudeCodeAgent(toolkits=[toolkit])
    paths = agent._collect_mcp_configs()

    assert len(paths) == 1
    with open(paths[0]) as f:
      data = json.load(f)

    server = data["mcpServers"]["filesystem"]
    assert server["command"] == "npx"
    assert server["args"] == ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    assert server["env"] == {"NODE_PATH": "/usr/local"}
    assert "type" not in server  # stdio doesn't set "type"

    agent._cleanup_mcp_temps()

  def test_collect_mcp_configs_sse(self):
    """SSE transport produces correct JSON config."""
    import json
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.mcp.toolkit import MCPToolkit
    from definable.mcp.config import MCPConfig, MCPServerConfig

    toolkit = MCPToolkit(
      config=MCPConfig(
        servers=[
          MCPServerConfig(name="web", transport="sse", url="http://localhost:3000/sse"),
        ]
      )
    )

    agent = ClaudeCodeAgent(toolkits=[toolkit])
    paths = agent._collect_mcp_configs()

    assert len(paths) == 1
    with open(paths[0]) as f:
      data = json.load(f)

    server = data["mcpServers"]["web"]
    assert server["type"] == "sse"
    assert server["url"] == "http://localhost:3000/sse"

    agent._cleanup_mcp_temps()

  def test_collect_mcp_configs_empty_without_mcp_toolkit(self):
    """No MCPToolkit in toolkits → empty list."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import MagicMock

    mock_toolkit = MagicMock()
    mock_toolkit.__class__ = type("OtherToolkit", (), {})

    agent = ClaudeCodeAgent(toolkits=[mock_toolkit])
    paths = agent._collect_mcp_configs()
    assert paths == []

  def test_collect_mcp_configs_no_toolkits(self):
    """No toolkits at all → empty list."""
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent()
    paths = agent._collect_mcp_configs()
    assert paths == []

  def test_collect_mcp_configs_multiple_servers(self):
    """Multiple servers in one MCPToolkit produce one config file with all servers."""
    import json
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.mcp.toolkit import MCPToolkit
    from definable.mcp.config import MCPConfig, MCPServerConfig

    toolkit = MCPToolkit(
      config=MCPConfig(
        servers=[
          MCPServerConfig(name="gmail", transport="http", url="https://composio.dev/gmail"),
          MCPServerConfig(name="calendar", transport="http", url="https://composio.dev/calendar"),
        ]
      )
    )

    agent = ClaudeCodeAgent(toolkits=[toolkit])
    paths = agent._collect_mcp_configs()

    assert len(paths) == 1
    with open(paths[0]) as f:
      data = json.load(f)

    assert "gmail" in data["mcpServers"]
    assert "calendar" in data["mcpServers"]

    agent._cleanup_mcp_temps()

  def test_mcp_config_in_cli_args(self):
    """--mcp-config appears in built args when MCP passthrough paths are provided."""
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent()
    args = agent._build_cli_args("system prompt", mcp_config_paths=["/tmp/mcp1.json", "/tmp/mcp2.json"])

    # Should have two --mcp-config entries
    mcp_indices = [i for i, a in enumerate(args) if a == "--mcp-config"]
    assert len(mcp_indices) == 2
    assert args[mcp_indices[0] + 1] == "/tmp/mcp1.json"
    assert args[mcp_indices[1] + 1] == "/tmp/mcp2.json"

  def test_coexistence_custom_tools_and_mcp(self):
    """Both @tool functions and MCPToolkit coexist — bridge has tools, passthrough has config."""
    import json
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.mcp.toolkit import MCPToolkit
    from definable.mcp.config import MCPConfig, MCPServerConfig

    def deploy(branch: str = "main") -> str:
      """Deploy the application."""
      return f"Deployed {branch}"

    fn = Function.from_callable(deploy)

    mcp_toolkit = MCPToolkit(
      config=MCPConfig(
        servers=[
          MCPServerConfig(name="gmail", transport="http", url="https://composio.dev/gmail"),
        ]
      )
    )

    agent = ClaudeCodeAgent(tools=[fn], toolkits=[mcp_toolkit])
    agent._ensure_initialized()

    # Custom tool is in bridge
    assert agent._tool_bridge.tool_count == 1
    assert "deploy" in agent._tool_bridge._tools

    # MCP passthrough produces config files
    paths = agent._collect_mcp_configs()
    assert len(paths) == 1
    with open(paths[0]) as f:
      data = json.load(f)
    assert "gmail" in data["mcpServers"]

    agent._cleanup_mcp_temps()

  def test_cleanup_mcp_temps(self):
    """Temp files are removed after cleanup."""
    import tempfile
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent()

    # Create a real temp file and track it
    fd, path = tempfile.mkstemp(prefix="test_mcp_", suffix=".json")
    import os

    os.close(fd)
    agent._mcp_config_temps.append(path)

    assert os.path.exists(path)
    agent._cleanup_mcp_temps()
    assert not os.path.exists(path)
    assert agent._mcp_config_temps == []


# ===========================================================================
# Streaming (arun_stream) tests
# ===========================================================================


async def _mock_transport_messages(messages_dicts):
  """Helper: yields raw dicts as an async iterator (simulating transport.receive)."""
  for msg in messages_dicts:
    yield msg


@pytest.mark.unit
class TestArunStream:
  @pytest.mark.asyncio
  async def test_stream_yields_content_events(self):
    """arun_stream() yields RunContentEvent for text blocks."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import AsyncMock, patch

    agent = ClaudeCodeAgent(model="claude-sonnet-4-6")

    raw_messages = [
      {
        "type": "assistant",
        "message": {
          "content": [{"type": "text", "text": "Hello world"}],
          "model": "claude-sonnet-4-6",
        },
      },
      {
        "type": "result",
        "subtype": "success",
        "session_id": "sess-1",
        "input_tokens": 100,
        "output_tokens": 50,
      },
    ]

    mock_transport = AsyncMock()
    mock_transport.connect = AsyncMock()
    mock_transport.send = AsyncMock()
    mock_transport.receive = lambda: _mock_transport_messages(raw_messages)
    mock_transport.close = AsyncMock()
    mock_transport.is_running = True

    with patch("definable.claude_code.agent.SubprocessTransport", return_value=mock_transport):
      events = []
      async for event in agent.arun_stream("hi"):
        events.append(event)

    # Should have RunContentEvent + RunCompletedEvent
    content_events = [e for e in events if isinstance(e, RunContentEvent)]
    completed_events = [e for e in events if isinstance(e, RunCompletedEvent)]
    assert len(content_events) == 1
    assert content_events[0].content == "Hello world"
    assert len(completed_events) == 1

  @pytest.mark.asyncio
  async def test_stream_yields_tool_events(self):
    """arun_stream() yields ToolCallStartedEvent for tool use blocks."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import AsyncMock, patch

    agent = ClaudeCodeAgent(model="claude-sonnet-4-6")

    raw_messages = [
      {
        "type": "assistant",
        "message": {
          "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}],
          "model": "claude-sonnet-4-6",
        },
      },
      {
        "type": "result",
        "subtype": "success",
        "session_id": "sess-1",
        "input_tokens": 100,
        "output_tokens": 50,
      },
    ]

    mock_transport = AsyncMock()
    mock_transport.connect = AsyncMock()
    mock_transport.send = AsyncMock()
    mock_transport.receive = lambda: _mock_transport_messages(raw_messages)
    mock_transport.close = AsyncMock()
    mock_transport.is_running = True

    with patch("definable.claude_code.agent.SubprocessTransport", return_value=mock_transport):
      events = []
      async for event in agent.arun_stream("run ls"):
        events.append(event)

    tool_events = [e for e in events if isinstance(e, ToolCallStartedEvent)]
    assert len(tool_events) == 1
    assert tool_events[0].tool.tool_name == "Bash"

  @pytest.mark.asyncio
  async def test_stream_handles_control_requests(self):
    """arun_stream() auto-responds to can_use_tool control requests."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import AsyncMock, patch

    agent = ClaudeCodeAgent(model="claude-sonnet-4-6")

    raw_messages = [
      {
        "type": "control_request",
        "id": "req-1",
        "subtype": "can_use_tool",
        "tool_name": "Read",
        "input": {"file_path": "/a.py"},
      },
      {
        "type": "assistant",
        "message": {
          "content": [{"type": "text", "text": "Done"}],
          "model": "claude-sonnet-4-6",
        },
      },
      {
        "type": "result",
        "subtype": "success",
        "session_id": "sess-1",
        "input_tokens": 100,
        "output_tokens": 50,
      },
    ]

    mock_transport = AsyncMock()
    mock_transport.connect = AsyncMock()
    mock_transport.send = AsyncMock()
    mock_transport.receive = lambda: _mock_transport_messages(raw_messages)
    mock_transport.close = AsyncMock()
    mock_transport.is_running = True

    with patch("definable.claude_code.agent.SubprocessTransport", return_value=mock_transport):
      events = []
      async for event in agent.arun_stream("read file"):
        events.append(event)

    # Should have sent the allow response
    send_calls = mock_transport.send.call_args_list
    # First call: user message, second call: control response
    control_resp = send_calls[1][0][0]
    assert control_resp["behavior"] == "allow"
    assert control_resp["id"] == "req-1"

  @pytest.mark.asyncio
  async def test_stream_input_guardrail_blocks(self):
    """arun_stream() yields RunCompletedEvent with blocked content when input guardrail fires."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.agent.guardrail import Guardrails, input_guardrail, GuardrailResult

    @input_guardrail
    async def block_all(text, context):
      return GuardrailResult(action="block", message="No prompts allowed")

    agent = ClaudeCodeAgent(
      model="claude-sonnet-4-6",
      guardrails=Guardrails(input=[block_all]),
    )

    events = []
    async for event in agent.arun_stream("anything"):
      events.append(event)

    assert len(events) == 1
    assert isinstance(events[0], RunCompletedEvent)
    assert "blocked" in (events[0].content or "").lower()

  @pytest.mark.asyncio
  async def test_stream_connect_failure(self):
    """arun_stream() yields RunErrorEvent when CLI fails to start."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.agent.events import RunErrorEvent, RunStartedEvent
    from unittest.mock import AsyncMock, patch

    agent = ClaudeCodeAgent(model="claude-sonnet-4-6")

    mock_transport = AsyncMock()
    mock_transport.connect = AsyncMock(side_effect=FileNotFoundError("claude not found"))
    mock_transport.close = AsyncMock()

    with patch("definable.claude_code.agent.SubprocessTransport", return_value=mock_transport):
      events = []
      async for event in agent.arun_stream("hi"):
        events.append(event)

    assert len(events) == 2
    assert isinstance(events[0], RunStartedEvent)
    assert isinstance(events[1], RunErrorEvent)
    assert "not found" in (events[1].content or "")


# ===========================================================================
# HITL (confirm_tools) tests
# ===========================================================================


@pytest.mark.unit
class TestHITL:
  @pytest.mark.asyncio
  async def test_handle_control_hitl_pause(self, run_context):
    """_handle_control returns RunPausedEvent when tool matches confirm_tools."""
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent(confirm_tools=["Bash", "Write"])
    request = ControlRequest(id="req-1", subtype="can_use_tool", tool_name="Bash", input={"command": "rm -rf /"})
    response = await agent._handle_control(request, run_context)
    assert isinstance(response, RunPausedEvent)
    assert len(response.requirements) == 1
    assert response.requirements[0].tool_execution.tool_name == "Bash"
    assert response.requirements[0].tool_execution.tool_args == {"command": "rm -rf /"}

  @pytest.mark.asyncio
  async def test_handle_control_hitl_no_match(self, run_context):
    """_handle_control auto-allows when tool NOT in confirm_tools."""
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent(confirm_tools=["Bash", "Write"])
    request = ControlRequest(id="req-1", subtype="can_use_tool", tool_name="Read", input={})
    response = await agent._handle_control(request, run_context)
    assert isinstance(response, dict)
    assert response["behavior"] == "allow"

  @pytest.mark.asyncio
  async def test_handle_control_guardrail_takes_precedence(self, run_context):
    """Guardrail deny takes precedence over HITL pause."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.agent.guardrail import Guardrails, tool_blocklist

    agent = ClaudeCodeAgent(
      confirm_tools=["Bash"],
      guardrails=Guardrails(tool=[tool_blocklist({"Bash"})]),
    )
    request = ControlRequest(id="req-1", subtype="can_use_tool", tool_name="Bash", input={})
    response = await agent._handle_control(request, run_context)
    # Should be deny (guardrail), NOT RunPausedEvent
    assert isinstance(response, dict)
    assert response["behavior"] == "deny"

  @pytest.mark.asyncio
  async def test_arun_pauses_on_confirm_tool(self):
    """arun() returns paused RunOutput when confirm_tools matches."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import AsyncMock, patch

    agent = ClaudeCodeAgent(model="claude-sonnet-4-6", confirm_tools=["Bash"])

    raw_messages = [
      {
        "type": "control_request",
        "id": "req-1",
        "subtype": "can_use_tool",
        "tool_name": "Bash",
        "input": {"command": "ls"},
      },
    ]

    mock_transport = AsyncMock()
    mock_transport.connect = AsyncMock()
    mock_transport.send = AsyncMock()
    mock_transport.receive = lambda: _mock_transport_messages(raw_messages)
    mock_transport.close = AsyncMock()
    mock_transport.is_running = True

    with patch("definable.claude_code.agent.SubprocessTransport", return_value=mock_transport):
      result = await agent.arun("run ls")

    assert result.is_paused
    assert result.status == RunStatus.paused
    assert result.requirements is not None
    assert len(result.requirements) == 1
    assert result.requirements[0].tool_execution.tool_name == "Bash"
    # Transport should NOT be closed (still alive for continue_run)
    mock_transport.close.assert_not_called()

  @pytest.mark.asyncio
  async def test_arun_stream_pauses_on_confirm_tool(self):
    """arun_stream() yields RunPausedEvent when confirm_tools matches."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import AsyncMock, patch

    agent = ClaudeCodeAgent(model="claude-sonnet-4-6", confirm_tools=["Write"])

    raw_messages = [
      {
        "type": "control_request",
        "id": "req-1",
        "subtype": "can_use_tool",
        "tool_name": "Write",
        "input": {"file_path": "/tmp/x.py", "content": "pass"},
      },
    ]

    mock_transport = AsyncMock()
    mock_transport.connect = AsyncMock()
    mock_transport.send = AsyncMock()
    mock_transport.receive = lambda: _mock_transport_messages(raw_messages)
    mock_transport.close = AsyncMock()
    mock_transport.is_running = True

    with patch("definable.claude_code.agent.SubprocessTransport", return_value=mock_transport):
      events = []
      async for event in agent.arun_stream("write file"):
        events.append(event)

    assert len(events) == 2
    assert isinstance(events[0], RunStartedEvent)
    assert isinstance(events[1], RunPausedEvent)
    assert events[1].requirements[0].tool_execution.tool_name == "Write"
    mock_transport.close.assert_not_called()

  @pytest.mark.asyncio
  async def test_continue_run_approved(self):
    """continue_run() resumes with allow after user confirms."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import AsyncMock, patch

    agent = ClaudeCodeAgent(model="claude-sonnet-4-6", confirm_tools=["Bash"])

    # Phase 1: pause
    pause_messages = [
      {
        "type": "control_request",
        "id": "req-1",
        "subtype": "can_use_tool",
        "tool_name": "Bash",
        "input": {"command": "ls"},
      },
    ]

    mock_transport = AsyncMock()
    mock_transport.connect = AsyncMock()
    mock_transport.send = AsyncMock()
    mock_transport.receive = lambda: _mock_transport_messages(pause_messages)
    mock_transport.close = AsyncMock()
    mock_transport.is_running = True

    with patch("definable.claude_code.agent.SubprocessTransport", return_value=mock_transport):
      result = await agent.arun("run ls")

    assert result.is_paused

    # Phase 2: user confirms and we continue
    resume_messages = [
      {
        "type": "assistant",
        "message": {
          "content": [{"type": "text", "text": "Done listing files"}],
          "model": "claude-sonnet-4-6",
        },
      },
      {
        "type": "result",
        "subtype": "success",
        "session_id": "sess-1",
        "input_tokens": 200,
        "output_tokens": 100,
      },
    ]

    # Swap the receive method for resume messages
    mock_transport.receive = lambda: _mock_transport_messages(resume_messages)

    for req in result.requirements:
      req.confirm()

    result2 = await agent.continue_run(run_output=result)
    assert result2.status == RunStatus.completed
    assert result2.content == "Done listing files"

    # Should have sent allow control response
    send_calls = mock_transport.send.call_args_list
    # Last send before resume messages should be the control response
    control_resp = send_calls[-1][0][0]
    assert control_resp["behavior"] == "allow"

  @pytest.mark.asyncio
  async def test_continue_run_rejected(self):
    """continue_run() sends deny when user rejects."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import AsyncMock, patch

    agent = ClaudeCodeAgent(model="claude-sonnet-4-6", confirm_tools=["Bash"])

    pause_messages = [
      {
        "type": "control_request",
        "id": "req-1",
        "subtype": "can_use_tool",
        "tool_name": "Bash",
        "input": {"command": "rm -rf /"},
      },
    ]

    mock_transport = AsyncMock()
    mock_transport.connect = AsyncMock()
    mock_transport.send = AsyncMock()
    mock_transport.receive = lambda: _mock_transport_messages(pause_messages)
    mock_transport.close = AsyncMock()
    mock_transport.is_running = True

    with patch("definable.claude_code.agent.SubprocessTransport", return_value=mock_transport):
      result = await agent.arun("delete everything")

    assert result.is_paused

    resume_messages = [
      {
        "type": "assistant",
        "message": {
          "content": [{"type": "text", "text": "Operation cancelled."}],
          "model": "claude-sonnet-4-6",
        },
      },
      {
        "type": "result",
        "subtype": "success",
        "session_id": "sess-1",
        "input_tokens": 200,
        "output_tokens": 50,
      },
    ]

    mock_transport.receive = lambda: _mock_transport_messages(resume_messages)

    for req in result.requirements:
      req.reject()

    await agent.continue_run(run_output=result)

    # Check deny was sent
    send_calls = mock_transport.send.call_args_list
    control_resp = send_calls[-1][0][0]
    assert control_resp["behavior"] == "deny"

  @pytest.mark.asyncio
  async def test_continue_run_not_paused_raises(self):
    """continue_run() raises ValueError when RunOutput is not paused."""
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent()
    result = RunOutput(
      run_id="r1",
      session_id="s1",
      status=RunStatus.completed,
    )

    with pytest.raises(ValueError, match="not paused"):
      await agent.continue_run(run_output=result)

  @pytest.mark.asyncio
  async def test_continue_run_no_state_raises(self):
    """continue_run() raises ValueError when no HITL state exists."""
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent()
    result = RunOutput(
      run_id="nonexistent",
      session_id="s1",
      status=RunStatus.paused,
      requirements=[],
    )

    with pytest.raises(ValueError, match="No HITL state"):
      await agent.continue_run(run_output=result)

  @pytest.mark.asyncio
  async def test_continue_run_stream_approved(self):
    """continue_run_stream() yields events after user confirms."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import AsyncMock, patch

    agent = ClaudeCodeAgent(model="claude-sonnet-4-6", confirm_tools=["Bash"])

    pause_messages = [
      {
        "type": "control_request",
        "id": "req-1",
        "subtype": "can_use_tool",
        "tool_name": "Bash",
        "input": {"command": "ls"},
      },
    ]

    mock_transport = AsyncMock()
    mock_transport.connect = AsyncMock()
    mock_transport.send = AsyncMock()
    mock_transport.receive = lambda: _mock_transport_messages(pause_messages)
    mock_transport.close = AsyncMock()
    mock_transport.is_running = True

    with patch("definable.claude_code.agent.SubprocessTransport", return_value=mock_transport):
      events = []
      async for event in agent.arun_stream("run ls"):
        events.append(event)

    assert len(events) == 2
    assert isinstance(events[0], RunStartedEvent)
    assert isinstance(events[1], RunPausedEvent)

    resume_messages = [
      {
        "type": "assistant",
        "message": {
          "content": [{"type": "text", "text": "files listed"}],
          "model": "claude-sonnet-4-6",
        },
      },
      {
        "type": "result",
        "subtype": "success",
        "session_id": "sess-1",
        "input_tokens": 200,
        "output_tokens": 100,
      },
    ]

    mock_transport.receive = lambda: _mock_transport_messages(resume_messages)

    paused_event = events[1]
    for req in paused_event.requirements:
      req.confirm()

    resumed_events = []
    async for event in agent.continue_run_stream(run_output=paused_event):
      resumed_events.append(event)

    content_events = [e for e in resumed_events if isinstance(e, RunContentEvent)]
    completed_events = [e for e in resumed_events if isinstance(e, RunCompletedEvent)]
    assert len(content_events) == 1
    assert content_events[0].content == "files listed"
    assert len(completed_events) == 1

  @pytest.mark.asyncio
  async def test_hitl_state_cleaned_up_on_aexit(self):
    """__aexit__ closes dangling HITL transports."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import AsyncMock, patch

    agent = ClaudeCodeAgent(model="claude-sonnet-4-6", confirm_tools=["Bash"])

    pause_messages = [
      {
        "type": "control_request",
        "id": "req-1",
        "subtype": "can_use_tool",
        "tool_name": "Bash",
        "input": {"command": "ls"},
      },
    ]

    mock_transport = AsyncMock()
    mock_transport.connect = AsyncMock()
    mock_transport.send = AsyncMock()
    mock_transport.receive = lambda: _mock_transport_messages(pause_messages)
    mock_transport.close = AsyncMock()
    mock_transport.is_running = True

    with patch("definable.claude_code.agent.SubprocessTransport", return_value=mock_transport):
      async with agent:
        result = await agent.arun("run ls")
        assert result.is_paused
        # Don't call continue_run — abandon the paused state

    # __aexit__ should have closed the dangling transport
    mock_transport.close.assert_called()

  @pytest.mark.asyncio
  async def test_confirm_tools_none_auto_allows(self, run_context):
    """When confirm_tools is None, all tools auto-allow (original behavior)."""
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent()
    request = ControlRequest(id="req-1", subtype="can_use_tool", tool_name="Bash", input={})
    response = await agent._handle_control(request, run_context)
    assert isinstance(response, dict)
    assert response["behavior"] == "allow"

  @pytest.mark.asyncio
  async def test_confirm_tools_empty_list_auto_allows(self, run_context):
    """When confirm_tools is empty list, all tools auto-allow."""
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent(confirm_tools=[])
    request = ControlRequest(id="req-1", subtype="can_use_tool", tool_name="Bash", input={})
    response = await agent._handle_control(request, run_context)
    assert isinstance(response, dict)
    assert response["behavior"] == "allow"

  @pytest.mark.asyncio
  async def test_arun_no_hitl_still_works(self):
    """arun() works normally when confirm_tools is not set."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import AsyncMock, patch

    agent = ClaudeCodeAgent(model="claude-sonnet-4-6")

    raw_messages = [
      {
        "type": "control_request",
        "id": "req-1",
        "subtype": "can_use_tool",
        "tool_name": "Bash",
        "input": {"command": "ls"},
      },
      {
        "type": "assistant",
        "message": {
          "content": [{"type": "text", "text": "Done"}],
          "model": "claude-sonnet-4-6",
        },
      },
      {
        "type": "result",
        "subtype": "success",
        "session_id": "sess-1",
        "input_tokens": 100,
        "output_tokens": 50,
      },
    ]

    mock_transport = AsyncMock()
    mock_transport.connect = AsyncMock()
    mock_transport.send = AsyncMock()
    mock_transport.receive = lambda: _mock_transport_messages(raw_messages)
    mock_transport.close = AsyncMock()
    mock_transport.is_running = True

    with patch("definable.claude_code.agent.SubprocessTransport", return_value=mock_transport):
      result = await agent.arun("run ls")

    assert result.status == RunStatus.completed
    assert result.content == "Done"
    mock_transport.close.assert_called()

  @pytest.mark.asyncio
  async def test_stream_with_memory(self):
    """arun_stream() stores memory after completion."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from unittest.mock import AsyncMock, patch

    agent = ClaudeCodeAgent(model="claude-sonnet-4-6", memory=True)
    agent._ensure_initialized()

    mock_manager = AsyncMock()
    mock_manager._ensure_initialized = AsyncMock()
    mock_manager.add = AsyncMock()
    mock_manager.get_entries = AsyncMock(return_value=[])
    agent._memory_manager = mock_manager

    raw_messages = [
      {
        "type": "assistant",
        "message": {
          "content": [{"type": "text", "text": "memory test"}],
          "model": "claude-sonnet-4-6",
        },
      },
      {
        "type": "result",
        "subtype": "success",
        "session_id": "sess-1",
        "input_tokens": 100,
        "output_tokens": 50,
      },
    ]

    mock_transport = AsyncMock()
    mock_transport.connect = AsyncMock()
    mock_transport.send = AsyncMock()
    mock_transport.receive = lambda: _mock_transport_messages(raw_messages)
    mock_transport.close = AsyncMock()
    mock_transport.is_running = True

    with patch("definable.claude_code.agent.SubprocessTransport", return_value=mock_transport):
      events = []
      async for event in agent.arun_stream("test prompt"):
        events.append(event)

    # Give fire-and-forget task time to complete
    await asyncio.sleep(0.05)
    # Memory recall + store should have been called
    mock_manager.get_entries.assert_called_once()

  @pytest.mark.asyncio
  async def test_stream_reasoning_events(self):
    """arun_stream() yields ReasoningContentDeltaEvent for thinking blocks."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.agent.events import ReasoningContentDeltaEvent
    from unittest.mock import AsyncMock, patch

    agent = ClaudeCodeAgent(model="claude-sonnet-4-6")

    raw_messages = [
      {
        "type": "assistant",
        "message": {
          "content": [
            {"type": "thinking", "thinking": "Let me analyze...", "signature": "s1"},
            {"type": "text", "text": "Here is my answer."},
          ],
          "model": "claude-sonnet-4-6",
        },
      },
      {
        "type": "result",
        "subtype": "success",
        "session_id": "sess-1",
        "input_tokens": 100,
        "output_tokens": 50,
      },
    ]

    mock_transport = AsyncMock()
    mock_transport.connect = AsyncMock()
    mock_transport.send = AsyncMock()
    mock_transport.receive = lambda: _mock_transport_messages(raw_messages)
    mock_transport.close = AsyncMock()
    mock_transport.is_running = True

    with patch("definable.claude_code.agent.SubprocessTransport", return_value=mock_transport):
      events = []
      async for event in agent.arun_stream("think about this"):
        events.append(event)

    reasoning_events = [e for e in events if isinstance(e, ReasoningContentDeltaEvent)]
    assert len(reasoning_events) == 1
    assert reasoning_events[0].reasoning_content == "Let me analyze..."


# ===========================================================================
# ClaudeCodeCLI tests
# ===========================================================================


@pytest.mark.unit
class TestClaudeCodeCLI:
  def test_cli_init_defaults(self):
    """CLI initializes with default settings."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.claude_code.cli import ClaudeCodeCLI

    agent = ClaudeCodeAgent(model="claude-sonnet-4-6")
    cli = ClaudeCodeCLI(agent=agent)
    assert cli.show_banner is True
    assert cli.show_thinking is True
    assert cli.show_tool_args is True
    assert cli.show_tool_results is True
    assert cli.show_metrics is True
    assert cli.max_content_display == 500
    assert cli._console is not None

  def test_cli_custom_settings(self):
    """CLI respects custom settings."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.claude_code.cli import ClaudeCodeCLI

    agent = ClaudeCodeAgent()
    cli = ClaudeCodeCLI(
      agent=agent,
      show_thinking=False,
      show_tool_results=False,
      max_content_display=100,
    )
    assert cli.show_thinking is False
    assert cli.show_tool_results is False
    assert cli.max_content_display == 100


@pytest.mark.unit
class TestCLIRendering:
  def test_render_content_event(self, capsys):
    """RunContentEvent prints raw content to stdout."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.claude_code.cli import ClaudeCodeCLI

    agent = ClaudeCodeAgent()
    cli = ClaudeCodeCLI(agent=agent)
    event = RunContentEvent(
      agent_id="a",
      agent_name="n",
      run_id="r",
      session_id="s",
      content="Hello world",
    )
    cli._render_event(event)
    captured = capsys.readouterr()
    assert "Hello world" in captured.out

  def test_render_tool_started_with_args(self):
    """ToolCallStartedEvent renders readable tool header."""
    from io import StringIO
    from rich.console import Console
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.claude_code.cli import ClaudeCodeCLI
    from definable.model.response import ToolExecution

    agent = ClaudeCodeAgent()
    cli = ClaudeCodeCLI(agent=agent, show_tool_args=True)
    buf = StringIO()
    cli._console = Console(file=buf, force_terminal=True)

    event = ToolCallStartedEvent(
      agent_id="a",
      agent_name="n",
      run_id="r",
      session_id="s",
      tool=ToolExecution(tool_name="Bash", tool_args={"command": "ls -la"}),
    )
    cli._render_event(event)
    output = buf.getvalue()
    assert "Running" in output
    assert "ls -la" in output

  def test_render_tool_started_without_args(self):
    """ToolCallStartedEvent always shows readable header including primary arg."""
    from io import StringIO
    from rich.console import Console
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.claude_code.cli import ClaudeCodeCLI
    from definable.model.response import ToolExecution

    agent = ClaudeCodeAgent()
    cli = ClaudeCodeCLI(agent=agent, show_tool_args=False)
    buf = StringIO()
    cli._console = Console(file=buf, no_color=True)

    event = ToolCallStartedEvent(
      agent_id="a",
      agent_name="n",
      run_id="r",
      session_id="s",
      tool=ToolExecution(tool_name="Read", tool_args={"file_path": "/secret.txt"}),
    )
    cli._render_event(event)
    output = buf.getvalue()
    # Header always shows readable description with primary arg
    assert "Reading" in output
    assert "/secret.txt" in output

  def test_render_tool_completed_shown(self):
    """ToolCallCompletedEvent renders result when show_tool_results=True."""
    from io import StringIO
    from rich.console import Console
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.claude_code.cli import ClaudeCodeCLI

    agent = ClaudeCodeAgent()
    cli = ClaudeCodeCLI(agent=agent, show_tool_results=True)
    buf = StringIO()
    cli._console = Console(file=buf, force_terminal=True)

    event = ToolCallCompletedEvent(
      agent_id="a",
      agent_name="n",
      run_id="r",
      session_id="s",
      content="file contents here",
    )
    cli._render_event(event)
    output = buf.getvalue()
    assert "file contents here" in output

  def test_render_tool_completed_hidden(self):
    """ToolCallCompletedEvent is suppressed when show_tool_results=False."""
    from io import StringIO
    from rich.console import Console
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.claude_code.cli import ClaudeCodeCLI

    agent = ClaudeCodeAgent()
    cli = ClaudeCodeCLI(agent=agent, show_tool_results=False)
    buf = StringIO()
    cli._console = Console(file=buf, force_terminal=True)

    event = ToolCallCompletedEvent(
      agent_id="a",
      agent_name="n",
      run_id="r",
      session_id="s",
      content="should not appear",
    )
    cli._render_event(event)
    output = buf.getvalue()
    assert output.strip() == ""

  def test_render_reasoning_shown(self):
    """ReasoningContentDeltaEvent renders when show_thinking=True."""
    from io import StringIO
    from rich.console import Console
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.claude_code.cli import ClaudeCodeCLI

    agent = ClaudeCodeAgent()
    cli = ClaudeCodeCLI(agent=agent, show_thinking=True)
    buf = StringIO()
    cli._console = Console(file=buf, force_terminal=True)

    event = ReasoningContentDeltaEvent(
      agent_id="a",
      agent_name="n",
      run_id="r",
      session_id="s",
      reasoning_content="Let me think about this...",
    )
    cli._render_event(event)
    output = buf.getvalue()
    assert "Let me think about this" in output

  def test_render_reasoning_hidden(self):
    """ReasoningContentDeltaEvent suppressed when show_thinking=False."""
    from io import StringIO
    from rich.console import Console
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.claude_code.cli import ClaudeCodeCLI

    agent = ClaudeCodeAgent()
    cli = ClaudeCodeCLI(agent=agent, show_thinking=False)
    buf = StringIO()
    cli._console = Console(file=buf, force_terminal=True)

    event = ReasoningContentDeltaEvent(
      agent_id="a",
      agent_name="n",
      run_id="r",
      session_id="s",
      reasoning_content="should not appear",
    )
    cli._render_event(event)
    output = buf.getvalue()
    assert output.strip() == ""

  def test_render_metrics(self):
    """RunCompletedEvent renders compact metrics line."""
    from io import StringIO
    from rich.console import Console
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.claude_code.cli import ClaudeCodeCLI
    from definable.model.metrics import Metrics

    agent = ClaudeCodeAgent()
    cli = ClaudeCodeCLI(agent=agent, show_metrics=True)
    buf = StringIO()
    cli._console = Console(file=buf, no_color=True)

    event = RunCompletedEvent(
      agent_id="a",
      agent_name="n",
      run_id="r",
      session_id="s",
      content="done",
      metrics=Metrics(total_tokens=2500, cost=0.045, duration=15.0),
    )
    cli._render_event(event)
    output = buf.getvalue()
    assert "2.5k tokens" in output
    assert "$0.0450" in output
    assert "15.0s" in output

  def test_render_error_event(self):
    """RunErrorEvent renders a red error panel."""
    from io import StringIO
    from rich.console import Console
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.claude_code.cli import ClaudeCodeCLI

    agent = ClaudeCodeAgent()
    cli = ClaudeCodeCLI(agent=agent)
    buf = StringIO()
    cli._console = Console(file=buf, force_terminal=True)

    event = RunErrorEvent(
      agent_id="a",
      agent_name="n",
      run_id="r",
      session_id="s",
      content="something broke",
    )
    cli._render_event(event)
    output = buf.getvalue()
    assert "something broke" in output
    assert "Error" in output


@pytest.mark.unit
class TestCLICommands:
  def test_command_quit(self):
    """'/quit' returns True (exit signal)."""
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.claude_code.cli import ClaudeCodeCLI

    agent = ClaudeCodeAgent()
    cli = ClaudeCodeCLI(agent=agent)
    assert cli._handle_command("/quit") is True
    assert cli._handle_command("/exit") is True
    assert cli._handle_command("/q") is True

  def test_command_help(self):
    """'/help' does not exit."""
    from io import StringIO
    from rich.console import Console
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.claude_code.cli import ClaudeCodeCLI

    agent = ClaudeCodeAgent()
    cli = ClaudeCodeCLI(agent=agent)
    buf = StringIO()
    cli._console = Console(file=buf, no_color=True)

    assert cli._handle_command("/help") is False
    output = buf.getvalue()
    assert "/quit" in output

  def test_command_info(self):
    """'/info' prints agent config."""
    from io import StringIO
    from rich.console import Console
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.claude_code.cli import ClaudeCodeCLI

    agent = ClaudeCodeAgent(model="claude-opus-4-6", confirm_tools=["Bash"])
    cli = ClaudeCodeCLI(agent=agent)
    buf = StringIO()
    cli._console = Console(file=buf, force_terminal=True)

    assert cli._handle_command("/info") is False
    output = buf.getvalue()
    assert "claude-opus-4-6" in output

  def test_command_unknown(self):
    """Unknown command prints error."""
    from io import StringIO
    from rich.console import Console
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.claude_code.cli import ClaudeCodeCLI

    agent = ClaudeCodeAgent()
    cli = ClaudeCodeCLI(agent=agent)
    buf = StringIO()
    cli._console = Console(file=buf, force_terminal=True)

    assert cli._handle_command("/bogus") is False
    output = buf.getvalue()
    assert "Unknown command" in output


@pytest.mark.unit
class TestCLIHelpers:
  def test_truncate_short(self):
    from definable.claude_code.cli import _truncate

    assert _truncate("hello", 100) == "hello"

  def test_truncate_long(self):
    from definable.claude_code.cli import _truncate

    result = _truncate("a" * 200, 50)
    assert len(result) == 50
    assert result.endswith("...")

  def test_format_token_count_small(self):
    from definable.claude_code.cli import _format_token_count

    assert _format_token_count(150) == "150"

  def test_format_token_count_large(self):
    from definable.claude_code.cli import _format_token_count

    assert _format_token_count(2500) == "2.5k"
    assert _format_token_count(10000) == "10.0k"


@pytest.mark.unit
class TestToolHeaders:
  def test_format_tool_header_read(self):
    """Read tool shows 'Reading /path'."""
    from definable.claude_code.cli import _format_tool_header

    result = _format_tool_header("Read", {"file_path": "/src/auth.py"})
    assert result == "Reading /src/auth.py"

  def test_format_tool_header_bash(self):
    """Bash tool shows 'Running `command`'."""
    from definable.claude_code.cli import _format_tool_header

    result = _format_tool_header("Bash", {"command": "npm test"})
    assert result == "Running `npm test`"

  def test_format_tool_header_unknown(self):
    """Unknown tool shows name(key=value)."""
    from definable.claude_code.cli import _format_tool_header

    result = _format_tool_header("MyTool", {"arg": "value"})
    assert result == "MyTool(arg=value)"

  def test_format_tool_header_no_args(self):
    """Known tool without args shows just the verb."""
    from definable.claude_code.cli import _format_tool_header

    result = _format_tool_header("Read", None)
    assert result == "Reading"

  def test_format_tool_header_mcp_prefix(self):
    """MCP tools strip mcp__ prefix."""
    from definable.claude_code.cli import _format_tool_header

    result = _format_tool_header("mcp__definable__search", {"query": "auth"})
    assert "definable:search" in result


@pytest.mark.unit
class TestToolResultFormat:
  def test_format_long_result(self):
    """Results >10 lines are truncated with 'more lines' indicator."""
    from definable.claude_code.cli import _format_tool_result

    lines = [f"line {i}" for i in range(20)]
    text = "\n".join(lines)
    result = _format_tool_result(text)
    assert "more lines" in result
    # First 5 lines should be present
    assert "line 0" in result
    assert "line 4" in result
    # Last 2 lines should be present
    assert "line 18" in result
    assert "line 19" in result

  def test_format_empty_result(self):
    """Empty content returns empty string."""
    from definable.claude_code.cli import _format_tool_result

    assert _format_tool_result(None) == ""
    assert _format_tool_result("") == ""
    assert _format_tool_result("   ") == ""


@pytest.mark.unit
class TestCLIBanner:
  def test_banner_prints(self):
    """Banner renders agent model and instructions."""
    from io import StringIO
    from rich.console import Console
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.claude_code.cli import ClaudeCodeCLI

    agent = ClaudeCodeAgent(
      model="claude-sonnet-4-6",
      instructions="Expert Python dev.",
      confirm_tools=["Bash", "Write"],
    )
    cli = ClaudeCodeCLI(agent=agent)
    buf = StringIO()
    cli._console = Console(file=buf, force_terminal=True)

    cli._print_banner()
    output = buf.getvalue()
    assert "claude-sonnet-4-6" in output
    assert "Expert Python dev" in output
    assert "Bash" in output
    assert "Write" in output


@pytest.mark.unit
class TestTaskPanel:
  """Tests for the sticky real-time task panel."""

  def _make_cli(self):
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.claude_code.cli import ClaudeCodeCLI

    agent = ClaudeCodeAgent()
    cli = ClaudeCodeCLI(agent=agent)
    return cli

  def _make_console(self):
    from io import StringIO
    from rich.console import Console

    buf = StringIO()
    console = Console(file=buf, no_color=True)
    return console, buf

  def test_track_todo_write(self):
    """TodoWrite replaces all tracked tasks with correct subjects/statuses."""
    cli = self._make_cli()
    cli._track_todo_write({
      "todos": [
        {"content": "Research API design", "status": "completed"},
        {"content": "Implement rate limiter", "status": "in_progress", "activeForm": "Implementing..."},
        {"content": "Write tests", "status": "pending"},
      ]
    })
    assert len(cli._tracked_tasks) == 3
    assert cli._tracked_tasks[0].subject == "Research API design"
    assert cli._tracked_tasks[0].status == "completed"
    assert cli._tracked_tasks[1].status == "in_progress"
    assert cli._tracked_tasks[1].active_form == "Implementing..."
    assert cli._tracked_tasks[2].status == "pending"

  def test_track_todo_write_empty(self):
    """TodoWrite with empty todos clears tracked tasks."""
    from definable.claude_code.cli import _TrackedTask

    cli = self._make_cli()
    cli._tracked_tasks.append(_TrackedTask(id="1", subject="old"))
    cli._track_todo_write({"todos": []})
    assert len(cli._tracked_tasks) == 0

  def test_track_task_create(self):
    """TaskCreate appends a new task with auto-incremented ID."""
    cli = self._make_cli()
    cli._track_task_create({"subject": "First task"})
    cli._track_task_create({"subject": "Second task"})
    assert len(cli._tracked_tasks) == 2
    assert cli._tracked_tasks[0].id == "1"
    assert cli._tracked_tasks[0].subject == "First task"
    assert cli._tracked_tasks[1].id == "2"
    assert cli._tracked_tasks[1].subject == "Second task"

  def test_track_task_update(self):
    """TaskUpdate finds task by ID and updates status."""
    cli = self._make_cli()
    cli._track_task_create({"subject": "Research"})
    cli._track_task_update({"taskId": "1", "status": "completed"})
    assert cli._tracked_tasks[0].status == "completed"

  def test_track_task_update_unknown_id(self):
    """TaskUpdate with unknown ID is a no-op."""
    cli = self._make_cli()
    cli._track_task_create({"subject": "Research"})
    cli._track_task_update({"taskId": "99", "status": "completed"})
    assert cli._tracked_tasks[0].status == "pending"  # unchanged

  def test_render_panel_completed_strikethrough(self):
    """Completed tasks render with ANSI strikethrough escape sequence."""
    import sys
    from unittest.mock import patch

    cli = self._make_cli()
    cli._track_task_create({"subject": "Done task"})
    cli._track_task_update({"taskId": "1", "status": "completed"})

    captured = []
    with patch.object(sys, "stdout") as mock_stdout:
      mock_stdout.isatty.return_value = True
      mock_stdout.write = lambda s: captured.append(s)
      mock_stdout.flush = lambda: None
      cli._render_task_panel()

    output = "".join(captured)
    # ANSI strikethrough: \033[9m
    assert "\033[9m" in output
    assert "Done task" in output

  def test_render_panel_mixed_statuses(self):
    """Panel renders all 3 subjects."""
    import sys
    from unittest.mock import patch

    cli = self._make_cli()
    cli._track_todo_write({
      "todos": [
        {"content": "Done", "status": "completed"},
        {"content": "Active", "status": "in_progress"},
        {"content": "Waiting", "status": "pending"},
      ]
    })

    captured = []
    with patch.object(sys, "stdout") as mock_stdout:
      mock_stdout.isatty.return_value = True
      mock_stdout.write = lambda s: captured.append(s)
      mock_stdout.flush = lambda: None
      cli._render_task_panel()

    output = "".join(captured)
    assert "Done" in output
    assert "Active" in output
    assert "Waiting" in output

  def test_render_panel_empty(self):
    """Empty task list produces no output."""
    import sys
    from unittest.mock import patch

    cli = self._make_cli()
    captured = []
    with patch.object(sys, "stdout") as mock_stdout:
      mock_stdout.isatty.return_value = True
      mock_stdout.write = lambda s: captured.append(s)
      mock_stdout.flush = lambda: None
      cli._render_task_panel()

    assert len(captured) == 0

  def test_render_panel_overflow(self):
    """>12 tasks shows '+N more' message."""
    import sys
    from unittest.mock import patch

    cli = self._make_cli()
    for i in range(15):
      cli._track_task_create({"subject": f"Task {i + 1}"})

    captured = []
    with patch.object(sys, "stdout") as mock_stdout:
      mock_stdout.isatty.return_value = True
      mock_stdout.write = lambda s: captured.append(s)
      mock_stdout.flush = lambda: None
      cli._render_task_panel()

    output = "".join(captured)
    # Rich adds ANSI codes within the string; strip them for assertion
    import re as _re

    plain = _re.sub(r"\033\[[0-9;]*m", "", output)
    assert "+3 more" in plain

  def test_render_panel_active_form(self):
    """In-progress tasks show activeForm text."""
    import sys
    from unittest.mock import patch

    cli = self._make_cli()
    cli._track_task_create({"subject": "Build it", "activeForm": "Building it..."})
    cli._track_task_update({"taskId": "1", "status": "in_progress"})

    captured = []
    with patch.object(sys, "stdout") as mock_stdout:
      mock_stdout.isatty.return_value = True
      mock_stdout.write = lambda s: captured.append(s)
      mock_stdout.flush = lambda: None
      cli._render_task_panel()

    output = "".join(captured)
    # Rich may style the ellipsis separately; strip ANSI for assertion
    import re as _re

    plain = _re.sub(r"\033\[[0-9;]*m", "", output)
    assert "Building it..." in plain

  def test_clear_panel_ansi(self):
    """_clear_task_panel emits cursor-up + clear-to-end ANSI sequences."""
    import sys
    from unittest.mock import patch

    cli = self._make_cli()
    cli._panel_line_count = 5

    captured = []
    with patch.object(sys, "stdout") as mock_stdout:
      mock_stdout.isatty.return_value = True
      mock_stdout.write = lambda s: captured.append(s)
      mock_stdout.flush = lambda: None
      cli._clear_task_panel()

    output = "".join(captured)
    assert "\033[5A" in output  # cursor up 5
    assert "\033[J" in output  # clear to end
    assert cli._panel_line_count == 0

  def test_event_flow_todo_write(self):
    """Feeding a TodoWrite ToolCallStartedEvent tracks tasks in the panel."""
    from definable.model.response import ToolExecution

    cli = self._make_cli()
    console, buf = self._make_console()
    cli._console = console

    event = ToolCallStartedEvent(
      agent_id="a",
      agent_name="n",
      run_id="r",
      session_id="s",
      tool=ToolExecution(
        tool_name="TodoWrite",
        tool_args={
          "todos": [
            {"content": "Step 1", "status": "completed"},
            {"content": "Step 2", "status": "pending"},
          ]
        },
      ),
    )
    cli._render_event(event)
    assert len(cli._tracked_tasks) == 2
    assert cli._tracked_tasks[0].subject == "Step 1"
    assert cli._tracked_tasks[0].status == "completed"
    output = buf.getvalue()
    assert "Updating tasks" in output

  def test_event_flow_generic_tool_unchanged(self):
    """Non-task tools still use generic rendering path."""
    from definable.model.response import ToolExecution

    cli = self._make_cli()
    console, buf = self._make_console()
    cli._console = console

    event = ToolCallStartedEvent(
      agent_id="a",
      agent_name="n",
      run_id="r",
      session_id="s",
      tool=ToolExecution(tool_name="CustomTool", tool_args={"foo": "bar"}),
    )
    cli._render_event(event)
    output = buf.getvalue()
    assert "CustomTool" in output
    assert "foo=bar" in output
    assert len(cli._tracked_tasks) == 0

  def test_toolbar_shows_task_summary(self):
    """After tasks are tracked, toolbar includes task count."""
    cli = self._make_cli()
    cli._track_task_create({"subject": "Task A"})
    cli._track_task_create({"subject": "Task B"})
    cli._track_task_update({"taskId": "1", "status": "completed"})

    done = sum(1 for t in cli._tracked_tasks if t.status == "completed")
    total = len(cli._tracked_tasks)
    task_info = f"tasks {done}/{total}"
    assert task_info == "tasks 1/2"


@pytest.mark.unit
class TestCLIImports:
  def test_import_from_claude_code_package(self):
    """ClaudeCodeCLI and run_cli are importable from definable.claude_code."""
    from definable.claude_code import ClaudeCodeCLI, run_cli

    assert ClaudeCodeCLI is not None
    assert callable(run_cli)


# ===========================================================================
# AskUserQuestion tests
# ===========================================================================


@pytest.mark.unit
class TestParseAskUserInput:
  """Tests for parse_ask_user_input type parsing."""

  def test_single_question(self):
    """Single question with options parses correctly."""
    from definable.claude_code.types import parse_ask_user_input

    raw = {
      "questions": [
        {
          "question": "Which framework?",
          "header": "Framework",
          "multiSelect": False,
          "options": [
            {"label": "React", "description": "A UI library"},
            {"label": "Vue", "description": "Progressive framework"},
          ],
        }
      ]
    }
    result = parse_ask_user_input(raw)
    assert len(result.questions) == 1
    q = result.questions[0]
    assert q.question == "Which framework?"
    assert q.header == "Framework"
    assert q.multi_select is False
    assert len(q.options) == 2
    assert q.options[0].label == "React"
    assert q.options[1].description == "Progressive framework"

  def test_multi_select_question(self):
    """multiSelect flag is parsed correctly."""
    from definable.claude_code.types import parse_ask_user_input

    raw = {
      "questions": [
        {
          "question": "Which features?",
          "header": "Features",
          "multiSelect": True,
          "options": [
            {"label": "Auth", "description": ""},
            {"label": "DB", "description": ""},
          ],
        }
      ]
    }
    result = parse_ask_user_input(raw)
    assert result.questions[0].multi_select is True

  def test_multiple_questions(self):
    """Multiple questions in a single input."""
    from definable.claude_code.types import parse_ask_user_input

    raw = {
      "questions": [
        {"question": "Q1?", "header": "H1", "options": [], "multiSelect": False},
        {"question": "Q2?", "header": "H2", "options": [], "multiSelect": False},
      ]
    }
    result = parse_ask_user_input(raw)
    assert len(result.questions) == 2
    assert result.questions[0].question == "Q1?"
    assert result.questions[1].question == "Q2?"

  def test_empty_input(self):
    """Empty or missing questions returns empty list."""
    from definable.claude_code.types import parse_ask_user_input

    assert len(parse_ask_user_input({}).questions) == 0
    assert len(parse_ask_user_input({"questions": []}).questions) == 0

  def test_missing_optional_fields(self):
    """Missing header/options defaults gracefully."""
    from definable.claude_code.types import parse_ask_user_input

    raw = {"questions": [{"question": "Hello?"}]}
    result = parse_ask_user_input(raw)
    q = result.questions[0]
    assert q.question == "Hello?"
    assert q.header == ""
    assert q.options == []
    assert q.multi_select is False


@pytest.mark.unit
class TestHandleControlAskUserQuestion:
  """Tests for AskUserQuestion intercept in _handle_control."""

  async def test_ask_user_returns_paused_event(self, run_context):
    """AskUserQuestion control request returns RunPausedEvent."""
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent()
    agent._ensure_initialized()

    request = ControlRequest(
      id="ctrl-1",
      subtype="can_use_tool",
      tool_name="AskUserQuestion",
      input={
        "questions": [
          {
            "question": "Which approach?",
            "header": "Approach",
            "options": [{"label": "A", "description": "Option A"}],
            "multiSelect": False,
          }
        ]
      },
    )
    response = await agent._handle_control(request, run_context)

    assert isinstance(response, RunPausedEvent)
    assert response.run_id == "run-1"
    assert len(response.requirements) == 1
    req = response.requirements[0]
    assert req.tool_execution.tool_name == "AskUserQuestion"
    assert req.tool_execution.requires_user_input is True

  async def test_ask_user_empty_questions_auto_allows(self, run_context):
    """AskUserQuestion with no questions auto-allows with empty answers."""
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent()
    agent._ensure_initialized()

    request = ControlRequest(
      id="ctrl-2",
      subtype="can_use_tool",
      tool_name="AskUserQuestion",
      input={"questions": []},
    )
    response = await agent._handle_control(request, run_context)

    assert isinstance(response, dict)
    assert response["behavior"] == "allow"
    assert response["updatedInput"] == {"questions": [], "answers": {}}

  async def test_ask_user_none_input_auto_allows(self, run_context):
    """AskUserQuestion with None input auto-allows."""
    from definable.claude_code.agent import ClaudeCodeAgent

    agent = ClaudeCodeAgent()
    agent._ensure_initialized()

    request = ControlRequest(
      id="ctrl-3",
      subtype="can_use_tool",
      tool_name="AskUserQuestion",
      input=None,
    )
    response = await agent._handle_control(request, run_context)

    assert isinstance(response, dict)
    assert response["behavior"] == "allow"


@pytest.mark.unit
class TestAskUserControlResponse:
  """Tests for the updatedInput format in continue_run."""

  def test_hitl_state_stores_ask_user_input(self):
    """_HitlState stores ask_user_input when set."""
    from definable.claude_code.agent import _HitlState
    from unittest.mock import MagicMock

    state = _HitlState(
      transport=MagicMock(),
      control_request_id="ctrl-1",
      context=RunContext(run_id="r", session_id="s"),
      sdk_messages=[],
      prompt="test",
      ask_user_input={"questions": [{"question": "Q?"}]},
    )
    assert state.ask_user_input is not None
    assert state.ask_user_input["questions"][0]["question"] == "Q?"

  def test_hitl_state_default_none(self):
    """_HitlState.ask_user_input defaults to None."""
    from definable.claude_code.agent import _HitlState
    from unittest.mock import MagicMock

    state = _HitlState(
      transport=MagicMock(),
      control_request_id="ctrl-1",
      context=RunContext(run_id="r", session_id="s"),
      sdk_messages=[],
      prompt="test",
    )
    assert state.ask_user_input is None


@pytest.mark.unit
class TestAskUserAnswerParsing:
  """Tests for _parse_ask_user_response in the CLI."""

  def _make_cli(self):
    from definable.claude_code.agent import ClaudeCodeAgent
    from definable.claude_code.cli import ClaudeCodeCLI
    from definable.claude_code.types import AskUserOption

    agent = ClaudeCodeAgent()
    cli = ClaudeCodeCLI(agent=agent)
    return cli, AskUserOption

  def test_numeric_single_select(self):
    """Selecting option 1 returns the first option label."""
    cli, AskUserOption = self._make_cli()
    options = [AskUserOption(label="React", description=""), AskUserOption(label="Vue", description="")]
    result = cli._parse_ask_user_response("1", options, other_idx=3, multi_select=False)
    assert result == "React"

  def test_numeric_second_option(self):
    """Selecting option 2 returns the second label."""
    cli, AskUserOption = self._make_cli()
    options = [AskUserOption(label="React", description=""), AskUserOption(label="Vue", description="")]
    result = cli._parse_ask_user_response("2", options, other_idx=3, multi_select=False)
    assert result == "Vue"

  def test_multi_select_comma_separated(self):
    """Comma-separated numbers returns comma-separated labels."""
    cli, AskUserOption = self._make_cli()
    options = [
      AskUserOption(label="Auth", description=""),
      AskUserOption(label="DB", description=""),
      AskUserOption(label="Cache", description=""),
    ]
    result = cli._parse_ask_user_response("1, 3", options, other_idx=4, multi_select=True)
    assert result == "Auth, Cache"

  def test_multi_select_rejected_for_single(self):
    """Multiple numbers rejected when multi_select is False."""
    cli, AskUserOption = self._make_cli()
    options = [AskUserOption(label="A", description=""), AskUserOption(label="B", description="")]
    result = cli._parse_ask_user_response("1, 2", options, other_idx=3, multi_select=False)
    assert result is None  # Invalid — should re-prompt

  def test_out_of_range_returns_none(self):
    """Out-of-range number returns None to trigger re-prompt."""
    cli, AskUserOption = self._make_cli()
    options = [AskUserOption(label="A", description="")]
    result = cli._parse_ask_user_response("5", options, other_idx=2, multi_select=False)
    assert result is None

  def test_zero_returns_none(self):
    """Zero is out of range and returns None."""
    cli, AskUserOption = self._make_cli()
    options = [AskUserOption(label="A", description="")]
    result = cli._parse_ask_user_response("0", options, other_idx=2, multi_select=False)
    assert result is None

  def test_free_text_returns_none(self):
    """Non-numeric text returns None from _parse (treated as free text by caller)."""
    cli, AskUserOption = self._make_cli()
    options = [AskUserOption(label="A", description="")]
    result = cli._parse_ask_user_response("some free text", options, other_idx=2, multi_select=False)
    assert result is None

  def test_other_option_prompts_free_text(self, monkeypatch):
    """Selecting the 'Other' index prompts for free text."""
    cli, AskUserOption = self._make_cli()
    options = [AskUserOption(label="A", description=""), AskUserOption(label="B", description="")]
    other_idx = 3
    monkeypatch.setattr("builtins.input", lambda _: "my custom answer")
    result = cli._parse_ask_user_response(str(other_idx), options, other_idx=other_idx, multi_select=False)
    assert result == "my custom answer"
