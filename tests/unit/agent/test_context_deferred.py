"""Unit tests for deferred tool loading — progressive disclosure of tool schemas."""

import pytest

from definable.agent.context import Context
from definable.agent.context.deferred import LOAD_TOOLS_NAME, DeferredToolManager
from definable.tool.decorator import tool


# ── Test tools ────────────────────────────────────────────────


@tool
def search(query: str) -> str:
  """Search the web for information."""
  return f"Results for: {query}"


@tool
def write_file(path: str, content: str) -> str:
  """Write content to a file on disk."""
  return f"Wrote {len(content)} chars to {path}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
  """Send an email to a recipient."""
  return f"Sent email to {to}"


@tool
def resize_image(path: str, width: int, height: int) -> str:
  """Resize an image to specified dimensions."""
  return f"Resized {path} to {width}x{height}"


def _make_tools():
  return {
    "search": search,
    "write_file": write_file,
    "send_email": send_email,
    "resize_image": resize_image,
  }


# ── Catalog generation ────────────────────────────────────────


@pytest.mark.unit
class TestCatalogGeneration:
  def test_catalog_contains_all_tool_names(self):
    mgr = DeferredToolManager(_make_tools())
    catalog = mgr.build_catalog()
    assert "search" in catalog
    assert "write_file" in catalog
    assert "send_email" in catalog
    assert "resize_image" in catalog

  def test_catalog_contains_descriptions(self):
    mgr = DeferredToolManager(_make_tools())
    catalog = mgr.build_catalog()
    assert "Search the web" in catalog
    assert "Write content to a file" in catalog

  def test_catalog_has_load_tools_instruction(self):
    mgr = DeferredToolManager(_make_tools())
    catalog = mgr.build_catalog()
    assert "load_tools" in catalog

  def test_catalog_is_compact(self):
    mgr = DeferredToolManager(_make_tools())
    catalog = mgr.build_catalog()
    # 4 tools should produce a short catalog (< 500 chars)
    assert len(catalog) < 500

  def test_empty_tools_returns_empty_catalog(self):
    mgr = DeferredToolManager({})
    assert mgr.build_catalog() == ""


# ── Loader tool ───────────────────────────────────────────────


@pytest.mark.unit
class TestLoaderTool:
  def test_get_loader_tool_returns_function(self):
    mgr = DeferredToolManager(_make_tools())
    loader = mgr.get_loader_tool()
    assert loader.name == LOAD_TOOLS_NAME

  def test_loader_tool_is_cacheable(self):
    """Calling get_loader_tool twice returns the same instance."""
    mgr = DeferredToolManager(_make_tools())
    a = mgr.get_loader_tool()
    b = mgr.get_loader_tool()
    assert a is b

  def test_loader_tool_has_schema(self):
    mgr = DeferredToolManager(_make_tools())
    loader = mgr.get_loader_tool()
    schema = loader.to_dict()
    assert schema["name"] == LOAD_TOOLS_NAME
    assert "parameters" in schema


# ── Loading tools ─────────────────────────────────────────────


@pytest.mark.unit
class TestToolLoading:
  def test_load_activates_tools(self):
    mgr = DeferredToolManager(_make_tools())
    loaded = mgr.load(["search", "write_file"])
    assert loaded == ["search", "write_file"]
    assert "search" in mgr.loaded_tool_names
    assert "write_file" in mgr.loaded_tool_names

  def test_load_unknown_tool_skipped(self):
    mgr = DeferredToolManager(_make_tools())
    loaded = mgr.load(["search", "nonexistent"])
    assert loaded == ["search"]
    assert "nonexistent" not in mgr.loaded_tool_names

  def test_prepare_for_run_resets(self):
    mgr = DeferredToolManager(_make_tools())
    mgr.load(["search"])
    assert len(mgr.loaded_tool_names) == 1
    mgr.prepare_for_run()
    assert len(mgr.loaded_tool_names) == 0


# ── Active tools ──────────────────────────────────────────────


@pytest.mark.unit
class TestActiveTools:
  def test_active_always_includes_loader(self):
    mgr = DeferredToolManager(_make_tools())
    active = mgr.get_active_tools()
    assert LOAD_TOOLS_NAME in active
    assert len(active) == 1  # Only loader, nothing loaded yet

  def test_active_includes_loaded_tools(self):
    mgr = DeferredToolManager(_make_tools())
    mgr.load(["search", "send_email"])
    active = mgr.get_active_tools()
    assert LOAD_TOOLS_NAME in active
    assert "search" in active
    assert "send_email" in active
    assert len(active) == 3  # loader + 2 loaded

  def test_active_does_not_include_unloaded_tools(self):
    mgr = DeferredToolManager(_make_tools())
    mgr.load(["search"])
    active = mgr.get_active_tools()
    assert "write_file" not in active
    assert "resize_image" not in active

  def test_get_tools_dicts_format(self):
    mgr = DeferredToolManager(_make_tools())
    mgr.load(["search"])
    dicts = mgr.get_tools_dicts()
    assert isinstance(dicts, list)
    assert len(dicts) == 2  # loader + search
    for d in dicts:
      assert d["type"] == "function"
      assert "function" in d
      assert "name" in d["function"]


# ── load_tools function execution ─────────────────────────────


@pytest.mark.unit
class TestLoadToolsExecution:
  def test_calling_load_tools_activates(self):
    """Simulate what happens when the model calls load_tools."""
    mgr = DeferredToolManager(_make_tools())
    loader = mgr.get_loader_tool()

    # The model calls load_tools(names=["search", "write_file"])
    assert loader.entrypoint is not None
    result = loader.entrypoint(names=["search", "write_file"])
    assert "Loaded: search, write_file" in result

    # Now those tools should be active
    active = mgr.get_active_tools()
    assert "search" in active
    assert "write_file" in active

  def test_load_tools_with_unknown_name(self):
    mgr = DeferredToolManager(_make_tools())
    loader = mgr.get_loader_tool()

    assert loader.entrypoint is not None
    result = loader.entrypoint(names=["search", "nonexistent_tool"])
    assert "Loaded: search" in result
    assert "Not found: nonexistent_tool" in result

  def test_load_tools_empty_list(self):
    mgr = DeferredToolManager(_make_tools())
    loader = mgr.get_loader_tool()

    assert loader.entrypoint is not None
    result = loader.entrypoint(names=[])
    assert "No tools specified" in result

  def test_load_tools_self_reference_ignored(self):
    """Model asking to load 'load_tools' itself should be a no-op."""
    mgr = DeferredToolManager(_make_tools())
    loader = mgr.get_loader_tool()

    assert loader.entrypoint is not None
    result = loader.entrypoint(names=["load_tools", "search"])
    assert "Loaded: search" in result
    # load_tools should not appear as "loaded" — it's always available
    assert "load_tools" not in result.split("Loaded: ")[1].split(".")[0]


# ── Agent integration ─────────────────────────────────────────


@pytest.mark.unit
class TestAgentDeferredIntegration:
  def test_agent_creates_deferred_manager(self):
    from definable.agent.agent import Agent
    from definable.agent.testing import MockModel

    agent = Agent(
      model=MockModel(),  # type: ignore[arg-type]
      tools=[search, write_file, send_email],
      context=Context(deferred_tools=True),
    )
    assert agent._deferred_tool_manager is not None
    assert len(agent._deferred_tool_manager.all_tool_names) == 3

  def test_agent_no_deferred_without_flag(self):
    from definable.agent.agent import Agent
    from definable.agent.testing import MockModel

    agent = Agent(
      model=MockModel(),  # type: ignore[arg-type]
      tools=[search, write_file],
      context=True,
    )
    assert agent._deferred_tool_manager is None

  def test_agent_no_deferred_without_context(self):
    from definable.agent.agent import Agent
    from definable.agent.testing import MockModel

    agent = Agent(
      model=MockModel(),  # type: ignore[arg-type]
      tools=[search, write_file],
    )
    assert agent._deferred_tool_manager is None

  def test_all_tool_names_accessible(self):
    from definable.agent.agent import Agent
    from definable.agent.testing import MockModel

    agent = Agent(
      model=MockModel(),  # type: ignore[arg-type]
      tools=[search, write_file, send_email, resize_image],
      context=Context(deferred_tools=True),
    )
    mgr = agent._deferred_tool_manager
    assert mgr is not None
    names = mgr.all_tool_names
    assert "search" in names
    assert "write_file" in names
    assert "send_email" in names
    assert "resize_image" in names
