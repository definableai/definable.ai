"""Tests for the unified workspace directory utility."""

from definable.utils.workspace import (
  _DIR_NAME,
  _GITIGNORE_CONTENT,
  ensure_workspace,
  get_project_root,
  get_workspace_dir,
  workspace_path,
)


class TestGetProjectRoot:
  def test_defaults_to_cwd(self, monkeypatch, tmp_path):
    monkeypatch.delenv("DEFINABLE_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert get_project_root() == tmp_path.resolve()

  def test_respects_env_var(self, monkeypatch, tmp_path):
    monkeypatch.setenv("DEFINABLE_ROOT", str(tmp_path))
    assert get_project_root() == tmp_path.resolve()

  def test_resolves_relative_env_var(self, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEFINABLE_ROOT", ".")
    assert get_project_root() == tmp_path.resolve()


class TestGetWorkspaceDir:
  def test_returns_dotdefinable_under_project_root(self, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEFINABLE_ROOT", raising=False)
    ws = get_workspace_dir()
    assert ws == tmp_path.resolve() / _DIR_NAME

  def test_does_not_create_directory(self, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEFINABLE_ROOT", raising=False)
    ws = get_workspace_dir()
    assert not ws.exists()


class TestEnsureWorkspace:
  def test_creates_directory(self, monkeypatch, tmp_path):
    monkeypatch.setenv("DEFINABLE_ROOT", str(tmp_path))
    ws = ensure_workspace()
    assert ws.exists()
    assert ws.is_dir()
    assert ws == tmp_path / _DIR_NAME

  def test_seeds_gitignore(self, monkeypatch, tmp_path):
    monkeypatch.setenv("DEFINABLE_ROOT", str(tmp_path))
    ws = ensure_workspace()
    gitignore = ws / ".gitignore"
    assert gitignore.exists()
    assert gitignore.read_text(encoding="utf-8") == _GITIGNORE_CONTENT

  def test_idempotent(self, monkeypatch, tmp_path):
    monkeypatch.setenv("DEFINABLE_ROOT", str(tmp_path))
    ws1 = ensure_workspace()
    ws2 = ensure_workspace()
    assert ws1 == ws2
    assert ws1.exists()

  def test_does_not_overwrite_existing_gitignore(self, monkeypatch, tmp_path):
    monkeypatch.setenv("DEFINABLE_ROOT", str(tmp_path))
    ws_dir = tmp_path / _DIR_NAME
    ws_dir.mkdir()
    gitignore = ws_dir / ".gitignore"
    gitignore.write_text("custom content\n")

    ensure_workspace()
    assert gitignore.read_text() == "custom content\n"


class TestWorkspacePath:
  def test_single_segment_file(self, monkeypatch, tmp_path):
    monkeypatch.setenv("DEFINABLE_ROOT", str(tmp_path))
    p = workspace_path("memory.db")
    assert p == tmp_path / _DIR_NAME / "memory.db"
    # Parent dir should exist
    assert p.parent.exists()

  def test_single_segment_directory(self, monkeypatch, tmp_path):
    monkeypatch.setenv("DEFINABLE_ROOT", str(tmp_path))
    p = workspace_path("traces")
    assert p == tmp_path / _DIR_NAME / "traces"
    # Directory itself should exist (no suffix → treated as dir)
    assert p.exists()
    assert p.is_dir()

  def test_nested_segments(self, monkeypatch, tmp_path):
    monkeypatch.setenv("DEFINABLE_ROOT", str(tmp_path))
    p = workspace_path("cache", "tools")
    assert p == tmp_path / _DIR_NAME / "cache" / "tools"
    assert p.exists()
    assert p.is_dir()

  def test_nested_file(self, monkeypatch, tmp_path):
    monkeypatch.setenv("DEFINABLE_ROOT", str(tmp_path))
    p = workspace_path("browser", "screenshots", "page.png")
    assert p == tmp_path / _DIR_NAME / "browser" / "screenshots" / "page.png"
    assert p.parent.exists()
    assert not p.exists()  # File itself not created

  def test_mkdir_false_skips_creation(self, monkeypatch, tmp_path):
    monkeypatch.setenv("DEFINABLE_ROOT", str(tmp_path))
    p = workspace_path("nonexistent", "deep", "path", mkdir=False)
    assert not p.parent.exists()

  def test_creates_workspace_gitignore(self, monkeypatch, tmp_path):
    monkeypatch.setenv("DEFINABLE_ROOT", str(tmp_path))
    workspace_path("memory.db")
    gitignore = tmp_path / _DIR_NAME / ".gitignore"
    assert gitignore.exists()


class TestConsumerDefaults:
  """Verify that consumers resolve to .definable/ when no explicit path is given."""

  def test_file_store_default(self, monkeypatch, tmp_path):
    monkeypatch.setenv("DEFINABLE_ROOT", str(tmp_path))
    from definable.memory.store.file import FileStore

    store = FileStore()
    assert str(store.base_dir) == str(tmp_path / _DIR_NAME / "memory")

  def test_file_store_explicit_overrides(self, tmp_path):
    from definable.memory.store.file import FileStore

    store = FileStore(base_dir=str(tmp_path / "custom"))
    assert str(store.base_dir) == str(tmp_path / "custom")

  def test_sqlite_store_default(self, monkeypatch, tmp_path):
    monkeypatch.setenv("DEFINABLE_ROOT", str(tmp_path))
    from definable.memory.store.sqlite import SQLiteStore

    store = SQLiteStore()
    assert store.db_path == str(tmp_path / _DIR_NAME / "memory.db")

  def test_sqlite_store_explicit_overrides(self, tmp_path):
    from definable.memory.store.sqlite import SQLiteStore

    store = SQLiteStore(db_path=str(tmp_path / "custom.db"))
    assert store.db_path == str(tmp_path / "custom.db")

  def test_jsonl_exporter_default(self, monkeypatch, tmp_path):
    monkeypatch.setenv("DEFINABLE_ROOT", str(tmp_path))
    from definable.agent.tracing.jsonl import JSONLExporter

    exporter = JSONLExporter()
    assert exporter.trace_dir == tmp_path / _DIR_NAME / "traces"
    exporter.shutdown()

  def test_jsonl_exporter_explicit_overrides(self, tmp_path):
    from definable.agent.tracing.jsonl import JSONLExporter

    trace_dir = tmp_path / "custom_traces"
    exporter = JSONLExporter(trace_dir=str(trace_dir))
    assert exporter.trace_dir == trace_dir
    exporter.shutdown()

  def test_trace_browser_default(self, monkeypatch, tmp_path):
    monkeypatch.setenv("DEFINABLE_ROOT", str(tmp_path))
    from definable.agent.observability.trace_browser import TraceBrowser

    browser = TraceBrowser()
    assert browser.trace_dir == str(tmp_path / _DIR_NAME / "traces")

  def test_trace_browser_explicit_overrides(self, tmp_path):
    from definable.agent.observability.trace_browser import TraceBrowser

    browser = TraceBrowser(trace_dir=str(tmp_path / "custom"))
    assert browser.trace_dir == str(tmp_path / "custom")

  def test_identity_resolver_default(self, monkeypatch, tmp_path):
    monkeypatch.setenv("DEFINABLE_ROOT", str(tmp_path))
    from definable.agent.interface.identity import SQLiteIdentityResolver

    resolver = SQLiteIdentityResolver()
    assert resolver.db_path == str(tmp_path / _DIR_NAME / "identity.db")

  def test_identity_resolver_explicit_overrides(self, tmp_path):
    from definable.agent.interface.identity import SQLiteIdentityResolver

    resolver = SQLiteIdentityResolver(db_path=str(tmp_path / "custom.db"))
    assert resolver.db_path == str(tmp_path / "custom.db")
