from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]

DOC_PAGES = [
  "docs/_snippets/models-note.mdx",
  "docs/concepts/overview.mdx",
  "docs/concepts/agent-vs-team-vs-workflow.mdx",
  "docs/concepts/knowledge-vs-memory.mdx",
  "docs/concepts/tools-vs-toolkits-vs-skills-vs-mcp.mdx",
  "docs/concepts/runtime-vs-interface.mdx",
  "docs/concepts/run-lifecycle.mdx",
  "docs/tutorials/overview.mdx",
  "docs/tutorials/build-a-support-agent.mdx",
  "docs/tutorials/build-a-rag-assistant.mdx",
  "docs/tutorials/build-a-multi-interface-agent.mdx",
  "docs/tutorials/build-a-runtime-service.mdx",
  "docs/tutorials/build-a-review-workflow.mdx",
  "docs/releases/overview.mdx",
  "docs/releases/current-series.mdx",
  "docs/releases/versioning-policy.mdx",
  "docs/migrations/overview.mdx",
  "docs/migrations/pre-upgrade-checklist.mdx",
  "docs/migrations/post-upgrade-validation.mdx",
  "docs/quickstart.mdx",
  "docs/first-agent.mdx",
  "docs/agents/overview.mdx",
  "docs/agents/building-agents.mdx",
  "docs/agents/running-agents.mdx",
  "docs/agents/configuration.mdx",
  "docs/agents/auth.mdx",
  "docs/agents/context.mdx",
  "docs/agents/compression.mdx",
  "docs/agents/evaluation.mdx",
  "docs/agents/pipeline.mdx",
  "docs/agents/plugins.mdx",
  "docs/agents/run.mdx",
  "docs/agents/runtime.mdx",
  "docs/agents/agentic-loop.mdx",
  "docs/agents/deep-research.mdx",
  "docs/guardrails/overview.mdx",
  "docs/input-output/structured-output.mdx",
  "docs/input-output/streaming.mdx",
  "docs/input-output/multimodal.mdx",
  "docs/interfaces/overview.mdx",
  "docs/interfaces/call.mdx",
  "docs/interfaces/custom-interfaces.mdx",
  "docs/interfaces/discord.mdx",
  "docs/interfaces/email.mdx",
  "docs/interfaces/hooks.mdx",
  "docs/interfaces/identity.mdx",
  "docs/interfaces/multi-interface.mdx",
  "docs/interfaces/sessions.mdx",
  "docs/interfaces/slack.mdx",
  "docs/interfaces/telegram.mdx",
  "docs/interfaces/websocket.mdx",
  "docs/interfaces/whatsapp.mdx",
  "docs/interfaces/usage/discord-bot.mdx",
  "docs/interfaces/usage/multi-platform.mdx",
  "docs/interfaces/usage/slack-bot.mdx",
  "docs/interfaces/usage/telegram-bot.mdx",
  "docs/observability/overview.mdx",
  "docs/reasoning/thinking.mdx",
  "docs/reasoning/deep-research.mdx",
  "docs/replay/overview.mdx",
  "docs/browser/overview.mdx",
  "docs/knowledge/overview.mdx",
  "docs/mcp/overview.mdx",
  "docs/mcp/getting-started.mdx",
  "docs/mcp/configuration.mdx",
  "docs/mcp/error-handling.mdx",
  "docs/mcp/mock-servers.mdx",
  "docs/mcp/prompts.mdx",
  "docs/mcp/resources.mdx",
  "docs/memory/overview.mdx",
  "docs/memory/stores.mdx",
  "docs/memory/agent-integration.mdx",
  "docs/models/overview.mdx",
  "docs/models/openai.mdx",
  "docs/models/anthropic.mdx",
  "docs/models/google.mdx",
  "docs/models/deepseek.mdx",
  "docs/models/mistral.mdx",
  "docs/models/moonshot.mdx",
  "docs/models/perplexity.mdx",
  "docs/models/xai.mdx",
  "docs/models/openrouter.mdx",
  "docs/models/openai-like.mdx",
  "docs/models/ollama.mdx",
  "docs/models/model-as-string.mdx",
  "docs/models/resilience.mdx",
  "docs/models/streaming.mdx",
  "docs/models/structured-output.mdx",
  "docs/models/vision-and-audio.mdx",
  "docs/models/metrics-and-pricing.mdx",
  "docs/skills/overview.mdx",
  "docs/tools/overview.mdx",
  "docs/tools/parameters.mdx",
  "docs/tools/hooks.mdx",
  "docs/tools/caching.mdx",
  "docs/tools/async-tools.mdx",
  "docs/tools/dependencies.mdx",
  "docs/tools/creating-tools/overview.mdx",
  "docs/tools/creating-tools/parameters.mdx",
  "docs/tools/creating-tools/hooks.mdx",
  "docs/tools/creating-tools/caching.mdx",
  "docs/tools/creating-tools/async-tools.mdx",
  "docs/tools/creating-tools/dependencies.mdx",
  "docs/toolkits/overview.mdx",
  "docs/toolkits/knowledge-toolkit.mdx",
  "docs/toolkits/mcp-toolkit.mdx",
  "docs/knowledge/agent-integration.mdx",
  "docs/knowledge/chunkers.mdx",
  "docs/knowledge/embedders.mdx",
  "docs/knowledge/documents.mdx",
  "docs/knowledge/hybrid-search.mdx",
  "docs/knowledge/readers.mdx",
  "docs/knowledge/rerankers.mdx",
  "docs/knowledge/vector-databases.mdx",
  "docs/knowledge/usage/basic-rag.mdx",
  "docs/knowledge/usage/hybrid-search-example.mdx",
  "docs/agents/teams.mdx",
  "docs/agents/workflows.mdx",
  "docs/teams/overview.mdx",
  "docs/teams/building-teams.mdx",
  "docs/teams/running-teams.mdx",
  "docs/teams/delegation.mdx",
  "docs/workflows/overview.mdx",
  "docs/workflows/building-workflows.mdx",
  "docs/workflows/running-workflows.mdx",
  "docs/workflows/patterns/sequential.mdx",
  "docs/workflows/patterns/parallel.mdx",
  "docs/workflows/patterns/conditional.mdx",
  "docs/workflows/patterns/loop.mdx",
  "docs/workflows/patterns/router.mdx",
  "docs/advanced/cost-tracking.mdx",
  "docs/advanced/run-output.mdx",
  "docs/advanced/error-handling.mdx",
  "docs/advanced/media-types.mdx",
  "docs/agents/middleware.mdx",
  "docs/agents/multi-turn.mdx",
  "docs/agents/testing.mdx",
  "docs/agents/tracing.mdx",
  "docs/agents/thinking.mdx",
  "docs/agents/security.mdx",
  "docs/agents/scheduling.mdx",
  "docs/agents/usage/agent-with-knowledge.mdx",
  "docs/agents/usage/agent-with-memory.mdx",
  "docs/agents/usage/agent-with-streaming.mdx",
  "docs/agents/usage/agent-with-structured-output.mdx",
  "docs/agents/usage/agent-with-tools.mdx",
  "docs/claude-code/overview.mdx",
  "docs/introduction.mdx",
  "docs/installation.mdx",
  "docs/knowledge/pipeline/chunkers.mdx",
  "docs/knowledge/pipeline/embedders.mdx",
  "docs/knowledge/pipeline/readers.mdx",
  "docs/knowledge/pipeline/rerankers.mdx",
  "docs/knowledge/pipeline/vector-databases.mdx",
  "docs/examples/tools.mdx",
  "docs/examples/overview.mdx",
  "docs/examples/agents.mdx",
  "docs/examples/auth.mdx",
  "docs/examples/browser.mdx",
  "docs/examples/evaluation.mdx",
  "docs/examples/guardrails.mdx",
  "docs/examples/interfaces.mdx",
  "docs/examples/knowledge.mdx",
  "docs/examples/memory.mdx",
  "docs/examples/models.mdx",
  "docs/examples/mcp.mdx",
  "docs/examples/replay.mdx",
  "docs/examples/security.mdx",
  "docs/examples/skills.mdx",
  "docs/examples/readers.mdx",
  "docs/examples/teams.mdx",
  "docs/examples/workflows.mdx",
  "docs/readers/overview.mdx",
  "docs/reference/agents/agent.mdx",
  "docs/reference/agents/config.mdx",
  "docs/reference/agents/run-output.mdx",
  "docs/reference/knowledge/knowledge.mdx",
  "docs/reference/memory/memory.mdx",
  "docs/reference/models/model.mdx",
  "docs/reference/teams/team.mdx",
  "docs/reference/tools/decorator.mdx",
  "docs/reference/workflows/workflow.mdx",
  "docs/skills/macos.mdx",
  "docs/skills/registry.mdx",
  "docs/skills/built-in.mdx",
  "docs/teams/usage/collaborate-team.mdx",
  "docs/teams/usage/coordinate-team.mdx",
  "docs/teams/usage/route-team.mdx",
  "docs/teams/usage/tasks-team.mdx",
  "docs/workflows/usage/basic-workflow.mdx",
  "docs/workflows/usage/nested-workflows.mdx",
  "docs/production/overview.mdx",
  "docs/production/deployment-checklist.mdx",
  "docs/production/auth-and-security.mdx",
  "docs/production/observability-and-costs.mdx",
  "docs/troubleshooting/overview.mdx",
  "docs/troubleshooting/models-and-auth.mdx",
  "docs/troubleshooting/retrieval-and-memory.mdx",
  "docs/troubleshooting/runtime-and-interfaces.mdx",
  "docs/troubleshooting/tools-and-structured-output.mdx",
]

PROSE_ONLY_DOC_PAGES = [
  "docs/_snippets/agents-resources.mdx",
  "docs/_snippets/async-note.mdx",
  "docs/_snippets/export-openai-key.mdx",
  "docs/_snippets/install-definable.mdx",
  "docs/_snippets/knowledge-true-warning.mdx",
  "docs/_snippets/output-schema-warning.mdx",
  "docs/_snippets/run-agent-step.mdx",
  "docs/_snippets/setup-venv.mdx",
]

EXAMPLE_SCRIPTS = [
  "examples/docs/agent_basics.py",
  "examples/docs/agent_auth.py",
  "examples/docs/agent_guardrails.py",
  "examples/docs/agent_middleware.py",
  "examples/docs/agent_runtime.py",
  "examples/docs/agent_reasoning.py",
  "examples/docs/agent_replay.py",
  "examples/docs/agent_scheduling.py",
  "examples/docs/agent_security.py",
  "examples/docs/agent_tracing.py",
  "examples/docs/browser_toolkit.py",
  "examples/docs/evaluation_basics.py",
  "examples/docs/interfaces_basics.py",
  "examples/docs/interfaces_call_modes.py",
  "examples/docs/knowledge_basics.py",
  "examples/docs/knowledge_pipeline.py",
  "examples/docs/mcp_basics.py",
  "examples/docs/memory_basics.py",
  "examples/docs/models_basics.py",
  "examples/docs/readers_basics.py",
  "examples/docs/reference_basics.py",
  "examples/docs/skills_basics.py",
  "examples/docs/tools_basics.py",
  "examples/docs/tools_parameters.py",
  "examples/docs/tools_hooks.py",
  "examples/docs/tools_caching.py",
  "examples/docs/tools_async.py",
  "examples/docs/tools_dependencies.py",
  "examples/docs/toolkits_basics.py",
  "examples/docs/teams_basics.py",
  "examples/docs/workflows_basics.py",
]


def _extract_python_blocks(text: str) -> list[str]:
  return [block for info, block in _extract_fenced_blocks(text) if info.startswith("python")]


def _extract_fenced_blocks(text: str) -> list[tuple[str, str]]:
  blocks: list[tuple[str, str]] = []
  inside = False
  info = ""
  current: list[str] = []

  for line in text.splitlines():
    if line.startswith("```"):
      if not inside:
        info = line[3:].strip()
        inside = True
        current = []
      else:
        blocks.append((info, "\n".join(current) + "\n"))
        inside = False
        info = ""
        current = []
      continue

    if inside:
      current.append(line)

  return blocks


def _docs_pages_relative_to_docs_root() -> set[str]:
  return {str(path.relative_to(ROOT / "docs")).removesuffix(".mdx") for path in (ROOT / "docs").rglob("*.mdx")}


def _docs_assets_relative_to_docs_root() -> set[str]:
  return {str(path.relative_to(ROOT / "docs")) for path in (ROOT / "docs").rglob("*") if path.is_file() and path.suffix != ".mdx"}


def _navigation_pages() -> set[str]:
  config = json.loads((ROOT / "docs" / "docs.json").read_text())
  pages: set[str] = set()

  def add_page(value: str) -> None:
    if value.startswith("http"):
      return
    normalized = value.lstrip("/")
    if normalized.endswith(".mdx"):
      normalized = normalized.removesuffix(".mdx")
    pages.add(normalized)

  def walk(node: object) -> None:
    if isinstance(node, dict):
      page_value = node.get("page")
      if isinstance(page_value, str):
        add_page(page_value)

      page_list = node.get("pages")
      if isinstance(page_list, list):
        for item in page_list:
          if isinstance(item, str):
            add_page(item)
          else:
            walk(item)

      for key, value in node.items():
        if key not in {"page", "pages"}:
          walk(value)
    elif isinstance(node, list):
      for item in node:
        walk(item)

  walk(config.get("navigation", {}))
  return pages


def _iter_markdown_and_jsx_links(text: str) -> list[str]:
  markdown_links = re.findall(r"\]\((/[^)#?\s]+)", text)
  jsx_links = re.findall(r'href="(/[^"#?]+)', text)
  return markdown_links + jsx_links


def _iter_mdx_import_paths(text: str) -> list[str]:
  matches = re.findall(r'^\s*import\s+\w+\s+from\s+"([^"]+)"\s*$', text, flags=re.MULTILINE)
  return [match for match in matches if match.endswith(".mdx")]


def _extract_frontmatter(text: str) -> dict[str, str]:
  if not text.startswith("---\n"):
    return {}

  end = text.find("\n---\n", 4)
  if end == -1:
    return {}

  frontmatter: dict[str, str] = {}
  block = text[4:end]
  for line in block.splitlines():
    if ":" not in line:
      continue
    key, value = line.split(":", 1)
    frontmatter[key.strip()] = value.strip().strip('"')
  return frontmatter


@pytest.mark.unit
@pytest.mark.parametrize("page_path", DOC_PAGES)
def test_documentation_python_snippets(page_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.chdir(ROOT)
  page = ROOT / page_path
  blocks = _extract_python_blocks(page.read_text())

  assert blocks, f"No python snippets found in {page_path}"

  for index, block in enumerate(blocks, start=1):
    namespace = {
      "__name__": "__main__",
      "__file__": str(page),
    }
    exec(compile(block, f"{page_path}#snippet-{index}", "exec"), namespace, namespace)


@pytest.mark.unit
@pytest.mark.parametrize("page_path", PROSE_ONLY_DOC_PAGES)
def test_prose_only_documentation_pages_have_no_fenced_code_blocks(page_path: str) -> None:
  page = ROOT / page_path
  blocks = _extract_fenced_blocks(page.read_text())

  assert not blocks, f"{page_path} should not contain fenced code blocks"


@pytest.mark.unit
def test_all_mdx_pages_are_registered() -> None:
  all_docs = {str(path.relative_to(ROOT)) for path in (ROOT / "docs").rglob("*.mdx")}
  registered = set(DOC_PAGES) | set(PROSE_ONLY_DOC_PAGES)

  assert registered == all_docs


@pytest.mark.unit
def test_all_navigable_docs_pages_are_in_navigation() -> None:
  docs_pages = {page for page in _docs_pages_relative_to_docs_root() if not page.startswith("_snippets/")}

  assert _navigation_pages() == docs_pages


@pytest.mark.unit
def test_internal_docs_links_resolve() -> None:
  docs_pages = _docs_pages_relative_to_docs_root()
  docs_assets = _docs_assets_relative_to_docs_root()

  for path in (ROOT / "docs").rglob("*.mdx"):
    text = path.read_text()
    for target in _iter_markdown_and_jsx_links(text):
      assert not target.startswith("/Users/"), f"Machine-local link in {path}: {target}"

      normalized = target.lstrip("/")
      page_target = normalized.removesuffix(".mdx")

      assert page_target in docs_pages or normalized in docs_assets, f"Broken internal docs link in {path}: {target}"


@pytest.mark.unit
def test_mdx_import_targets_exist() -> None:
  for path in (ROOT / "docs").rglob("*.mdx"):
    text = path.read_text()
    for import_path in _iter_mdx_import_paths(text):
      resolved = (path.parent / import_path).resolve()
      assert resolved.exists(), f"Missing MDX import target in {path}: {import_path}"


@pytest.mark.unit
def test_docs_have_no_machine_local_paths_or_stale_repo_prefixes() -> None:
  bad_patterns = ["/Users/", "definable/examples/"]

  for path in (ROOT / "docs").rglob("*.mdx"):
    text = path.read_text()
    for pattern in bad_patterns:
      assert pattern not in text, f"Found disallowed path pattern {pattern!r} in {path}"


@pytest.mark.unit
def test_docs_have_no_shell_code_fences() -> None:
  for path in (ROOT / "docs").rglob("*.mdx"):
    fenced_blocks = _extract_fenced_blocks(path.read_text())
    shell_blocks = [info for info, _block in fenced_blocks if info.startswith("bash") or info.startswith("sh") or info.startswith("shell")]
    assert not shell_blocks, f"Shell code fence found in {path}: {shell_blocks}"


@pytest.mark.unit
def test_all_rendered_docs_pages_have_title_and_description_frontmatter() -> None:
  for path in (ROOT / "docs").rglob("*.mdx"):
    relative = str(path.relative_to(ROOT / "docs"))
    if relative.startswith("_snippets/"):
      continue

    frontmatter = _extract_frontmatter(path.read_text())
    assert frontmatter.get("title"), f"Missing title frontmatter in {path}"
    assert frontmatter.get("description"), f"Missing description frontmatter in {path}"


@pytest.mark.unit
def test_docs_workflow_matches_current_docs_layout() -> None:
  workflow = (ROOT / ".github" / "workflows" / "docs.yml").read_text()

  assert '"docs/**"' in workflow
  assert '"tests/docs/**"' in workflow
  assert "pytest tests/docs/test_documentation_examples.py" in workflow
  assert "mintlify broken-links" in workflow
  assert "working-directory: docs" in workflow


@pytest.mark.unit
@pytest.mark.parametrize("script_path", EXAMPLE_SCRIPTS)
def test_documentation_example_scripts(script_path: str) -> None:
  script = ROOT / script_path
  result = subprocess.run(
    [sys.executable, str(script)],
    cwd=ROOT,
    capture_output=True,
    text=True,
  )

  assert result.returncode == 0, f"{script_path} failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
