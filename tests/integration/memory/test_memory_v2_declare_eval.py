"""
Memory v2 evaluation for the Declare agent.

This is a standalone integration script that:
  - scripts compact, human-like conversations for multiple users
  - runs them against memory v2 with a live OpenAI model
  - measures per-turn tokens, latency, tool usage, and memory growth
  - verifies stored facts, cold-start recall, and cross-user isolation

Usage:
    export OPENAI_API_KEY=sk-...
    python definable/tests/integration/memory/test_memory_v2_declare_eval.py
    python definable/tests/integration/memory/test_memory_v2_declare_eval.py --model gpt-5-mini --json-out /tmp/declare-memory-v2.json
"""

import argparse
import asyncio
import json
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from definable.agent import Agent
from definable.memory.v2 import Memory, SQLiteStore
from definable.model.metrics import Metrics
from definable.model.openai import OpenAIChat


def _text(value: Any) -> str:
  if value is None:
    return ""
  if isinstance(value, str):
    return value
  if isinstance(value, (dict, list, tuple)):
    try:
      return json.dumps(value, ensure_ascii=True)
    except TypeError:
      return str(value)
  return str(value)


def _message_char_count(messages: Optional[list[Any]]) -> int:
  if not messages:
    return 0
  return sum(len(_text(getattr(message, "content", ""))) for message in messages)


def _extract_tool_names(output: Any) -> list[str]:
  names: list[str] = []

  for execution in output.tools or []:
    if getattr(execution, "tool_name", None):
      names.append(execution.tool_name)

  if names:
    return names

  for message in output.messages or []:
    for tool_call in getattr(message, "tool_calls", None) or []:
      if isinstance(tool_call, dict):
        name = tool_call.get("function", {}).get("name")
        if name:
          names.append(name)
      else:
        name = getattr(tool_call, "name", None)
        if name:
          names.append(name)

  return names


def _response_text(output: Any) -> str:
  content = _text(output.content)
  if content:
    return content

  for message in reversed(output.messages or []):
    if getattr(message, "role", None) != "assistant":
      continue
    message_content = _text(getattr(message, "content", ""))
    if message_content:
      return message_content
  return ""


def _percentile(values: Iterable[float], pct: float) -> float:
  ordered = sorted(values)
  if not ordered:
    return 0.0
  if len(ordered) == 1:
    return ordered[0]
  rank = (len(ordered) - 1) * pct
  low = int(rank)
  high = min(low + 1, len(ordered) - 1)
  if low == high:
    return ordered[low]
  fraction = rank - low
  return ordered[low] + (ordered[high] - ordered[low]) * fraction


@dataclass
class Probe:
  name: str
  prompt: str
  must_contain_any: list[list[str]] = field(default_factory=list)
  must_not_contain: list[str] = field(default_factory=list)


@dataclass
class SessionSpec:
  name: str
  messages: list[str]


@dataclass
class QueryCheck:
  name: str
  query: str
  should_exist: bool = True


@dataclass
class WorkingMemoryCheck:
  name: str
  must_contain: str
  must_not_contain: Optional[str] = None


@dataclass
class UserScenario:
  user_id: str
  label: str
  sessions: list[SessionSpec]
  archive_checks: list[QueryCheck]
  working_memory_checks: list[WorkingMemoryCheck]
  probes: list[Probe]


@dataclass
class TurnMetric:
  user_id: str
  user_label: str
  session_name: str
  session_kind: str
  session_id: str
  turn_index: int
  user_message: str
  agent_response: str
  status: str
  tool_calls: list[str] = field(default_factory=list)
  input_tokens: int = 0
  output_tokens: int = 0
  total_tokens: int = 0
  reasoning_tokens: int = 0
  cache_read_tokens: int = 0
  latency_ms: float = 0.0
  time_to_first_token_ms: Optional[float] = None
  history_messages: int = 0
  history_chars: int = 0
  prompt_injection_chars: int = 0
  recent_context_chars: int = 0
  working_memory_chars: int = 0
  working_memory_version: int = 0
  warm_memory_chars: int = 0
  archive_entry_count: int = 0
  archive_total_chars: int = 0
  audit_issues: list[str] = field(default_factory=list)


@dataclass
class SessionMetric:
  user_id: str
  user_label: str
  session_name: str
  session_kind: str
  session_id: str
  turn_count: int = 0
  prompt_injection_chars: int = 0
  recent_context_chars: int = 0
  wm_chars_start: int = 0
  wm_chars_end: int = 0
  archive_count_start: int = 0
  archive_count_end: int = 0
  input_tokens: int = 0
  output_tokens: int = 0
  total_tokens: int = 0
  reasoning_tokens: int = 0
  cache_read_tokens: int = 0
  latency_ms: float = 0.0
  tool_calls: list[str] = field(default_factory=list)


@dataclass
class CheckResult:
  user_id: str
  user_label: str
  kind: str
  name: str
  passed: bool
  details: str = ""


SCENARIOS: list[UserScenario] = [
  UserScenario(
    user_id="declare_alice",
    label="Alice",
    sessions=[
      SessionSpec(
        name="profile",
        messages=[
          "Hi Declare, I'm Alice Chen. I work at Stripe.",
          "I'm in San Francisco. Keep answers in bullets when possible.",
          "I'm on our fraud scoring system. It uses Kafka and Redis. Deadline is June 15.",
        ],
      ),
      SessionSpec(
        name="updates",
        messages=[
          "Quick update: we replaced Redis with DynamoDB.",
          "The deadline moved to July 1.",
          "Also, I prefer Polars over pandas and I like dark mode.",
        ],
      ),
    ],
    archive_checks=[
      QueryCheck(name="alice company stored", query="Stripe"),
      QueryCheck(name="alice city stored", query="San Francisco"),
      QueryCheck(name="alice stack stored", query="Kafka"),
      QueryCheck(name="alice preference stored", query="Polars"),
      QueryCheck(name="alice display preference stored", query="dark mode"),
    ],
    working_memory_checks=[
      WorkingMemoryCheck(name="alice current datastore", must_contain="DynamoDB", must_not_contain="Redis"),
      WorkingMemoryCheck(name="alice current deadline", must_contain="July 1", must_not_contain="June 15"),
    ],
    probes=[
      Probe(
        name="alice cold recall identity",
        prompt="Where do I work and live?",
        must_contain_any=[["stripe"], ["san francisco"]],
      ),
      Probe(
        name="alice cold recall project",
        prompt="What's my current project stack and deadline?",
        must_contain_any=[["kafka"], ["dynamodb"], ["july 1", "july first"]],
        must_not_contain=["redis", "june 15"],
      ),
      Probe(
        name="alice cold recall preference",
        prompt="How should you format answers for me?",
        must_contain_any=[["bullet"]],
      ),
      Probe(
        name="alice isolation from bob",
        prompt="What's my dog's name?",
        must_not_contain=["nova"],
      ),
    ],
  ),
  UserScenario(
    user_id="declare_bob",
    label="Bob",
    sessions=[
      SessionSpec(
        name="profile",
        messages=[
          "Hey Declare, I'm Bob Rivera. I work at Notion.",
          "I live in Austin and my dog's name is Nova.",
          "I'm building a docs sync tool in Rust. Deadline is August 20.",
        ],
      ),
      SessionSpec(
        name="updates",
        messages=[
          "We settled on Postgres and NATS for it.",
          "Please answer in short paragraphs, not bullets.",
          "The deadline slipped to September 5.",
        ],
      ),
    ],
    archive_checks=[
      QueryCheck(name="bob company stored", query="Notion"),
      QueryCheck(name="bob city stored", query="Austin"),
      QueryCheck(name="bob dog stored", query="Nova"),
      QueryCheck(name="bob language stored", query="Rust"),
      QueryCheck(name="bob stack stored", query="Postgres"),
    ],
    working_memory_checks=[
      WorkingMemoryCheck(name="bob current deadline", must_contain="September 5", must_not_contain="August 20"),
      WorkingMemoryCheck(name="bob format preference", must_contain="short paragraph"),
    ],
    probes=[
      Probe(
        name="bob cold recall identity",
        prompt="Where do I work and live?",
        must_contain_any=[["notion"], ["austin"]],
      ),
      Probe(
        name="bob cold recall project",
        prompt="What's my dog's name and current deadline?",
        must_contain_any=[["nova"], ["september 5", "sep 5"]],
      ),
      Probe(
        name="bob cold recall preference",
        prompt="How should you format answers for me?",
        must_contain_any=[["paragraph"]],
        must_not_contain=["bullet points", "bullets"],
      ),
      Probe(
        name="bob isolation from alice",
        prompt="Which city do I live in?",
        must_contain_any=[["austin"]],
        must_not_contain=["san francisco"],
      ),
    ],
  ),
]


class DeclareMemoryV2Evaluator:
  """Runs a compact multi-user memory-v2 evaluation."""

  def __init__(self, db_path: str, model_id: str = "gpt-5-mini"):
    self.db_path = db_path
    self.model_id = model_id
    self.store = SQLiteStore(db_path)
    self.memory = Memory(store=self.store)
    self.turn_metrics: list[TurnMetric] = []
    self.session_metrics: list[SessionMetric] = []
    self.check_results: list[CheckResult] = []

  async def _create_agent(self, user_id: str, session_id: str) -> tuple[Agent, int, int]:
    skill = self.memory.get_skill()
    tools = self.memory.get_tools(user_id=user_id, session_id=session_id)
    wm_block = await self.memory.get_prompt_injection(user_id)
    recent_context = await self.memory.get_session_preamble(user_id, limit=3)

    instructions = (
      "You are Declare, a concise AI assistant. "
      "Keep each reply to one short paragraph or up to 3 bullets. "
      "Use memory tools for durable user facts, corrections, and recall.\n\n"
      f"{wm_block}"
    )
    if recent_context:
      instructions += f"\n\n{recent_context}"

    agent = Agent(
      name="Declare",
      model=OpenAIChat(id=self.model_id, max_completion_tokens=400),
      instructions=instructions,
      skills=[skill],
      tools=tools,
    )
    return agent, len(wm_block), len(recent_context)

  async def _get_user_stats(self, user_id: str) -> dict[str, Any]:
    stats = await self.memory.get_stats(user_id)
    return {
      "wm_chars": stats.wm_chars,
      "wm_version": stats.wm_version,
      "warm_chars": stats.warm_chars,
      "entry_count": stats.entry_count,
      "archive_chars": stats.total_content_chars,
      "categories": dict(stats.categories),
      "oldest_entry": stats.oldest_entry.isoformat() if stats.oldest_entry else None,
      "newest_entry": stats.newest_entry.isoformat() if stats.newest_entry else None,
    }

  async def _run_turn(
    self,
    *,
    scenario: UserScenario,
    session_name: str,
    session_kind: str,
    session_id: str,
    turn_index: int,
    message: str,
    prompt_injection_chars: int,
    recent_context_chars: int,
    history: Optional[list[Any]],
    agent: Agent,
  ) -> tuple[Any, TurnMetric]:
    start = time.perf_counter()
    output = await agent.arun(message, messages=history) if history else await agent.arun(message)
    latency_ms = (time.perf_counter() - start) * 1000

    metrics = output.metrics or Metrics()
    stats = await self._get_user_stats(scenario.user_id)
    audit_issues = await self.memory.post_turn_audit(scenario.user_id)

    turn_metric = TurnMetric(
      user_id=scenario.user_id,
      user_label=scenario.label,
      session_name=session_name,
      session_kind=session_kind,
      session_id=session_id,
      turn_index=turn_index,
      user_message=message,
      agent_response=_response_text(output),
      status=str(output.status),
      tool_calls=_extract_tool_names(output),
      input_tokens=metrics.input_tokens,
      output_tokens=metrics.output_tokens,
      total_tokens=metrics.total_tokens,
      reasoning_tokens=metrics.reasoning_tokens,
      cache_read_tokens=metrics.cache_read_tokens,
      latency_ms=latency_ms,
      time_to_first_token_ms=(metrics.time_to_first_token * 1000) if metrics.time_to_first_token is not None else None,
      history_messages=len(history or []),
      history_chars=_message_char_count(history),
      prompt_injection_chars=prompt_injection_chars,
      recent_context_chars=recent_context_chars,
      working_memory_chars=stats["wm_chars"],
      working_memory_version=stats["wm_version"],
      warm_memory_chars=stats["warm_chars"],
      archive_entry_count=stats["entry_count"],
      archive_total_chars=stats["archive_chars"],
      audit_issues=audit_issues,
    )
    self.turn_metrics.append(turn_metric)
    return output, turn_metric

  async def _run_session(self, scenario: UserScenario, spec: SessionSpec, session_kind: str) -> None:
    session_id = f"{scenario.user_id}-{session_kind}-{spec.name}"
    agent, prompt_injection_chars, recent_context_chars = await self._create_agent(scenario.user_id, session_id)
    stats_before = await self._get_user_stats(scenario.user_id)
    history = None
    session_metric = SessionMetric(
      user_id=scenario.user_id,
      user_label=scenario.label,
      session_name=spec.name,
      session_kind=session_kind,
      session_id=session_id,
      prompt_injection_chars=prompt_injection_chars,
      recent_context_chars=recent_context_chars,
      wm_chars_start=stats_before["wm_chars"],
      archive_count_start=stats_before["entry_count"],
    )

    print(f"\n[{scenario.label}][{session_kind}:{spec.name}]")
    for turn_index, message in enumerate(spec.messages, start=1):
      output, turn_metric = await self._run_turn(
        scenario=scenario,
        session_name=spec.name,
        session_kind=session_kind,
        session_id=session_id,
        turn_index=turn_index,
        message=message,
        prompt_injection_chars=prompt_injection_chars,
        recent_context_chars=recent_context_chars,
        history=history,
        agent=agent,
      )
      history = output.messages

      session_metric.turn_count += 1
      session_metric.input_tokens += turn_metric.input_tokens
      session_metric.output_tokens += turn_metric.output_tokens
      session_metric.total_tokens += turn_metric.total_tokens
      session_metric.reasoning_tokens += turn_metric.reasoning_tokens
      session_metric.cache_read_tokens += turn_metric.cache_read_tokens
      session_metric.latency_ms += turn_metric.latency_ms
      session_metric.tool_calls.extend(turn_metric.tool_calls)

      print(f"  U{turn_index}: {message}")
      print(f"  A{turn_index}: {turn_metric.agent_response}")
      print(
        "       "
        f"tools={turn_metric.tool_calls or ['-']} "
        f"tokens={turn_metric.input_tokens}/{turn_metric.output_tokens}/{turn_metric.total_tokens} "
        f"wm={turn_metric.working_memory_chars} "
        f"archive={turn_metric.archive_entry_count} "
        f"latency={turn_metric.latency_ms:.0f}ms"
      )

    stats_after = await self._get_user_stats(scenario.user_id)
    session_metric.wm_chars_end = stats_after["wm_chars"]
    session_metric.archive_count_end = stats_after["entry_count"]
    self.session_metrics.append(session_metric)

  async def _query_memory(self, user_id: str, query: str) -> tuple[bool, str]:
    entries = await self.store.search_index(user_id=user_id, query=query, limit=10)
    wm = await self.store.get_working_memory(user_id)
    warm = await self.store.get_warm_memory(user_id)
    query_lower = query.lower()

    found_in_wm = query_lower in (wm.content.lower() if wm else "")
    found_in_warm = query_lower in (warm.content.lower() if warm else "")
    found = bool(entries) or found_in_wm or found_in_warm

    details = []
    if entries:
      details.append("hits=" + ", ".join(entry.summary for entry in entries[:3]))
    if found_in_wm:
      details.append("wm=yes")
    if found_in_warm:
      details.append("warm=yes")
    return found, " | ".join(details)

  async def _run_archive_checks(self, scenario: UserScenario) -> None:
    for check in scenario.archive_checks:
      found, details = await self._query_memory(scenario.user_id, check.query)
      passed = found if check.should_exist else not found
      self.check_results.append(
        CheckResult(
          user_id=scenario.user_id,
          user_label=scenario.label,
          kind="archive",
          name=check.name,
          passed=passed,
          details=details,
        )
      )

  async def _run_working_memory_checks(self, scenario: UserScenario) -> None:
    wm = await self.store.get_working_memory(scenario.user_id)
    wm_text = (wm.content if wm else "").lower()

    for check in scenario.working_memory_checks:
      contains_required = check.must_contain.lower() in wm_text
      contains_forbidden = check.must_not_contain.lower() in wm_text if check.must_not_contain else False
      passed = contains_required and not contains_forbidden
      details = f"required={contains_required}"
      if check.must_not_contain:
        details += f" forbidden={contains_forbidden}"
      self.check_results.append(
        CheckResult(
          user_id=scenario.user_id,
          user_label=scenario.label,
          kind="working_memory",
          name=check.name,
          passed=passed,
          details=details,
        )
      )

  async def _run_probe(self, scenario: UserScenario, probe: Probe) -> None:
    spec = SessionSpec(name=probe.name, messages=[probe.prompt])
    await self._run_session(scenario, spec, session_kind="probe")

    response = next(
      turn.agent_response
      for turn in reversed(self.turn_metrics)
      if turn.user_id == scenario.user_id and turn.session_name == probe.name and turn.session_kind == "probe"
    )
    response_lower = response.lower()

    required_results = []
    for group in probe.must_contain_any:
      required_results.append(any(option.lower() in response_lower for option in group))
    forbidden_results = [token.lower() in response_lower for token in probe.must_not_contain]
    passed = all(required_results) and not any(forbidden_results)

    detail_parts = []
    if probe.must_contain_any:
      detail_parts.append(f"required={required_results}")
    if probe.must_not_contain:
      detail_parts.append(f"forbidden={forbidden_results}")
    self.check_results.append(
      CheckResult(
        user_id=scenario.user_id,
        user_label=scenario.label,
        kind="probe",
        name=probe.name,
        passed=passed,
        details=" | ".join(detail_parts),
      )
    )

  async def run(self) -> dict[str, Any]:
    started_at = time.time()
    print("=" * 72)
    print("DECLARE AGENT MEMORY V2 EVALUATION")
    print(f"Model: {self.model_id}")
    print(f"DB: {self.db_path}")
    print("=" * 72)

    try:
      for scenario in SCENARIOS:
        for spec in scenario.sessions:
          await self._run_session(scenario, spec, session_kind="seed")

      print("\n" + "=" * 72)
      print("STORE CHECKS")
      print("=" * 72)
      for scenario in SCENARIOS:
        await self._run_archive_checks(scenario)
        await self._run_working_memory_checks(scenario)

      print("\n" + "=" * 72)
      print("COLD-START PROBES")
      print("=" * 72)
      for scenario in SCENARIOS:
        for probe in scenario.probes:
          await self._run_probe(scenario, probe)

      report = await self._build_report(started_at)
      self._print_report(report)
      return report
    finally:
      await self.memory.close()

  async def _build_report(self, started_at: float) -> dict[str, Any]:
    ended_at = time.time()
    per_user: dict[str, Any] = {}
    for scenario in SCENARIOS:
      turns = [turn for turn in self.turn_metrics if turn.user_id == scenario.user_id]
      sessions = [session for session in self.session_metrics if session.user_id == scenario.user_id]
      checks = [check for check in self.check_results if check.user_id == scenario.user_id]
      stats = await self._get_user_stats(scenario.user_id)
      wm = await self.store.get_working_memory(scenario.user_id)

      per_user[scenario.user_id] = {
        "label": scenario.label,
        "final_memory": {
          **stats,
          "working_memory": wm.content if wm else "",
        },
        "turns": [asdict(turn) for turn in turns],
        "sessions": [asdict(session) for session in sessions],
        "checks": [asdict(check) for check in checks],
        "summary": {
          "turns": len(turns),
          "sessions": len(sessions),
          "checks_passed": sum(1 for check in checks if check.passed),
          "checks_total": len(checks),
          "input_tokens": sum(turn.input_tokens for turn in turns),
          "output_tokens": sum(turn.output_tokens for turn in turns),
          "total_tokens": sum(turn.total_tokens for turn in turns),
          "reasoning_tokens": sum(turn.reasoning_tokens for turn in turns),
          "latency_ms_avg": (sum(turn.latency_ms for turn in turns) / len(turns) if turns else 0.0),
          "tool_calls": dict(Counter(tool for turn in turns for tool in turn.tool_calls)),
        },
      }

    latencies = [turn.latency_ms for turn in self.turn_metrics]
    token_totals = [turn.total_tokens for turn in self.turn_metrics]
    aggregate_checks = {
      "passed": sum(1 for check in self.check_results if check.passed),
      "total": len(self.check_results),
      "by_kind": {
        kind: {
          "passed": sum(1 for check in self.check_results if check.kind == kind and check.passed),
          "total": sum(1 for check in self.check_results if check.kind == kind),
        }
        for kind in sorted({check.kind for check in self.check_results})
      },
    }

    return {
      "model_id": self.model_id,
      "db_path": self.db_path,
      "started_at": started_at,
      "ended_at": ended_at,
      "duration_s": ended_at - started_at,
      "aggregate": {
        "users": len(SCENARIOS),
        "sessions": len(self.session_metrics),
        "turns": len(self.turn_metrics),
        "checks": aggregate_checks,
        "tokens": {
          "input": sum(turn.input_tokens for turn in self.turn_metrics),
          "output": sum(turn.output_tokens for turn in self.turn_metrics),
          "total": sum(turn.total_tokens for turn in self.turn_metrics),
          "reasoning": sum(turn.reasoning_tokens for turn in self.turn_metrics),
          "cache_read": sum(turn.cache_read_tokens for turn in self.turn_metrics),
          "avg_total_per_turn": (sum(token_totals) / len(token_totals)) if token_totals else 0.0,
        },
        "latency_ms": {
          "avg": (sum(latencies) / len(latencies)) if latencies else 0.0,
          "p50": _percentile(latencies, 0.50),
          "p95": _percentile(latencies, 0.95),
          "max": max(latencies) if latencies else 0.0,
        },
        "tool_calls": dict(Counter(tool for turn in self.turn_metrics for tool in turn.tool_calls)),
      },
      "per_user": per_user,
      "checks": [asdict(check) for check in self.check_results],
    }

  def _print_report(self, report: dict[str, Any]) -> None:
    aggregate = report["aggregate"]
    checks = aggregate["checks"]

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Checks: {checks['passed']}/{checks['total']} ({(100 * checks['passed'] / checks['total']) if checks['total'] else 0:.1f}%)")
    for kind, stats in checks["by_kind"].items():
      ratio = (100 * stats["passed"] / stats["total"]) if stats["total"] else 0.0
      print(f"  {kind}: {stats['passed']}/{stats['total']} ({ratio:.1f}%)")

    print(
      "\nTokens: "
      f"in={aggregate['tokens']['input']} "
      f"out={aggregate['tokens']['output']} "
      f"total={aggregate['tokens']['total']} "
      f"avg/turn={aggregate['tokens']['avg_total_per_turn']:.1f}"
    )
    print(
      "Latency: "
      f"avg={aggregate['latency_ms']['avg']:.0f}ms "
      f"p50={aggregate['latency_ms']['p50']:.0f}ms "
      f"p95={aggregate['latency_ms']['p95']:.0f}ms "
      f"max={aggregate['latency_ms']['max']:.0f}ms"
    )
    print(f"Tool calls: {aggregate['tool_calls']}")

    for user_id, user_report in report["per_user"].items():
      summary = user_report["summary"]
      final_memory = user_report["final_memory"]
      print("\n" + "-" * 72)
      print(f"{user_report['label']} ({user_id})")
      print(f"  Checks: {summary['checks_passed']}/{summary['checks_total']} | Turns: {summary['turns']} | Sessions: {summary['sessions']}")
      print(
        "  Tokens: "
        f"in={summary['input_tokens']} "
        f"out={summary['output_tokens']} "
        f"total={summary['total_tokens']} "
        f"latency_avg={summary['latency_ms_avg']:.0f}ms"
      )
      print(
        "  Memory: "
        f"wm={final_memory['wm_chars']} chars "
        f"(v{final_memory['wm_version']}), "
        f"warm={final_memory['warm_chars']} chars, "
        f"archive={final_memory['entry_count']} entries / {final_memory['archive_chars']} chars"
      )
      print(f"  Tool calls: {summary['tool_calls']}")

    failed = [check for check in report["checks"] if not check["passed"]]
    if failed:
      print("\nFailed checks:")
      for check in failed:
        print(f"  - [{check['kind']}] {check['user_label']}: {check['name']} :: {check['details']}")


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
  temp_db: Optional[str] = None
  db_path = args.db_path
  if not db_path:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as handle:
      temp_db = handle.name
      db_path = handle.name

  evaluator = DeclareMemoryV2Evaluator(db_path=db_path, model_id=args.model)
  report = await evaluator.run()

  if args.json_out:
    Path(args.json_out).write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"\nJSON report written to: {args.json_out}")

  if temp_db and not args.keep_db:
    for suffix in ["", "-shm", "-wal"]:
      path = Path(temp_db + suffix)
      if path.exists():
        path.unlink()
  elif temp_db:
    print(f"\nTemporary DB kept at: {temp_db}")

  return report


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Run the Declare agent memory-v2 evaluation.")
  parser.add_argument("--model", default="gpt-5-mini", help="Model id to use. Default: gpt-5-mini")
  parser.add_argument("--db-path", default="", help="Optional SQLite DB path. Defaults to a temp file.")
  parser.add_argument("--json-out", default="", help="Optional path for a machine-readable JSON report.")
  parser.add_argument("--keep-db", action="store_true", help="Keep the temp DB file after the run.")
  return parser


def main() -> None:
  args = build_parser().parse_args()
  asyncio.run(async_main(args))


if __name__ == "__main__":
  main()
