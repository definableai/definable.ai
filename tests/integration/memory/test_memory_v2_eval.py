"""
Memory v2 Evaluation — Realistic Human-AI Conversation Journey.

Tests the tool-based memory system (v2) through realistic multi-session
conversations, evaluating:

1. ACCURACY — Does the LLM extract relevant memories? Are they correct?
2. SIZE — How big does memory grow? What's the token overhead?

The test simulates a real user journey:
  Session 1: Introduction & personal info
  Session 2: Project discussion & technical context
  Session 3: Preferences, corrections & feedback
  Session 4: Recall across sessions (cold start)
  Session 5: High-volume information dump (stress test)
  Session 6: Contradiction handling & memory updates
  Session 7: Forgetting & privacy requests
  Session 8: Long-term coherence check

Requirements:
    export OPENAI_API_KEY=sk-...
"""

import asyncio
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from definable.agent import Agent
from definable.memory.v2 import Memory, SQLiteStore
from definable.model.openai import OpenAIChat


# ── Metrics Collection ──────────────────────────────────────────────


@dataclass
class TurnMetric:
  """Metrics for a single conversation turn."""

  session: int
  turn: int
  user_message: str
  agent_response: str
  tool_calls_made: list[str] = field(default_factory=list)
  working_memory_size: int = 0
  archive_entry_count: int = 0
  total_archive_chars: int = 0
  latency_ms: float = 0.0


@dataclass
class SessionMetric:
  """Aggregate metrics for a session."""

  session_id: int
  turns: list[TurnMetric] = field(default_factory=list)
  wm_size_start: int = 0
  wm_size_end: int = 0
  archive_count_start: int = 0
  archive_count_end: int = 0


@dataclass
class AccuracyCheck:
  """A specific fact we expect the system to remember or forget."""

  description: str
  query: str  # What to search for
  expected_present: bool  # Should this fact exist in memory?
  found: bool = False
  details: str = ""


# ── Test Harness ────────────────────────────────────────────────────


class MemoryV2Evaluator:
  """Evaluates memory v2 through realistic conversation journeys."""

  def __init__(self, db_path: str, model_id: str = "gpt-4o-mini"):
    self.db_path = db_path
    self.model_id = model_id
    self.user_id = "eval_user_alice"
    self.store = SQLiteStore(db_path)
    self.memory = Memory(store=self.store)
    self.session_metrics: list[SessionMetric] = []
    self.accuracy_checks: list[AccuracyCheck] = []
    self._session_counter = 0

  async def _create_agent(self, session_id: str) -> Agent:
    """Create an agent with v2 memory tools wired in."""
    skill = self.memory.get_skill()
    tools = self.memory.get_tools(user_id=self.user_id, session_id=session_id)

    # Get working memory injection for the prompt
    wm_block = await self.memory.get_prompt_injection(self.user_id)

    instructions = (
      "You are a helpful personal assistant named Aria. "
      "You remember everything the user tells you across conversations. "
      "Be concise in your responses — 1-3 sentences max.\n\n"
      f"{wm_block}"
    )

    return Agent(
      model=OpenAIChat(id=self.model_id),
      instructions=instructions,
      skills=[skill],
      tools=tools,
    )

  async def _get_store_stats(self) -> tuple[int, int, int]:
    """Get current memory size stats: (wm_chars, archive_count, archive_total_chars)."""
    wm = await self.store.get_working_memory(self.user_id)
    wm_chars = len(wm.content) if wm else 0

    entries = await self.store.search_index(user_id=self.user_id, limit=1000)
    archive_count = len(entries)

    # Sum up all archived content sizes
    if entries:
      entry_ids = [e.id for e in entries]
      full_entries = await self.store.get_entries(entry_ids)
      archive_chars = sum(len(e.content) for e in full_entries)
    else:
      archive_chars = 0

    return wm_chars, archive_count, archive_chars

  async def _run_turn(
    self,
    agent: Agent,
    message: str,
    session_num: int,
    turn_num: int,
    prev_messages: Optional[list] = None,
  ) -> tuple[Any, TurnMetric]:
    """Run a single conversation turn and collect metrics."""
    start = time.monotonic()

    kwargs: dict[str, Any] = {}
    if prev_messages:
      kwargs["messages"] = prev_messages

    output = await agent.arun(message, **kwargs)

    latency = (time.monotonic() - start) * 1000

    # Extract tool calls from the output
    tool_calls = []
    for msg in output.messages or []:
      if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
          tool_calls.append(tc.get("function", {}).get("name", "unknown") if isinstance(tc, dict) else getattr(tc, "name", "unknown"))

    wm_chars, archive_count, archive_chars = await self._get_store_stats()

    metric = TurnMetric(
      session=session_num,
      turn=turn_num,
      user_message=message[:100],
      agent_response=(output.content or "")[:200],
      tool_calls_made=tool_calls,
      working_memory_size=wm_chars,
      archive_entry_count=archive_count,
      total_archive_chars=archive_chars,
      latency_ms=latency,
    )

    return output, metric

  async def _run_session(self, session_num: int, messages: list[str]) -> SessionMetric:
    """Run a full conversation session with multiple turns."""
    self._session_counter += 1
    session_id = f"session_{self._session_counter}"

    wm_start, arc_start, _ = await self._get_store_stats()
    session_metric = SessionMetric(
      session_id=session_num,
      wm_size_start=wm_start,
      archive_count_start=arc_start,
    )

    agent = await self._create_agent(session_id)
    prev_messages = None

    for i, msg in enumerate(messages):
      output, turn_metric = await self._run_turn(agent, msg, session_num, i + 1, prev_messages)
      session_metric.turns.append(turn_metric)
      prev_messages = output.messages
      print(f"  Turn {i + 1}: User: {msg[:60]}...")
      print(f"           Aria: {(output.content or '')[:80]}...")
      print(f"           Tools: {turn_metric.tool_calls_made}")
      print(f"           WM: {turn_metric.working_memory_size} chars | Archive: {turn_metric.archive_entry_count} entries")
      print()

    wm_end, arc_end, _ = await self._get_store_stats()
    session_metric.wm_size_end = wm_end
    session_metric.archive_count_end = arc_end

    self.session_metrics.append(session_metric)
    return session_metric

  async def _check_accuracy(self, description: str, query: str, expected_present: bool) -> AccuracyCheck:
    """Check if a specific fact exists in memory (searches summaries, content, AND working memory)."""
    check = AccuracyCheck(
      description=description,
      query=query,
      expected_present=expected_present,
    )

    # Search archive via FTS (now searches summary + tags + content)
    entries = await self.store.search_index(user_id=self.user_id, query=query, limit=20)

    # Also check working memory
    wm = await self.store.get_working_memory(self.user_id)
    wm_content = (wm.content if wm else "").lower()

    query_lower = query.lower()

    # Check 1: FTS returned results (means the query matched somewhere)
    found_via_fts = len(entries) > 0

    # Check 2: Query text in summary
    found_in_summary = any(query_lower in e.summary.lower() for e in entries)

    # Check 3: Query text in full content of returned entries
    found_in_content = False
    if entries:
      entry_ids = [e.id for e in entries]
      full_entries = await self.store.get_entries(entry_ids)
      found_in_content = any(query_lower in e.content.lower() for e in full_entries)

    # Check 4: Query text in working memory
    found_in_wm = query_lower in wm_content

    # Check 5: Broader matching (key terms from query)
    key_terms = [t for t in query_lower.split() if len(t) > 3]
    broad_wm_match = all(term in wm_content for term in key_terms) if key_terms else False

    check.found = found_via_fts or found_in_summary or found_in_content or found_in_wm or broad_wm_match

    # Build details string
    details_parts = []
    if entries:
      details_parts.append(f"FTS hits: {[e.summary[:60] for e in entries[:3]]}")
    if found_in_content:
      details_parts.append("In content: yes")
    if found_in_wm:
      details_parts.append("In WM: yes")
    check.details = " | ".join(details_parts)

    self.accuracy_checks.append(check)
    return check

  # ── Conversation Sessions ─────────────────────────────────────────

  async def session_1_introduction(self):
    """Session 1: Basic introduction and personal info."""
    print("\n" + "=" * 60)
    print("SESSION 1: Introduction & Personal Info")
    print("=" * 60)
    await self._run_session(
      1,
      [
        "Hi! I'm Alice Chen. I'm a senior data scientist at Stripe.",
        "I live in San Francisco and I've been coding in Python for about 8 years now.",
        "My team is working on fraud detection models. We use PyTorch and scikit-learn mostly.",
        "Oh and I prefer dark mode in everything. And I hate when people send me PDFs instead of markdown.",
      ],
    )

  async def session_2_project_discussion(self):
    """Session 2: Deeper project context."""
    print("\n" + "=" * 60)
    print("SESSION 2: Project Discussion & Technical Context")
    print("=" * 60)
    await self._run_session(
      2,
      [
        "Hey Aria, I need to talk about our Q2 project.",
        "We're building a real-time transaction scoring system. The deadline is June 15th.",
        "The architecture uses Kafka for streaming, and we store features in Redis. The model runs on an NVIDIA A100 GPU cluster.",
        "My main concern is latency — we need sub-50ms inference time per transaction.",
        "Can you remember that our team lead is Marcus and the PM is Sarah? They'll come up in future discussions.",
      ],
    )

  async def session_3_preferences_and_corrections(self):
    """Session 3: Preferences, corrections, feedback."""
    print("\n" + "=" * 60)
    print("SESSION 3: Preferences, Corrections & Feedback")
    print("=" * 60)
    await self._run_session(
      3,
      [
        "Quick correction — I said I use scikit-learn but actually we switched to XGBoost last month.",
        "When I ask you to write code, always use type hints and Google-style docstrings. That's how our team does it.",
        "Also, never suggest using pandas for large datasets. We use Polars exclusively.",
        "I like when you give me bullet points instead of long paragraphs.",
        "Oh, and my work email is alice.chen@stripe.com. I'll need you to remember that.",
      ],
    )

  async def session_4_cold_start_recall(self):
    """Session 4: Fresh session — test recall from cold start."""
    print("\n" + "=" * 60)
    print("SESSION 4: Cold Start Recall Test")
    print("=" * 60)
    await self._run_session(
      4,
      [
        "Hey, what do you remember about me?",
        "What project am I working on and when is the deadline?",
        "What are my coding preferences?",
        "Who are my team members?",
      ],
    )

  async def session_5_stress_test(self):
    """Session 5: Dump a lot of information quickly."""
    print("\n" + "=" * 60)
    print("SESSION 5: High-Volume Information Stress Test")
    print("=" * 60)
    await self._run_session(
      5,
      [
        (
          "Let me tell you about a bunch of things at once. "
          "My favorite programming languages ranked: Python, Rust, Go, TypeScript. "
          "I have a golden retriever named Pixel. "
          "My birthday is March 12th. "
          "I'm allergic to shellfish. "
          "I graduated from MIT in 2016 with a CS degree. "
          "My manager's name is David Park. Wait, actually Marcus is the team lead, David is the VP."
        ),
        (
          "More context: I'm also a part-time instructor at Stanford teaching ML. "
          "I run a blog at alicechen.dev. "
          "I use VS Code with Vim keybindings. "
          "My favorite coffee is a cortado from Blue Bottle. "
          "I'm training for the SF marathon in October."
        ),
        (
          "For the fraud detection project specifically: "
          "we're using feature store version 3.2, "
          "the model is a gradient boosted ensemble with 500 trees, "
          "we have 2.3 billion historical transactions, "
          "and our current F1 score is 0.94 but we need to hit 0.97 for production."
        ),
      ],
    )

  async def session_6_contradictions(self):
    """Session 6: Contradictions and updates to existing facts."""
    print("\n" + "=" * 60)
    print("SESSION 6: Contradiction Handling & Memory Updates")
    print("=" * 60)
    await self._run_session(
      6,
      [
        "Hey, update on the project — the deadline moved from June 15th to July 1st.",
        "Also, I got promoted! I'm now a Staff Data Scientist, not Senior.",
        "We decided to drop Redis and use DynamoDB instead for the feature store.",
        "One more thing — Sarah left the company. The new PM is Jason Liu.",
      ],
    )

  async def session_7_forgetting(self):
    """Session 7: Explicit forget requests and privacy."""
    print("\n" + "=" * 60)
    print("SESSION 7: Forgetting & Privacy Requests")
    print("=" * 60)
    await self._run_session(
      7,
      [
        "Can you forget my email address? I don't want that stored.",
        "Also forget the thing about my allergy — that's too personal.",
        "What do you still know about me? Give me a summary.",
      ],
    )

  async def session_8_coherence_check(self):
    """Session 8: Final coherence — does everything still make sense?"""
    print("\n" + "=" * 60)
    print("SESSION 8: Long-Term Coherence Check")
    print("=" * 60)
    await self._run_session(
      8,
      [
        "What's my current role and where do I work?",
        "What's the project deadline now?",
        "What tech stack are we using for the fraud detection system?",
        "What are my preferences for code style?",
        "Do you remember anything about my email or allergies? You shouldn't.",
      ],
    )

  # ── Accuracy Evaluation ───────────────────────────────────────────

  async def run_accuracy_checks(self):
    """Run all accuracy checks against the final memory state."""
    print("\n" + "=" * 60)
    print("ACCURACY CHECKS")
    print("=" * 60)

    # --- Recall checks: facts that should be remembered ---
    recall_checks = [
      ("Name: Alice Chen", "Alice Chen"),
      ("Company: Stripe", "Stripe"),
      ("City: San Francisco", "San Francisco"),
      ("Role: Staff Data Scientist (updated)", "Staff"),
      ("Python experience: 8 years", "Python"),
      ("Project: fraud detection", "fraud detection"),
      ("Deadline: July 1st (updated)", "July"),
      ("Tech: Kafka streaming", "Kafka"),
      ("Tech: DynamoDB (replaced Redis)", "DynamoDB"),
      ("Team: Marcus (team lead)", "Marcus"),
      ("Team: Jason Liu (new PM)", "Jason"),
      ("Preference: type hints + Google docstrings", "type hints"),
      ("Preference: Polars over pandas", "Polars"),
      ("Preference: bullet points", "bullet"),
      ("Preference: dark mode", "dark mode"),
      ("Dog: Pixel (golden retriever)", "Pixel"),
      ("Education: MIT 2016", "MIT"),
      ("Side: Stanford instructor", "Stanford"),
      ("Blog: alicechen.dev", "alicechen"),
      ("Model: XGBoost (corrected from sklearn)", "XGBoost"),
      ("Target: F1 0.97", "0.97"),
      ("Marathon: SF October", "marathon"),
    ]

    print("\n  --- RECALL (should remember) ---")
    for desc, query in recall_checks:
      check = await self._check_accuracy(desc, query, True)
      status = "PASS" if check.found else "FAIL"
      symbol = "+" if status == "PASS" else "x"
      print(f"  [{symbol}] {status}: {desc}")
      if check.details:
        print(f"       {check.details[:120]}")
      print()

    # --- Forget checks: explicitly deleted facts ---
    forget_checks = [
      ("Email: should be forgotten", "alice.chen@stripe.com"),
      ("Allergy: should be forgotten", "shellfish"),
    ]

    print("  --- FORGET (explicitly deleted) ---")
    for desc, query in forget_checks:
      check = await self._check_accuracy(desc, query, False)
      status = "PASS" if not check.found else "FAIL"
      symbol = "+" if status == "PASS" else "x"
      print(f"  [{symbol}] {status}: {desc}")
      if check.details:
        print(f"       {check.details[:120]}")
      print()

    # --- Supersession checks: WM has the CORRECT current value ---
    # For these, we check that the WORKING MEMORY contains the new value,
    # not whether old archive entries still exist. The agent's behavior is
    # determined by WM (always loaded), not stale archive entries.
    supersession_checks = [
      ("WM has Staff (not Senior)", "Staff", "Senior Data Scientist"),
      ("WM has July (not June 15)", "July", "June 15"),
      ("WM has Jason (not Sarah as PM)", "Jason", None),
      ("WM has DynamoDB (not Redis)", "DynamoDB", "Redis"),
    ]

    print("  --- SUPERSESSION (WM has correct current value) ---")
    wm = await self.store.get_working_memory(self.user_id)
    wm_text = (wm.content if wm else "").lower()
    for desc, new_val, old_val in supersession_checks:
      new_in_wm = new_val.lower() in wm_text
      old_in_wm = old_val.lower() in wm_text if old_val else False
      passed = new_in_wm and not old_in_wm
      symbol = "+" if passed else "x"
      status = "PASS" if passed else "FAIL"
      detail = f"new='{new_val}' in WM: {new_in_wm}"
      if old_val:
        detail += f" | old='{old_val}' in WM: {old_in_wm}"
      print(f"  [{symbol}] {status}: {desc}")
      print(f"       {detail}")
      # Track in accuracy_checks for report
      acc = AccuracyCheck(description=desc, query=new_val, expected_present=True, found=new_in_wm)
      acc.details = detail
      self.accuracy_checks.append(acc)
      print()

  # ── Report ────────────────────────────────────────────────────────

  def print_report(self):
    """Print the full evaluation report."""
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    # ── Accuracy Summary
    total = len(self.accuracy_checks)
    passed = sum(1 for c in self.accuracy_checks if c.found == c.expected_present)
    failed = total - passed

    should_present = [c for c in self.accuracy_checks if c.expected_present]
    should_absent = [c for c in self.accuracy_checks if not c.expected_present]

    recall_correct = sum(1 for c in should_present if c.found)
    forget_correct = sum(1 for c in should_absent if not c.found)

    print("\n## Accuracy")
    print(f"  Overall: {passed}/{total} ({100 * passed / total:.1f}%)")
    print(f"  Recall accuracy (should remember): {recall_correct}/{len(should_present)} ({100 * recall_correct / len(should_present):.1f}%)")
    print(f"  Forget accuracy (should forget): {forget_correct}/{len(should_absent)} ({100 * forget_correct / len(should_absent):.1f}%)")
    print()

    if failed > 0:
      print("  Failed checks:")
      for c in self.accuracy_checks:
        if c.found != c.expected_present:
          expected_str = "present" if c.expected_present else "absent"
          print(f"    - {c.description} (expected {expected_str})")
      print()

    # ── Memory Size Growth
    print("## Memory Size Growth")
    print(f"  {'Session':<10} {'WM Start':<12} {'WM End':<12} {'Archive Start':<15} {'Archive End':<15} {'WM Delta':<12} {'Arc Delta':<12}")
    print(f"  {'-' * 88}")
    for sm in self.session_metrics:
      wm_delta = sm.wm_size_end - sm.wm_size_start
      arc_delta = sm.archive_count_end - sm.archive_count_start
      print(
        f"  {sm.session_id:<10} {sm.wm_size_start:<12} {sm.wm_size_end:<12} "
        f"{sm.archive_count_start:<15} {sm.archive_count_end:<15} "
        f"{'+' if wm_delta >= 0 else ''}{wm_delta:<11} {'+' if arc_delta >= 0 else ''}{arc_delta:<11}"
      )
    print()

    # Final state
    if self.session_metrics:
      final = self.session_metrics[-1]
      last_turn = final.turns[-1] if final.turns else None
      if last_turn:
        print("  Final state:")
        print(f"    Working memory: {last_turn.working_memory_size} chars (~{last_turn.working_memory_size // 4} tokens)")
        print(f"    Archive entries: {last_turn.archive_entry_count}")
        print(f"    Archive total: {last_turn.total_archive_chars} chars (~{last_turn.total_archive_chars // 4} tokens)")
        total_memory = last_turn.working_memory_size + last_turn.total_archive_chars
        print(f"    Total memory footprint: {total_memory} chars (~{total_memory // 4} tokens)")
        print()

    # ── Tool Usage Summary
    print("## Tool Usage per Session")
    for sm in self.session_metrics:
      all_tools = []
      for t in sm.turns:
        all_tools.extend(t.tool_calls_made)
      tool_counts: dict[str, int] = {}
      for tc in all_tools:
        tool_counts[str(tc)] = tool_counts.get(str(tc), 0) + 1
      print(f"  Session {sm.session_id}: {tool_counts}")
    print()

    # ── Latency
    all_latencies = [t.latency_ms for sm in self.session_metrics for t in sm.turns]
    if all_latencies:
      avg_latency = sum(all_latencies) / len(all_latencies)
      max_latency = max(all_latencies)
      min_latency = min(all_latencies)
      print("## Latency")
      print(f"  Average: {avg_latency:.0f}ms")
      print(f"  Min: {min_latency:.0f}ms | Max: {max_latency:.0f}ms")
      print()

    # ── Working Memory Content
    print("## Final Working Memory Content")
    print("-" * 40)

  async def print_final_memory_state(self):
    """Print the actual contents of working memory and archive."""
    wm = await self.store.get_working_memory(self.user_id)
    if wm:
      print(wm.content)
    else:
      print("(empty)")
    print("-" * 40)

    print("\n## Archived Memory Entries")
    entries = await self.store.search_index(user_id=self.user_id, limit=100)
    if entries:
      entry_ids = [e.id for e in entries]
      full_entries = await self.store.get_entries(entry_ids)
      entry_map = {e.id: e for e in full_entries}
      for i, idx in enumerate(entries, 1):
        full = entry_map.get(idx.id)
        content_preview = (full.content[:120] + "...") if full and len(full.content) > 120 else (full.content if full else "N/A")
        print(f"  [{i}] ({idx.category}) {idx.summary}")
        print(f"      Tags: {idx.tags}")
        print(f"      Content: {content_preview}")
        print()
    else:
      print("  (no archived entries)")

  # ── Main Runner ───────────────────────────────────────────────────

  async def run_full_evaluation(self):
    """Run the complete evaluation journey."""
    print("=" * 60)
    print("MEMORY V2 EVALUATION — Realistic Conversation Journey")
    print(f"Model: {self.model_id}")
    print(f"DB: {self.db_path}")
    print("=" * 60)

    try:
      # Phase 1: Run all conversation sessions
      await self.session_1_introduction()
      await self.session_2_project_discussion()
      await self.session_3_preferences_and_corrections()
      await self.session_4_cold_start_recall()
      await self.session_5_stress_test()
      await self.session_6_contradictions()
      await self.session_7_forgetting()
      await self.session_8_coherence_check()

      # Phase 2: Accuracy evaluation
      await self.run_accuracy_checks()

      # Phase 3: Report
      self.print_report()
      await self.print_final_memory_state()

    finally:
      await self.memory.close()


async def main():
  # Use a temp file for the DB so we start clean
  with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
    db_path = f.name

  try:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o-mini"
    evaluator = MemoryV2Evaluator(db_path=db_path, model_id=model_id)
    await evaluator.run_full_evaluation()
  finally:
    # Clean up
    for ext in ["", "-shm", "-wal"]:
      p = Path(db_path + ext)
      if p.exists():
        p.unlink()


if __name__ == "__main__":
  asyncio.run(main())
