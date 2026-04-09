"""
Live LoCoMo evaluation for memory v2.

This standalone integration script:
  - ingests each LoCoMo conversation session into memory v2
  - cold-starts question answering from memory
  - scores answers with the official LoCoMo QA metric logic

Usage:
    export OPENAI_API_KEY=sk-...
    ./.venv/bin/python definable/tests/integration/memory/test_memory_v2_locomo_eval.py \
      --data-file /tmp/locomo/data/locomo10.json \
      --model gpt-4o-mini

    ./.venv/bin/python definable/tests/integration/memory/test_memory_v2_locomo_eval.py \
      --data-file /tmp/locomo/data/locomo10.json \
      --model gpt-4o-mini \
      --max-samples 1 \
      --max-questions 25 \
      --output-json /tmp/locomo-memory-v2-smoke.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import string
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import regex
from nltk.stem import PorterStemmer
from pydantic import BaseModel, Field, create_model

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from definable.agent import Agent
from definable.memory.v2 import Memory, SQLiteStore
from definable.memory.v2.models import MemoryStats
from definable.model.metrics import Metrics
from definable.model.openai import OpenAIChat

ps = PorterStemmer()

INGEST_PREFIX = "INGEST SESSION"
ANSWER_PREFIX = "ANSWER QUESTIONS"

AGENT_INSTRUCTIONS = """
You are evaluating long-term conversational memory over a benchmark transcript.

You operate in two modes:

1. If the user message starts with "INGEST SESSION":
   - Treat the message as an observed transcript between two people over time.
   - Update memory so you can later answer exact factual questions about both speakers.
   - Store concrete facts, relationships, dates, places, projects, preferences, corrections, and time-ordered events.
   - Prefer highly specific memory summaries with searchable nouns and values.
   - Tool-call syntax matters:
     * `update_working_memory` takes exactly one argument named `content`
     * `archive_to_memory` must include both `summary` and `content`
     * Use plain argument names like `content`, never `content:`
   - For `archive_to_memory.content`, format it as:
     Fact: ...
     Why: ...
     How to apply: ...
   - Reply with exactly: ACK

2. If the user message starts with "ANSWER QUESTIONS":
   - Answer using only remembered information from working memory and archived memory.
   - Search archived memory before concluding something is unavailable.
   - Do not invent.
   - Do not rewrite or add memory unless it is strictly necessary to retrieve already-stored information.
   - Return very short answers.
   - If the answer is unavailable, return exactly: Not mentioned in the conversation
   - For date questions, answer with an approximate calendar date grounded in the remembered conversation.
"""


@dataclass
class SampleResult:
  sample_id: str
  sessions_ingested: int
  questions_answered: int
  overall_score: float
  category_scores: dict[int, float] = field(default_factory=dict)
  wm_chars: int = 0
  wm_version: int = 0
  warm_chars: int = 0
  archive_entries: int = 0
  archive_chars: int = 0
  total_tokens: int = 0
  input_tokens: int = 0
  output_tokens: int = 0
  duration_s: float = 0.0


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--data-file", required=True, type=Path)
  parser.add_argument("--model", default="gpt-4o-mini")
  parser.add_argument("--output-json", type=Path)
  parser.add_argument("--max-samples", type=int)
  parser.add_argument("--sample-id", action="append", default=[])
  parser.add_argument("--max-questions", type=int)
  parser.add_argument("--question-batch-size", type=int, default=12)
  parser.add_argument("--sessions-per-ingest-chunk", type=int, default=4)
  parser.add_argument("--seed", type=int, default=7)
  return parser.parse_args()


def normalize_answer(text: str) -> str:
  text = text.replace(",", "")

  def remove_articles(value: str) -> str:
    return regex.sub(r"\b(a|an|the|and)\b", " ", value)

  def white_space_fix(value: str) -> str:
    return " ".join(value.split())

  def remove_punc(value: str) -> str:
    exclude = set(string.punctuation)
    return "".join(ch for ch in value if ch not in exclude)

  return white_space_fix(remove_articles(remove_punc(text.lower())))


def f1_score(prediction: str, ground_truth: str) -> float:
  prediction_tokens = [ps.stem(word) for word in normalize_answer(prediction).split()]
  ground_truth_tokens = [ps.stem(word) for word in normalize_answer(ground_truth).split()]
  common = defaultdict(int)
  for token in prediction_tokens:
    common[token] += 1

  num_same = 0
  for token in ground_truth_tokens:
    if common[token] > 0:
      common[token] -= 1
      num_same += 1

  if num_same == 0 or not prediction_tokens or not ground_truth_tokens:
    return 0.0

  precision = num_same / len(prediction_tokens)
  recall = num_same / len(ground_truth_tokens)
  return (2 * precision * recall) / (precision + recall)


def multi_answer_f1(prediction: str, ground_truth: str) -> float:
  predictions = [part.strip() for part in prediction.split(",") if part.strip()]
  truths = [part.strip() for part in ground_truth.split(",") if part.strip()]
  if not predictions or not truths:
    return 0.0
  return sum(max(f1_score(candidate, truth) for candidate in predictions) for truth in truths) / len(truths)


def score_prediction(qa: dict[str, Any], prediction: str) -> float:
  answer = get_answer_text(qa)
  if qa["category"] == 3:
    answer = answer.split(";")[0].strip()

  if qa["category"] in {2, 3, 4}:
    return f1_score(prediction, answer)
  if qa["category"] == 1:
    return multi_answer_f1(prediction, answer)
  if qa["category"] == 5:
    lowered = prediction.lower()
    if "no information available" in lowered or "not mentioned" in lowered:
      return 1.0
    return 0.0
  raise ValueError(f"Unsupported LoCoMo category: {qa['category']}")


def _format_dialog_turn(dialog: dict[str, Any]) -> str:
  turn = f'{dialog["speaker"]} said, "{dialog["text"]}"'
  if dialog.get("blip_caption"):
    turn += f" and shared {dialog['blip_caption']}"
  return turn


def get_answer_text(qa: dict[str, Any]) -> str:
  if "answer" in qa and qa["answer"] is not None:
    return qa["answer"] if isinstance(qa["answer"], str) else str(qa["answer"])
  if "adversarial_answer" in qa and qa["adversarial_answer"] is not None:
    return qa["adversarial_answer"] if isinstance(qa["adversarial_answer"], str) else str(qa["adversarial_answer"])
  raise KeyError(f"QA item has no supported answer field: {qa.keys()}")


def build_ingest_prompt(sample: dict[str, Any], session_nums: list[int]) -> str:
  conversation = sample["conversation"]
  parts = []
  for session_num in session_nums:
    date_time = conversation[f"session_{session_num}_date_time"]
    turns = "\n".join(_format_dialog_turn(dialog) for dialog in conversation[f"session_{session_num}"])
    parts.append(f"Session: {session_num}\nDATE: {date_time}\nCONVERSATION:\n{turns}")
  return (
    f"{INGEST_PREFIX}\n"
    f"Sample: {sample['sample_id']}\n"
    f"Speakers: {conversation['speaker_a']} and {conversation['speaker_b']}\n"
    f"Included sessions: {session_nums}\n\n" + "\n\n".join(parts) + "\n\n"
    "Update memory with the concrete facts from this transcript and then reply with ACK."
  )


def build_question_text(qa: dict[str, Any]) -> str:
  if qa["category"] == 2:
    return qa["question"] + " Use the remembered conversation date to answer with an approximate calendar date."
  if qa["category"] == 5:
    return qa["question"] + f" Select the correct answer: (a) {get_answer_text(qa)} (b) Not mentioned in the conversation."
  return qa["question"]


def build_answer_prompt(batch: list[tuple[str, dict[str, Any]]]) -> str:
  lines = [
    ANSWER_PREFIX,
    "Return one short answer for every key.",
    'If unknown, return exactly "Not mentioned in the conversation".',
    "Questions:",
  ]
  for key, qa in batch:
    lines.append(f"{key}: {build_question_text(qa)}")
  return "\n".join(lines)


def build_output_schema(keys: list[str]) -> type[BaseModel]:
  fields = {key: (str, Field(description=f"Short answer for {key}")) for key in keys}
  return create_model("LoCoMoBatchAnswers", **fields)


def extract_prediction_map(output: Any, keys: list[str]) -> dict[str, str]:
  if getattr(output, "parsed", None) is not None:
    parsed = output.parsed.model_dump()
    return {key: str(parsed[key]).strip() for key in keys}

  raw = str(getattr(output, "content", "") or "").strip()
  if raw.startswith("```"):
    raw = raw.strip("`")
    if raw.startswith("json"):
      raw = raw[4:].strip()
  data = json.loads(raw)
  return {key: str(data[key]).strip() for key in keys}


def normalize_cat5_prediction(raw_prediction: str, qa: dict[str, Any]) -> str:
  lowered = raw_prediction.strip().lower()
  if lowered in {"a", "(a)", "option a"} or lowered.startswith("(a)"):
    return get_answer_text(qa)
  if lowered in {"b", "(b)", "option b"} or lowered.startswith("(b)"):
    return "Not mentioned in the conversation"
  return raw_prediction.strip()


def iter_session_numbers(sample: dict[str, Any]) -> list[int]:
  numbers = []
  for key in sample["conversation"]:
    if not key.startswith("session_"):
      continue
    if key.endswith("_date_time"):
      continue
    value = key.split("_", 1)[1]
    if value.isdigit():
      numbers.append(int(value))
  return sorted(numbers)


class LoCoMoMemoryV2Evaluator:
  def __init__(
    self,
    data_file: Path,
    model_id: str,
    question_batch_size: int,
    sessions_per_ingest_chunk: int,
    seed: int,
  ) -> None:
    self.data_file = data_file
    self.model_id = model_id
    self.question_batch_size = question_batch_size
    self.sessions_per_ingest_chunk = sessions_per_ingest_chunk
    self.seed = seed

  def _make_agent(self, memory: Memory) -> Agent:
    return Agent(
      model=OpenAIChat(
        id=self.model_id,
        temperature=0,
        seed=self.seed,
        max_completion_tokens=800,
      ),
      memory=memory,
      instructions=AGENT_INSTRUCTIONS,
    )

  async def _ingest_sample(self, agent: Agent, sample: dict[str, Any], user_id: str) -> None:
    session_numbers = iter_session_numbers(sample)
    for chunk_start in range(0, len(session_numbers), self.sessions_per_ingest_chunk):
      chunk = session_numbers[chunk_start : chunk_start + self.sessions_per_ingest_chunk]
      prompt = build_ingest_prompt(sample, chunk)
      chunk_label = f"{chunk[0]}-{chunk[-1]}"
      output = await agent.arun(prompt, user_id=user_id, session_id=f"{sample['sample_id']}:ingest:{chunk_label}")
      content = str(output.content or "").strip()
      if content != "ACK":
        raise RuntimeError(f"Ingestion acknowledgement mismatch for {sample['sample_id']} chunk {chunk_label}: {content!r}")

  async def _answer_batch(
    self,
    agent: Agent,
    sample_id: str,
    user_id: str,
    batch: list[tuple[str, dict[str, Any]]],
    batch_index: int,
  ) -> tuple[dict[str, str], Optional[Metrics]]:
    keys = [key for key, _ in batch]
    prompt = build_answer_prompt(batch)
    schema = build_output_schema(keys)

    try:
      output = await agent.arun(
        prompt,
        user_id=user_id,
        session_id=f"{sample_id}:qa:{batch_index}",
        output_schema=schema,
      )
      predictions = extract_prediction_map(output, keys)
      return predictions, output.metrics
    except Exception:
      if len(batch) == 1:
        raise
      midpoint = len(batch) // 2
      left_predictions, left_metrics = await self._answer_batch(agent, sample_id, user_id, batch[:midpoint], batch_index * 2)
      right_predictions, right_metrics = await self._answer_batch(
        agent,
        sample_id,
        user_id,
        batch[midpoint:],
        batch_index * 2 + 1,
      )
      merged = {**left_predictions, **right_predictions}
      merged_metrics = None
      if left_metrics and right_metrics:
        merged_metrics = left_metrics + right_metrics
      elif left_metrics:
        merged_metrics = left_metrics
      elif right_metrics:
        merged_metrics = right_metrics
      return merged, merged_metrics

  async def evaluate_sample(self, sample: dict[str, Any], max_questions: Optional[int] = None) -> SampleResult:
    start = time.monotonic()
    user_id = f"locomo::{sample['sample_id']}"

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as handle:
      db_path = handle.name

    store = SQLiteStore(db_path)
    memory = Memory(store=store)
    agent = self._make_agent(memory)

    metrics_total = Metrics()

    try:
      await self._ingest_sample(agent, sample, user_id)

      qas = list(sample["qa"])
      if max_questions is not None:
        qas = qas[:max_questions]

      predictions_by_index: dict[int, str] = {}
      for batch_start in range(0, len(qas), self.question_batch_size):
        batch_qas = qas[batch_start : batch_start + self.question_batch_size]
        keyed_batch = [(f"q{batch_start + offset}", qa) for offset, qa in enumerate(batch_qas)]
        answers, batch_metrics = await self._answer_batch(
          agent=agent,
          sample_id=sample["sample_id"],
          user_id=user_id,
          batch=keyed_batch,
          batch_index=(batch_start // self.question_batch_size) + 1,
        )
        if batch_metrics:
          metrics_total += batch_metrics

        for key, qa in keyed_batch:
          prediction = answers[key]
          if qa["category"] == 5:
            prediction = normalize_cat5_prediction(prediction, qa)
          predictions_by_index[int(key[1:])] = prediction

      category_scores: dict[int, list[float]] = defaultdict(list)
      for index, qa in enumerate(qas):
        prediction = predictions_by_index[index]
        category_scores[qa["category"]].append(score_prediction(qa, prediction))

      stats: MemoryStats = await memory.get_stats(user_id)
      average_scores = {category: round(sum(values) / len(values), 3) for category, values in sorted(category_scores.items())}
      overall = round(sum(sum(values) for values in category_scores.values()) / len(qas), 3) if qas else 0.0

      return SampleResult(
        sample_id=sample["sample_id"],
        sessions_ingested=len(iter_session_numbers(sample)),
        questions_answered=len(qas),
        overall_score=overall,
        category_scores=average_scores,
        wm_chars=stats.wm_chars,
        wm_version=stats.wm_version,
        warm_chars=stats.warm_chars,
        archive_entries=stats.entry_count,
        archive_chars=stats.total_content_chars,
        total_tokens=metrics_total.total_tokens,
        input_tokens=metrics_total.input_tokens,
        output_tokens=metrics_total.output_tokens,
        duration_s=round(time.monotonic() - start, 2),
      )
    finally:
      await agent._ashutdown()
      for ext in ("", "-shm", "-wal"):
        path = Path(db_path + ext)
        if path.exists():
          path.unlink()

  async def run(self, sample_ids: list[str], max_samples: Optional[int], max_questions: Optional[int]) -> list[SampleResult]:
    samples = json.loads(self.data_file.read_text())
    if sample_ids:
      wanted = set(sample_ids)
      samples = [sample for sample in samples if sample["sample_id"] in wanted]
    if max_samples is not None:
      samples = samples[:max_samples]

    if not samples:
      raise ValueError("No LoCoMo samples selected.")

    results: list[SampleResult] = []
    for index, sample in enumerate(samples, start=1):
      print(f"[{index}/{len(samples)}] Evaluating {sample['sample_id']} ...", flush=True)
      result = await self.evaluate_sample(sample, max_questions=max_questions)
      results.append(result)
      print(
        f"  score={result.overall_score:.3f} "
        f"questions={result.questions_answered} "
        f"wm={result.wm_chars} "
        f"archive={result.archive_entries} "
        f"tokens={result.total_tokens}",
        flush=True,
      )
    return results


def summarize_results(results: list[SampleResult]) -> dict[str, Any]:
  total_questions = sum(result.questions_answered for result in results)
  weighted_overall = (
    round(sum(result.overall_score * result.questions_answered for result in results) / total_questions, 3) if total_questions else 0.0
  )

  category_totals: dict[int, list[float]] = defaultdict(list)
  for result in results:
    for category, score in result.category_scores.items():
      category_totals[int(category)].append(score)

  return {
    "overall_score": weighted_overall,
    "total_questions": total_questions,
    "samples": [asdict(result) for result in results],
    "category_scores": {category: round(sum(scores) / len(scores), 3) for category, scores in sorted(category_totals.items())},
    "totals": {
      "wm_chars": sum(result.wm_chars for result in results),
      "warm_chars": sum(result.warm_chars for result in results),
      "archive_entries": sum(result.archive_entries for result in results),
      "archive_chars": sum(result.archive_chars for result in results),
      "input_tokens": sum(result.input_tokens for result in results),
      "output_tokens": sum(result.output_tokens for result in results),
      "total_tokens": sum(result.total_tokens for result in results),
      "duration_s": round(sum(result.duration_s for result in results), 2),
    },
  }


async def main_async() -> None:
  args = parse_args()
  evaluator = LoCoMoMemoryV2Evaluator(
    data_file=args.data_file,
    model_id=args.model,
    question_batch_size=args.question_batch_size,
    sessions_per_ingest_chunk=args.sessions_per_ingest_chunk,
    seed=args.seed,
  )
  results = await evaluator.run(
    sample_ids=args.sample_id,
    max_samples=args.max_samples,
    max_questions=args.max_questions,
  )
  summary = summarize_results(results)

  print("\nLoCoMo Memory v2 Summary")
  print(f"Model: {args.model}")
  print(f"Samples: {len(results)}")
  print(f"Questions: {summary['total_questions']}")
  print(f"Overall score: {summary['overall_score']:.3f}")
  print(f"Category scores: {summary['category_scores']}")
  print(f"Totals: {summary['totals']}")

  if args.output_json:
    args.output_json.write_text(json.dumps(summary, indent=2))
    print(f"Saved results to {args.output_json}")


def main() -> None:
  asyncio.run(main_async())


if __name__ == "__main__":
  main()
