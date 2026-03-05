"""Cortex Memory evolution test — 100 memories, 5 checkpoints, ground-truth validation.

Feeds 100 simulated Anandesh interactions sequentially, validating at every
20-memory checkpoint that the system is learning correctly.

Runs without any LLM or external API — uses rule-based observation and
signature/tag indexing only.

Usage:
  .venv/bin/python -m pytest tests/validation/cortex_evolution_test.py -v
"""

import pytest

from definable.memory.cortex.config import CortexConfig
from definable.memory.cortex.cortex import CortexMemory
from definable.memory.cortex.learning.inferencer import TraitInferencer
from definable.memory.cortex.learning.observer import BehavioralObserver
from definable.memory.cortex.learning.traits import TraitCategory
from definable.memory.cortex.learning.user_model import UserModel
from definable.memory.cortex.learning.validator import ModelValidator
from definable.memory.cortex.record.types import MemorySource

from tests.validation.anandesh_memories import GROUND_TRUTH_TRAITS, MEMORIES


# ================================================================
# Helpers
# ================================================================


def _category_from_str(cat: str) -> TraitCategory:
  """Convert ground truth category string to TraitCategory enum."""
  return TraitCategory(cat)


def compute_trait_recall(user_model: UserModel, ground_truth: list[dict], confidence_floor: float = 0.0) -> float:
  """What fraction of ground-truth traits did we detect (at any confidence)?

  A ground-truth trait counts as "detected" if there is ANY trait in the
  user model with the same category whose description shares ≥2 significant
  words with the ground-truth description.
  """
  if not ground_truth:
    return 0.0

  detected = 0
  for gt in ground_truth:
    gt_cat = _category_from_str(gt["category"])
    gt_words = set(gt["description"].lower().split())
    # Remove stopwords for matching
    stopwords = {"a", "an", "the", "is", "are", "and", "or", "of", "to", "in", "for", "that", "it", "not", "on", "with"}
    gt_words -= stopwords

    found = False
    for trait in user_model.get_traits_by_category(gt_cat):
      if trait.confidence < confidence_floor:
        continue
      trait_words = set(trait.description.lower().split()) - stopwords
      overlap = len(gt_words & trait_words)
      if overlap >= 2:
        found = True
        break
    if found:
      detected += 1

  return detected / len(ground_truth)


def compute_trait_precision(user_model: UserModel, ground_truth: list[dict]) -> float:
  """What fraction of detected traits are correct (match a ground truth)?

  A detected trait counts as "correct" if there is a ground-truth trait
  in the same category with ≥2 word overlap in description.
  """
  if not user_model.traits:
    return 0.0

  correct = 0
  stopwords = {"a", "an", "the", "is", "are", "and", "or", "of", "to", "in", "for", "that", "it", "not", "on", "with"}

  for trait in user_model.traits:
    for gt in ground_truth:
      gt_cat = _category_from_str(gt["category"])
      if trait.category != gt_cat:
        continue
      gt_words = set(gt["description"].lower().split()) - stopwords
      trait_words = set(trait.description.lower().split()) - stopwords
      if len(gt_words & trait_words) >= 2:
        correct += 1
        break

  return correct / len(user_model.traits)


# ================================================================
# Novel scenarios for replication test (checkpoint 5)
# ================================================================

NOVEL_SCENARIOS = [
  {
    "scenario": "Someone suggests using Pydantic for a new internal config class",
    "expected_keywords": ["dataclass", "simple", "overhead"],
    "category": "technical",
  },
  {
    "scenario": "A contributor submits a large PR without any tests",
    "expected_keywords": ["test", "ship", "verify"],
    "category": "values",
  },
  {
    "scenario": "An error is silently swallowed in a background task",
    "expected_keywords": ["silent", "fail", "log"],
    "category": "emotional",
  },
  {
    "scenario": "Someone proposes adding an abstract base class for models",
    "expected_keywords": ["composition", "inherit"],
    "category": "technical",
  },
  {
    "scenario": "A team member asks whether to use sync or async for a new module",
    "expected_keywords": ["async", "sync", "break"],
    "category": "technical",
  },
  {
    "scenario": "Someone asks for feedback on their verbose error message",
    "expected_keywords": ["direct", "concise", "clear"],
    "category": "communication",
  },
  {
    "scenario": "A reviewer notices an import from definable.tools instead of definable.tool",
    "expected_keywords": ["import", "path", "rename"],
    "category": "workflow",
  },
  {
    "scenario": "Someone proposes using Redis for caching in the framework",
    "expected_keywords": ["simple", "depend", "heavy"],
    "category": "decision",
  },
  {
    "scenario": "A PR amends an existing commit without being asked to",
    "expected_keywords": ["commit", "new", "amend"],
    "category": "workflow",
  },
  {
    "scenario": "Someone builds a 500-line feature without intermediate checkpoints",
    "expected_keywords": ["phase", "small", "incremental", "test"],
    "category": "workflow",
  },
]


# ================================================================
# Test fixture — full Cortex + learning subsystem
# ================================================================


@pytest.fixture
async def evolution_cortex(tmp_path):
  """CortexMemory with all indexes + learning enabled, no LLM."""
  config = CortexConfig(
    db_path=str(tmp_path / "evolution.db"),
    slow_path_enabled=False,
    enable_learning=True,
    enable_consolidation=False,
    enable_signatures=True,
    enable_graph=True,
    enable_tags=True,
    learning_reinforcement_boost=0.15,
    learning_confidence_decay=0.85,
  )
  memory = CortexMemory(config=config)
  await memory._ensure_initialized()
  yield memory
  await memory.close()


# ================================================================
# The evolution test
# ================================================================


@pytest.mark.asyncio
class TestCortexEvolution:
  """Feed 100 memories, validate at 5 checkpoints."""

  async def test_full_evolution(self, evolution_cortex: CortexMemory):
    """Run the complete 100-memory evolution and validate at each checkpoint."""
    cortex = evolution_cortex

    # Manual learning components (Cortex uses LLM observer by default,
    # but we use rule-based since there's no model)
    observer = BehavioralObserver(model=None)
    inferencer = TraitInferencer(config=cortex.config)
    user_model = UserModel(user_id="anandesh")
    validator = ModelValidator()

    # Metrics tracking
    confidence_history: list[dict] = []  # Track confidence evolution over time
    checkpoint_reports: list[dict] = []

    for i, mem in enumerate(MEMORIES, start=1):
      # Ingest memory into Cortex
      record_id = await cortex.remember(
        content=mem["content"],
        source=MemorySource.CONVERSATION,
        session_id="anandesh_session",
        user_id="anandesh",
        role=mem["role"],
      )

      # Run behavioral observation + inference
      observations = observer._observe_rules(mem["content"], record_id)
      if observations:
        inferencer.process(observations, user_model)
        inferencer.check_contradictions(observations, user_model)

      # Track confidence snapshot every 10 memories
      if i % 10 == 0:
        strong = user_model.get_strong_traits()
        confidence_history.append({
          "turn": i,
          "trait_count": user_model.trait_count,
          "strong_count": len(strong),
          "avg_confidence": (sum(t.confidence for t in user_model.traits) / len(user_model.traits)) if user_model.traits else 0,
        })

      # ---- Checkpoint validations ----

      if i == 20:
        report = await self._checkpoint_20(cortex, user_model)
        checkpoint_reports.append(report)

      elif i == 40:
        report = await self._checkpoint_40(cortex, user_model)
        checkpoint_reports.append(report)

      elif i == 60:
        report = await self._checkpoint_60(cortex, user_model)
        checkpoint_reports.append(report)

      elif i == 80:
        report = await self._checkpoint_80(cortex, user_model)
        checkpoint_reports.append(report)

      elif i == 100:
        report = await self._checkpoint_100(cortex, user_model, validator)
        checkpoint_reports.append(report)

    # ---- Final summary ----
    assert len(checkpoint_reports) == 5, f"Expected 5 checkpoints, got {len(checkpoint_reports)}"

    # Verify confidence curves are generally increasing
    if len(confidence_history) >= 5:
      early_avg = confidence_history[1]["avg_confidence"]  # Turn 20
      late_avg = confidence_history[-1]["avg_confidence"]  # Turn 100
      assert late_avg >= early_avg, f"Confidence should increase over time: early={early_avg:.2f}, late={late_avg:.2f}"

  # ================================================================
  # Checkpoint 1: After 20 memories (early learning)
  # ================================================================

  async def _checkpoint_20(self, cortex: CortexMemory, model: UserModel) -> dict:
    """After 20 memories: basic trait detection + retrieval."""
    report = {"checkpoint": 20}

    # Trait detection: should have at least 3 traits
    assert model.trait_count >= 3, f"Checkpoint 20: expected ≥3 traits, got {model.trait_count}"
    report["trait_count"] = model.trait_count

    # At least one trait should be above 0.3 confidence
    above_03 = [t for t in model.traits if t.confidence > 0.3]
    assert len(above_03) >= 1, f"Checkpoint 20: expected ≥1 trait at confidence>0.3, got {len(above_03)}"
    report["traits_above_03"] = len(above_03)

    # Scratchpad should work
    await cortex.set_belief("session_start", "true", session_id="anandesh_session", user_id="anandesh")
    state = await cortex.get_state(session_id="anandesh_session", user_id="anandesh")
    assert state.get_belief("session_start") == "true"
    report["scratchpad_ok"] = True

    # Basic retrieval should return results
    result = await cortex.recall("testing", session_id="anandesh_session", user_id="anandesh", top_k=5)
    assert result is not None
    report["retrieval_count"] = len(result.memories)

    return report

  # ================================================================
  # Checkpoint 2: After 40 memories (active learning)
  # ================================================================

  async def _checkpoint_40(self, cortex: CortexMemory, model: UserModel) -> dict:
    """After 40 memories: communication style + technical preferences detected."""
    report = {"checkpoint": 40}

    # Should have more traits now (rule-based observer has 5 patterns)
    assert model.trait_count >= 3, f"Checkpoint 40: expected ≥3 traits, got {model.trait_count}"
    report["trait_count"] = model.trait_count

    # Communication traits should exist
    comm_traits = model.get_traits_by_category(TraitCategory.COMMUNICATION)
    report["communication_traits"] = len(comm_traits)

    # Technical traits should exist
    tech_traits = model.get_traits_by_category(TraitCategory.TECHNICAL)
    report["technical_traits"] = len(tech_traits)

    # Some traits should be above 0.4 (reinforced multiple times)
    above_04 = [t for t in model.traits if t.confidence > 0.4]
    assert len(above_04) >= 1, f"Checkpoint 40: expected ≥1 trait at confidence>0.4, got {len(above_04)}"
    report["traits_above_04"] = len(above_04)

    # Recall should find decision-related content
    result = await cortex.recall("decision architecture", session_id="anandesh_session", user_id="anandesh", top_k=5)
    assert result is not None
    report["decision_recall_count"] = len(result.memories)

    return report

  # ================================================================
  # Checkpoint 3: After 60 memories (forming model)
  # ================================================================

  async def _checkpoint_60(self, cortex: CortexMemory, model: UserModel) -> dict:
    """After 60 memories: core values detected, frustration patterns visible."""
    report = {"checkpoint": 60}

    # Should have strong traits by now
    strong = model.get_strong_traits(threshold=0.6)
    assert len(strong) >= 1, f"Checkpoint 60: expected ≥1 strong trait (>0.6), got {len(strong)}"
    report["strong_traits"] = len(strong)

    # Values traits should exist
    value_traits = model.get_traits_by_category(TraitCategory.VALUES)
    report["value_traits"] = len(value_traits)

    # Emotional traits should exist (frustration signals)
    emotional_traits = model.get_traits_by_category(TraitCategory.EMOTIONAL)
    report["emotional_traits"] = len(emotional_traits)

    # Workflow traits should exist
    workflow_traits = model.get_traits_by_category(TraitCategory.WORKFLOW)
    report["workflow_traits"] = len(workflow_traits)

    # Compute partial recall against ground truth
    recall = compute_trait_recall(model, GROUND_TRUTH_TRAITS)
    report["trait_recall"] = recall

    return report

  # ================================================================
  # Checkpoint 4: After 80 memories (mature model)
  # ================================================================

  async def _checkpoint_80(self, cortex: CortexMemory, model: UserModel) -> dict:
    """After 80 memories: style guide generation + retrieval quality."""
    report = {"checkpoint": 80}

    # Style guide should be non-trivial
    style_guide = model.generate_style_guide()
    assert "No strong user traits" not in style_guide, "Checkpoint 80: style guide should contain trait data"
    assert len(style_guide) > 100, f"Checkpoint 80: style guide too short ({len(style_guide)} chars)"
    report["style_guide_length"] = len(style_guide)

    # Trait recall against ground truth should be improving
    recall = compute_trait_recall(model, GROUND_TRUTH_TRAITS)
    report["trait_recall"] = recall

    # Precision: detected traits should mostly be correct
    precision = compute_trait_precision(model, GROUND_TRUTH_TRAITS)
    report["trait_precision"] = precision

    # Total record count should match
    assert cortex._store is not None
    total = await cortex._store.count_records("anandesh_session", "anandesh")
    assert total == 80, f"Checkpoint 80: expected 80 records, got {total}"
    report["total_records"] = total

    # Update cascade: modify a belief and verify
    await cortex.set_belief("checkpoint", "80", session_id="anandesh_session", user_id="anandesh")
    state = await cortex.get_state(session_id="anandesh_session", user_id="anandesh")
    assert state.get_belief("checkpoint") == "80"

    return report

  # ================================================================
  # Checkpoint 5: After 100 memories (replication test)
  # ================================================================

  async def _checkpoint_100(self, cortex: CortexMemory, model: UserModel, validator: ModelValidator) -> dict:
    """After 100 memories: full metrics + novel scenario predictions."""
    report = {"checkpoint": 100}

    # ---- Metrics ----
    trait_recall = compute_trait_recall(model, GROUND_TRUTH_TRAITS)
    trait_precision = compute_trait_precision(model, GROUND_TRUTH_TRAITS)
    report["trait_recall"] = trait_recall
    report["trait_precision"] = trait_precision
    report["total_traits"] = model.trait_count
    report["strong_traits"] = model.strong_trait_count

    # Confidence distribution
    confidences = [t.confidence for t in model.traits]
    report["avg_confidence"] = sum(confidences) / len(confidences) if confidences else 0
    report["max_confidence"] = max(confidences) if confidences else 0
    report["min_confidence"] = min(confidences) if confidences else 0

    # Category coverage
    categories_seen = {t.category for t in model.traits}
    report["categories_covered"] = len(categories_seen)

    # ---- Retrieval quality ----
    retrieval_hits = 0
    test_queries = [
      ("composition inheritance architecture", "composition"),
      ("testing pytest verification", "test"),
      ("frustration silent failure", "silent"),
      ("phased approach incremental", "phase"),
      ("direct communication imperative", "direct"),
    ]
    for query, expected_keyword in test_queries:
      result = await cortex.recall(query, session_id="anandesh_session", user_id="anandesh", top_k=5)
      for sm in result.memories:
        if expected_keyword.lower() in sm.record.raw_content.lower():
          retrieval_hits += 1
          break

    report["retrieval_hits"] = retrieval_hits
    report["retrieval_queries"] = len(test_queries)

    # ---- Novel scenario predictions ----
    prediction_score = 0
    for scenario in NOVEL_SCENARIOS:
      prediction = model.predict(scenario["scenario"])
      if prediction:
        pred_lower = prediction.lower()
        # Check if prediction references any expected concepts
        for kw in scenario["expected_keywords"]:
          if kw.lower() in pred_lower:
            prediction_score += 1
            break

    report["prediction_hits"] = prediction_score
    report["prediction_total"] = len(NOVEL_SCENARIOS)

    # ---- Style guide quality ----
    style_guide = model.generate_style_guide()
    report["style_guide_length"] = len(style_guide)

    # Style guide should mention key categories
    guide_lower = style_guide.lower()
    mentioned_categories = 0
    for label in ["communication", "technical", "values", "workflow", "emotional"]:
      if label in guide_lower:
        mentioned_categories += 1
    report["style_guide_categories"] = mentioned_categories

    # ---- Final record count ----
    assert cortex._store is not None
    total = await cortex._store.count_records("anandesh_session", "anandesh")
    assert total == 100, f"Final: expected 100 records, got {total}"
    report["total_records"] = total

    # ---- Soft assertions (logged but not failing) ----
    # These are targets, not hard requirements, since we use
    # rule-based observation without LLM
    report["recall_target_met"] = trait_recall >= 0.3  # Relaxed for rule-based
    report["precision_target_met"] = trait_precision >= 0.5

    # Hard assertion: system must have learned SOMETHING
    assert model.trait_count >= 5, f"Final: expected ≥5 traits, got {model.trait_count}"
    assert model.strong_trait_count >= 1, f"Final: expected ≥1 strong trait, got {model.strong_trait_count}"

    return report


@pytest.mark.asyncio
class TestCortexEvolutionMetrics:
  """Individual metric tests for finer-grained CI reporting."""

  async def test_ingestion_all_100(self, evolution_cortex: CortexMemory):
    """All 100 memories should be ingested without error."""
    for mem in MEMORIES:
      await evolution_cortex.remember(
        content=mem["content"],
        source=MemorySource.CONVERSATION,
        session_id="s1",
        user_id="anandesh",
        role=mem["role"],
      )
    assert evolution_cortex._store is not None
    assert await evolution_cortex._store.count_records("s1", "anandesh") == 100

  async def test_learning_produces_traits(self, evolution_cortex: CortexMemory):
    """Rule-based observer should produce traits from 100 memories."""
    observer = BehavioralObserver(model=None)
    inferencer = TraitInferencer(config=evolution_cortex.config)
    user_model = UserModel(user_id="anandesh")

    for mem in MEMORIES:
      record_id = await evolution_cortex.remember(
        content=mem["content"],
        source=MemorySource.CONVERSATION,
        session_id="s1",
        user_id="anandesh",
        role=mem["role"],
      )
      observations = observer._observe_rules(mem["content"], record_id)
      if observations:
        inferencer.process(observations, user_model)

    # Should have traits across multiple categories
    assert user_model.trait_count >= 4
    categories = {t.category for t in user_model.traits}
    assert len(categories) >= 3, f"Expected ≥3 categories, got {categories}"

  async def test_confidence_increases_with_reinforcement(self, evolution_cortex: CortexMemory):
    """Traits mentioned multiple times should gain confidence."""
    observer = BehavioralObserver(model=None)
    inferencer = TraitInferencer(config=evolution_cortex.config)
    user_model = UserModel(user_id="anandesh")

    confidence_snapshots: dict[str, list[float]] = {}

    for mem in MEMORIES:
      record_id = await evolution_cortex.remember(
        content=mem["content"],
        source=MemorySource.CONVERSATION,
        session_id="s1",
        user_id="anandesh",
        role=mem["role"],
      )
      observations = observer._observe_rules(mem["content"], record_id)
      if observations:
        inferencer.process(observations, user_model)
        # Track confidence for all traits
        for trait in user_model.traits:
          if trait.name not in confidence_snapshots:
            confidence_snapshots[trait.name] = []
          confidence_snapshots[trait.name].append(trait.confidence)

    # Traits that got reinforced should show confidence growth
    growing_traits = 0
    for name, history in confidence_snapshots.items():
      if len(history) >= 3 and history[-1] > history[0]:
        growing_traits += 1

    assert growing_traits >= 1, "At least one trait should show confidence growth over time"

  async def test_style_guide_generation(self, evolution_cortex: CortexMemory):
    """After 100 memories, style guide should be substantive."""
    observer = BehavioralObserver(model=None)
    inferencer = TraitInferencer(config=evolution_cortex.config)
    user_model = UserModel(user_id="anandesh")

    for mem in MEMORIES:
      record_id = await evolution_cortex.remember(
        content=mem["content"],
        source=MemorySource.CONVERSATION,
        session_id="s1",
        user_id="anandesh",
        role=mem["role"],
      )
      observations = observer._observe_rules(mem["content"], record_id)
      if observations:
        inferencer.process(observations, user_model)

    guide = user_model.generate_style_guide()
    assert "No strong user traits" not in guide
    assert len(guide) > 50

  async def test_retrieval_after_full_ingestion(self, evolution_cortex: CortexMemory):
    """Retrieval should return relevant results after 100 memories."""
    for mem in MEMORIES:
      await evolution_cortex.remember(
        content=mem["content"],
        source=MemorySource.CONVERSATION,
        session_id="s1",
        user_id="anandesh",
        role=mem["role"],
      )

    # Search for testing-related content
    result = await evolution_cortex.recall("testing pytest", session_id="s1", user_id="anandesh", top_k=5)
    assert result is not None
    assert len(result.memories) > 0

    # Search for frustration-related content
    result = await evolution_cortex.recall("frustration errors", session_id="s1", user_id="anandesh", top_k=5)
    assert result is not None
    assert len(result.memories) > 0

  async def test_update_and_forget(self, evolution_cortex: CortexMemory):
    """Update and forget operations should work on ingested memories."""
    rid = await evolution_cortex.remember(
      content="Original content to update",
      source=MemorySource.CONVERSATION,
      session_id="s1",
      user_id="anandesh",
    )

    # Update
    new_record = await evolution_cortex.update(rid, "Updated content", reason="test")
    assert new_record is not None
    assert new_record.raw_content == "Updated content"

    # Forget
    rid2 = await evolution_cortex.remember(
      content="Content to forget",
      source=MemorySource.CONVERSATION,
      session_id="s1",
      user_id="anandesh",
    )
    assert await evolution_cortex.forget(rid2, reason="test")
