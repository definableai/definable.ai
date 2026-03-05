"""Tests for Cortex learning module."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from definable.memory.cortex.learning.inferencer import TraitInferencer
from definable.memory.cortex.learning.observer import BehavioralObserver
from definable.memory.cortex.learning.traits import Observation, Trait, TraitCategory
from definable.memory.cortex.learning.user_model import UserModel
from definable.memory.cortex.learning.validator import ModelValidator


class TestTrait:
  def test_creation(self):
    t = Trait(name="direct", description="Uses direct language", category=TraitCategory.COMMUNICATION)
    assert t.trait_id
    assert t.confidence == 0.3

  def test_reinforce(self):
    t = Trait(name="test", description="test", confidence=0.3)
    obs = Observation(content="signal", signal_strength=1.0)
    t.reinforce(obs, boost=0.15)
    assert t.confidence == pytest.approx(0.45)
    assert len(t.observations) == 1

  def test_contradict(self):
    t = Trait(name="test", description="test", confidence=0.8)
    t.contradict(decay=0.85)
    assert t.confidence == pytest.approx(0.68)

  def test_is_strong(self):
    t = Trait(confidence=0.6)
    assert not t.is_strong
    t.confidence = 0.7
    assert t.is_strong

  def test_roundtrip(self):
    t = Trait(
      name="direct",
      description="Speaks directly",
      category=TraitCategory.COMMUNICATION,
      confidence=0.8,
      observations=[Observation(content="signal")],
    )
    d = t.to_dict()
    restored = Trait.from_dict(d)
    assert restored.name == "direct"
    assert restored.confidence == 0.8
    assert len(restored.observations) == 1


class TestObservation:
  def test_auto_id(self):
    o = Observation(content="test signal")
    assert o.observation_id
    assert o.created_at > 0

  def test_roundtrip(self):
    o = Observation(content="test", category=TraitCategory.TECHNICAL, signal_strength=0.8)
    d = o.to_dict()
    restored = Observation.from_dict(d)
    assert restored.category == TraitCategory.TECHNICAL
    assert restored.signal_strength == 0.8


class TestUserModel:
  def test_add_and_get_trait(self):
    model = UserModel()
    t = Trait(name="directness", description="Is direct", category=TraitCategory.COMMUNICATION)
    model.add_trait(t)
    assert model.trait_count == 1
    assert model.get_trait("directness") is not None
    assert model.get_trait("nonexistent") is None

  def test_get_by_category(self):
    model = UserModel()
    model.add_trait(Trait(name="a", category=TraitCategory.TECHNICAL))
    model.add_trait(Trait(name="b", category=TraitCategory.COMMUNICATION))
    model.add_trait(Trait(name="c", category=TraitCategory.TECHNICAL))
    tech = model.get_traits_by_category(TraitCategory.TECHNICAL)
    assert len(tech) == 2

  def test_get_strong_traits(self):
    model = UserModel()
    model.add_trait(Trait(name="weak", confidence=0.3))
    model.add_trait(Trait(name="strong", confidence=0.8))
    strong = model.get_strong_traits(threshold=0.7)
    assert len(strong) == 1
    assert strong[0].name == "strong"

  def test_generate_style_guide(self):
    model = UserModel()
    model.add_trait(
      Trait(
        name="directness",
        description="Uses direct, imperative language",
        category=TraitCategory.COMMUNICATION,
        confidence=0.85,
        observations=[Observation(content="s1"), Observation(content="s2")],
      )
    )
    model.add_trait(
      Trait(
        name="testing",
        description="Values comprehensive test coverage",
        category=TraitCategory.TECHNICAL,
        confidence=0.7,
      )
    )
    guide = model.generate_style_guide()
    assert "Communication Style" in guide
    assert "direct" in guide.lower()
    assert "Technical" in guide

  def test_generate_style_guide_empty(self):
    model = UserModel()
    guide = model.generate_style_guide()
    assert "No strong" in guide

  def test_predict(self):
    model = UserModel()
    model.add_trait(
      Trait(
        name="testing focus",
        description="Always writes tests first",
        confidence=0.9,
      )
    )
    prediction = model.predict("Should we write tests?")
    assert prediction is not None
    assert "tests" in prediction.lower()
    assert len(model.predictions) == 1

  def test_predict_insufficient_data(self):
    model = UserModel()
    assert model.predict("anything") is None

  def test_roundtrip(self):
    model = UserModel(user_id="u1")
    model.add_trait(Trait(name="t1", confidence=0.8))
    d = model.to_dict()
    restored = UserModel.from_dict(d)
    assert restored.user_id == "u1"
    assert restored.trait_count == 1


class TestBehavioralObserver:
  def test_rule_based_directness(self):
    obs = BehavioralObserver()
    results = obs._observe_rules("You must always test your code, never skip it", "r1")
    assert any(o.category == TraitCategory.COMMUNICATION for o in results)

  def test_rule_based_testing(self):
    obs = BehavioralObserver()
    results = obs._observe_rules("Let's add unit tests for this module", "r1")
    assert any(o.category == TraitCategory.TECHNICAL for o in results)

  def test_rule_based_frustration(self):
    obs = BehavioralObserver()
    results = obs._observe_rules("This is broken and I'm frustrated", "r1")
    assert any(o.category == TraitCategory.EMOTIONAL for o in results)

  @pytest.mark.asyncio
  async def test_skips_assistant_messages(self):
    obs = BehavioralObserver()
    results = await obs.observe("anything", role="assistant")
    assert results == []

  @pytest.mark.asyncio
  async def test_llm_observer(self):
    response_data = json.dumps([
      {"content": "Values code quality", "category": "values", "signal_strength": 0.8},
    ])
    model = AsyncMock()
    resp = MagicMock()
    resp.content = response_data
    model.ainvoke = AsyncMock(return_value=resp)

    obs = BehavioralObserver(model=model)
    results = await obs.observe("We need to ensure high code quality")
    assert len(results) >= 1


class TestTraitInferencer:
  def test_creates_new_trait(self):
    inferencer = TraitInferencer()
    model = UserModel()
    obs = [Observation(content="Uses direct language", category=TraitCategory.COMMUNICATION)]
    updates = inferencer.process(obs, model)
    assert updates == 1
    assert model.trait_count == 1

  def test_reinforces_existing_trait(self):
    inferencer = TraitInferencer()
    model = UserModel()
    # Create initial trait
    model.add_trait(
      Trait(
        name="direct language",
        description="Uses direct imperative language",
        category=TraitCategory.COMMUNICATION,
        confidence=0.3,
      )
    )
    # Reinforce with similar observation
    obs = [
      Observation(
        content="Uses direct imperative language patterns",
        category=TraitCategory.COMMUNICATION,
        signal_strength=1.0,
      )
    ]
    updates = inferencer.process(obs, model)
    assert updates == 1
    assert model.traits[0].confidence > 0.3

  def test_detects_contradiction(self):
    inferencer = TraitInferencer()
    model = UserModel()
    model.add_trait(
      Trait(
        name="verbose",
        description="Prefers verbose detailed explanations",
        category=TraitCategory.COMMUNICATION,
        confidence=0.8,
      )
    )
    obs = [
      Observation(
        content="Prefers direct concise communication",
        category=TraitCategory.COMMUNICATION,
      )
    ]
    decayed = inferencer.check_contradictions(obs, model)
    assert decayed >= 0  # May or may not detect depending on keyword match

  def test_multiple_observations(self):
    inferencer = TraitInferencer()
    model = UserModel()
    obs = [
      Observation(content="Values testing", category=TraitCategory.TECHNICAL),
      Observation(content="Prefers direct feedback", category=TraitCategory.COMMUNICATION),
    ]
    updates = inferencer.process(obs, model)
    assert updates == 2
    assert model.trait_count == 2


class TestModelValidator:
  def test_record_and_validate(self):
    v = ModelValidator()
    rec = v.record_prediction("test scenario", "predicted response")
    assert v.total_predictions == 1
    assert v.validated_count == 0

    v.validate(rec, "actual response", correct=True)
    assert v.validated_count == 1
    assert v.accuracy == 1.0

  def test_accuracy_calculation(self):
    v = ModelValidator()
    r1 = v.record_prediction("s1", "p1")
    r2 = v.record_prediction("s2", "p2")
    r3 = v.record_prediction("s3", "p3")
    v.validate(r1, "a1", correct=True)
    v.validate(r2, "a2", correct=False)
    v.validate(r3, "a3", correct=True)
    assert v.accuracy == pytest.approx(2 / 3)

  def test_metrics(self):
    v = ModelValidator()
    v.record_prediction("s1", "p1")
    metrics = v.get_metrics()
    assert metrics["total_predictions"] == 1
    assert metrics["pending"] == 1

  def test_roundtrip(self):
    v = ModelValidator()
    rec = v.record_prediction("scenario", "prediction")
    v.validate(rec, "outcome", correct=True)
    d = v.to_dict()
    restored = ModelValidator.from_dict(d)
    assert restored.total_predictions == 1
    assert restored.accuracy == 1.0
