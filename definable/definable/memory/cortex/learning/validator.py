"""Model validator — self-testing prediction loop."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PredictionRecord:
  """Record of a prediction and its outcome."""

  scenario: str = ""
  prediction: str = ""
  actual_outcome: Optional[str] = None
  correct: Optional[bool] = None
  timestamp: float = 0.0

  def __post_init__(self) -> None:
    if self.timestamp == 0.0:
      self.timestamp = time.time()


@dataclass
class ModelValidator:
  """Tracks predictions vs outcomes for model accuracy scoring.

  The validator maintains a history of predictions and their outcomes,
  computing accuracy metrics over time.
  """

  records: List[PredictionRecord] = field(default_factory=list)

  def record_prediction(self, scenario: str, prediction: str) -> PredictionRecord:
    """Record a prediction for later validation."""
    rec = PredictionRecord(scenario=scenario, prediction=prediction)
    self.records.append(rec)
    return rec

  def validate(self, record: PredictionRecord, actual_outcome: str, correct: bool) -> None:
    """Validate a prediction against the actual outcome."""
    record.actual_outcome = actual_outcome
    record.correct = correct

  @property
  def total_predictions(self) -> int:
    return len(self.records)

  @property
  def validated_count(self) -> int:
    return sum(1 for r in self.records if r.correct is not None)

  @property
  def accuracy(self) -> float:
    """Accuracy as fraction of validated predictions that were correct."""
    validated = [r for r in self.records if r.correct is not None]
    if not validated:
      return 0.0
    return sum(1 for r in validated if r.correct) / len(validated)

  def get_metrics(self) -> Dict[str, Any]:
    """Get validation metrics."""
    return {
      "total_predictions": self.total_predictions,
      "validated": self.validated_count,
      "accuracy": self.accuracy,
      "pending": self.total_predictions - self.validated_count,
    }

  def to_dict(self) -> Dict[str, Any]:
    return {
      "records": [
        {
          "scenario": r.scenario,
          "prediction": r.prediction,
          "actual_outcome": r.actual_outcome,
          "correct": r.correct,
          "timestamp": r.timestamp,
        }
        for r in self.records
      ],
    }

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "ModelValidator":
    records = []
    for rd in data.get("records", []):
      records.append(
        PredictionRecord(
          scenario=rd.get("scenario", ""),
          prediction=rd.get("prediction", ""),
          actual_outcome=rd.get("actual_outcome"),
          correct=rd.get("correct"),
          timestamp=rd.get("timestamp", 0.0),
        )
      )
    return cls(records=records)
