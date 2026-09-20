"""
Syntera Evaluation Framework — Contracts and Runner

Provides typed evaluation primitives: Dataset, Task, Prediction, Metric, Result.
Supports measuring AI capability rather than just software correctness.
"""
import time
import json
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from pydantic import BaseModel, Field
from datetime import datetime


class EvalSample(BaseModel):
    """A single evaluation sample with input, reference, and metadata."""
    id: str = ""
    input: str
    reference: str = ""  # ground truth / expected output
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvalDataset(BaseModel):
    """A collection of evaluation samples."""
    name: str
    samples: List[EvalSample] = Field(default_factory=list)
    version: str = "1.0"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvalPrediction(BaseModel):
    """A system's prediction for one sample."""
    sample_id: str
    output: str
    latency_ms: float = 0.0
    token_usage: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


class MetricResult(BaseModel):
    """Result of computing a single metric."""
    name: str
    value: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvalRunResult(BaseModel):
    """Complete result of an evaluation run."""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    dataset_name: str
    system_name: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    metrics: List[MetricResult] = Field(default_factory=list)
    predictions: List[EvalPrediction] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    total_samples: int = 0
    successful_samples: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseMetric(ABC):
    """Abstract metric for AI evaluation."""
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def compute(self, predictions: List[EvalPrediction], samples: List[EvalSample]) -> MetricResult:
        pass


# ── Built-in metrics ──────────────────────────────────────

class SuccessRateMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "success_rate"

    def compute(self, predictions: List[EvalPrediction], samples: List[EvalSample]) -> MetricResult:
        total = len(predictions)
        successes = sum(1 for p in predictions if p.success)
        rate = successes / total if total > 0 else 0.0
        return MetricResult(name=self.name, value=rate, metadata={"total": total, "successes": successes})


class MeanLatencyMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "mean_latency_ms"

    def compute(self, predictions: List[EvalPrediction], samples: List[EvalSample]) -> MetricResult:
        latencies = [p.latency_ms for p in predictions if p.success]
        mean = sum(latencies) / len(latencies) if latencies else 0.0
        return MetricResult(name=self.name, value=round(mean, 2))


class ExactMatchMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "exact_match"

    def compute(self, predictions: List[EvalPrediction], samples: List[EvalSample]) -> MetricResult:
        sample_map = {s.id: s for s in samples}
        matches = 0
        total = 0
        for p in predictions:
            if p.sample_id in sample_map and p.success:
                total += 1
                if p.output.strip() == sample_map[p.sample_id].reference.strip():
                    matches += 1
        rate = matches / total if total > 0 else 0.0
        return MetricResult(name=self.name, value=rate, metadata={"matches": matches, "total": total})


class StructuredOutputValidityMetric(BaseMetric):
    """Checks if predictions contain valid JSON."""
    @property
    def name(self) -> str:
        return "structured_output_validity"

    def compute(self, predictions: List[EvalPrediction], samples: List[EvalSample]) -> MetricResult:
        valid = 0
        total = len(predictions)
        for p in predictions:
            try:
                json.loads(p.output)
                valid += 1
            except (json.JSONDecodeError, TypeError):
                pass
        rate = valid / total if total > 0 else 0.0
        return MetricResult(name=self.name, value=rate, metadata={"valid": valid, "total": total})


class ContainsReferenceMetric(BaseMetric):
    """Fuzzy check: does the prediction contain the reference answer (case-insensitive)?"""
    @property
    def name(self) -> str:
        return "contains_reference"

    def compute(self, predictions: List[EvalPrediction], samples: List[EvalSample]) -> MetricResult:
        sample_map = {s.id: s for s in samples}
        matches = 0
        total = 0
        for p in predictions:
            if p.sample_id in sample_map and p.success and sample_map[p.sample_id].reference:
                total += 1
                if sample_map[p.sample_id].reference.lower() in p.output.lower():
                    matches += 1
        rate = matches / total if total > 0 else 0.0
        return MetricResult(name=self.name, value=rate, metadata={"matches": matches, "total": total})


# ── Evaluation Runner ─────────────────────────────────────

class EvaluationRunner:
    """
    Executes an evaluation run: feeds samples through a system function,
    collects predictions, and computes metrics.
    """

    def __init__(self, metrics: Optional[List[BaseMetric]] = None):
        self.metrics = metrics or [SuccessRateMetric(), MeanLatencyMetric()]

    def run(
        self,
        dataset: EvalDataset,
        system_fn: Callable[[str], str],
        system_name: str = "syntera",
    ) -> EvalRunResult:
        """
        Run evaluation.

        Args:
            dataset: The evaluation dataset.
            system_fn: A callable that takes an input string and returns an output string.
            system_name: Name of the system being evaluated.

        Returns:
            EvalRunResult with predictions and computed metrics.
        """
        predictions = []
        total_start = time.time()

        for sample in dataset.samples:
            start = time.time()
            try:
                output = system_fn(sample.input)
                latency = (time.time() - start) * 1000
                predictions.append(EvalPrediction(
                    sample_id=sample.id,
                    output=output,
                    latency_ms=latency,
                    success=True,
                ))
            except Exception as e:
                latency = (time.time() - start) * 1000
                predictions.append(EvalPrediction(
                    sample_id=sample.id,
                    output="",
                    latency_ms=latency,
                    success=False,
                    error=str(e),
                ))

        total_latency = (time.time() - total_start) * 1000

        # Compute metrics
        metric_results = []
        for metric in self.metrics:
            try:
                result = metric.compute(predictions, dataset.samples)
                metric_results.append(result)
            except Exception as e:
                metric_results.append(MetricResult(
                    name=metric.name,
                    value=-1.0,
                    metadata={"error": str(e)},
                ))

        return EvalRunResult(
            dataset_name=dataset.name,
            system_name=system_name,
            metrics=metric_results,
            predictions=predictions,
            total_latency_ms=round(total_latency, 2),
            total_samples=len(dataset.samples),
            successful_samples=sum(1 for p in predictions if p.success),
        )
