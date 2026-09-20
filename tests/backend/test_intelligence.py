"""
Test suite for Syntera Intelligence Core

Tests with fake providers — no real LLM required.
Tests structured output, routing, evaluation, and spec generation.
"""
import json
import pytest
from typing import Any, Type
from pydantic import BaseModel, Field

from intelligence.contracts import (
    BaseIntelligenceProvider,
    IntelligenceRequest,
    IntelligenceResponse,
    StructuredGenerationRequest,
    StructuredGenerationResult,
    ModelProfile,
    ModelCapability,
    TaskComplexity,
)
from intelligence.router import ModelRouter
from intelligence.core import IntelligenceCore
from intelligence.evaluation import (
    EvalSample,
    EvalDataset,
    EvalPrediction,
    EvalRunResult,
    EvaluationRunner,
    SuccessRateMetric,
    MeanLatencyMetric,
    ExactMatchMetric,
    ContainsReferenceMetric,
    StructuredOutputValidityMetric,
)
from intelligence.spec_generator import generate_specification, ParsedSpecification


# ── Fake Provider ─────────────────────────────────────────

class FakeIntelligenceProvider(BaseIntelligenceProvider):
    """Deterministic fake provider for unit testing."""

    def __init__(self, responses=None, structured_responses=None):
        self._responses = responses or {}
        self._structured_responses = structured_responses or {}
        self._call_count = 0
        self._profile = ModelProfile(
            name="fake-model",
            provider="fake",
            capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.STRUCTURED_OUTPUT],
            context_window=4096,
            cost_tier="low",
            local=True,
        )

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    def generate(self, request: IntelligenceRequest) -> IntelligenceResponse:
        self._call_count += 1
        content = self._responses.get(request.task, f"Fake response for: {request.prompt[:50]}")
        return IntelligenceResponse(
            content=content,
            model="fake-model",
            provider="fake",
            usage={"input_tokens": 10, "output_tokens": 20},
            latency_ms=5.0,
            success=True,
        )

    def structured_generate(self, request: StructuredGenerationRequest) -> StructuredGenerationResult:
        self._call_count += 1
        schema_class = request.output_schema

        # Return a pre-configured structured response if available
        if schema_class.__name__ in self._structured_responses:
            data = self._structured_responses[schema_class.__name__]
            if isinstance(data, dict):
                validated = schema_class.model_validate(data)
            else:
                validated = data
            return StructuredGenerationResult(
                data=validated,
                raw_response=json.dumps(data) if isinstance(data, dict) else str(data),
                model="fake-model",
                provider="fake",
                attempts=1,
                success=True,
                latency_ms=3.0,
            )

        return StructuredGenerationResult(
            success=False,
            error="No structured response configured for this schema",
            model="fake-model",
            provider="fake",
            attempts=1,
            latency_ms=1.0,
        )


# ── Intelligence Contracts Tests ──────────────────────────

class TestIntelligenceContracts:
    def test_request_creation(self):
        req = IntelligenceRequest(prompt="Hello", task="generate")
        assert req.prompt == "Hello"
        assert req.complexity == TaskComplexity.MODERATE

    def test_response_creation(self):
        resp = IntelligenceResponse(content="World", model="test", provider="test")
        assert resp.success is True
        assert resp.content == "World"

    def test_model_profile(self):
        profile = ModelProfile(name="test", provider="test", capabilities=[ModelCapability.TEXT_GENERATION])
        assert ModelCapability.TEXT_GENERATION in profile.capabilities


# ── Model Router Tests ────────────────────────────────────

class TestModelRouter:
    def test_register_and_route(self):
        provider = FakeIntelligenceProvider()
        router = ModelRouter(default_provider=provider)
        req = IntelligenceRequest(prompt="test")
        routed = router.route(req)
        assert routed == provider

    def test_explicit_provider_override(self):
        p1 = FakeIntelligenceProvider()
        p2 = FakeIntelligenceProvider()
        p2._profile = ModelProfile(name="special", provider="special", capabilities=[])
        router = ModelRouter(default_provider=p1)
        router.register_provider(p2)

        req = IntelligenceRequest(prompt="test", metadata={"provider": "special"})
        routed = router.route(req)
        assert routed.profile.name == "special"

    def test_no_provider_raises(self):
        router = ModelRouter()
        req = IntelligenceRequest(prompt="test")
        with pytest.raises(RuntimeError, match="No intelligence providers"):
            router.route(req)

    def test_generate_through_router(self):
        provider = FakeIntelligenceProvider(responses={"generate": "routed answer"})
        router = ModelRouter(default_provider=provider)
        resp = router.generate(IntelligenceRequest(prompt="question"))
        assert resp.success is True
        assert resp.content == "routed answer"


# ── Intelligence Core Tests ───────────────────────────────

class TestIntelligenceCore:
    def test_generate(self):
        provider = FakeIntelligenceProvider(responses={"generate": "core response"})
        router = ModelRouter(default_provider=provider)
        core = IntelligenceCore(router)
        resp = core.generate("Hello world")
        assert resp.success is True
        assert resp.content == "core response"

    def test_understand(self):
        provider = FakeIntelligenceProvider(responses={"generate": "understood"})
        router = ModelRouter(default_provider=provider)
        core = IntelligenceCore(router)
        resp = core.understand("Some complex text")
        assert resp.success is True

    def test_reason(self):
        provider = FakeIntelligenceProvider(responses={"generate": "step by step"})
        router = ModelRouter(default_provider=provider)
        core = IntelligenceCore(router)
        resp = core.reason("Why is the sky blue?")
        assert resp.success is True

    def test_plan(self):
        provider = FakeIntelligenceProvider(responses={"generate": "1. Research 2. Build"})
        router = ModelRouter(default_provider=provider)
        core = IntelligenceCore(router)
        resp = core.plan("Build a research AI")
        assert resp.success is True


# ── Structured Output Tests ───────────────────────────────

class SimpleOutput(BaseModel):
    answer: str
    confidence: float = 0.0

class TestStructuredOutput:
    def test_structured_generate_success(self):
        provider = FakeIntelligenceProvider(
            structured_responses={
                "SimpleOutput": {"answer": "42", "confidence": 0.95}
            }
        )
        router = ModelRouter(default_provider=provider)
        core = IntelligenceCore(router)
        result = core.structured_generate("What is the answer?", SimpleOutput)
        assert result.success is True
        assert result.data.answer == "42"
        assert result.data.confidence == 0.95

    def test_structured_generate_missing_schema(self):
        provider = FakeIntelligenceProvider()
        router = ModelRouter(default_provider=provider)
        core = IntelligenceCore(router)
        result = core.structured_generate("test", SimpleOutput)
        assert result.success is False


# ── Spec Generator Tests ──────────────────────────────────

class TestSpecGenerator:
    def test_nl_to_spec_success(self):
        provider = FakeIntelligenceProvider(
            structured_responses={
                "ParsedSpecification": {
                    "name": "Research Assistant",
                    "purpose": "Analyze documents and produce reports",
                    "capabilities": ["chat", "retrieval", "verification"],
                    "knowledge_mode": "hybrid",
                    "reranking": True,
                    "evaluation_metrics": ["factuality", "citation_correctness"],
                }
            }
        )
        router = ModelRouter(default_provider=provider)

        spec, result = generate_specification(
            "Build me a research AI that retrieves evidence and verifies sources",
            router,
        )

        assert result.success is True
        assert spec is not None
        assert spec.identity.name == "Research Assistant"
        assert len(spec.capabilities) == 3
        assert spec.knowledge.mode == "hybrid"
        assert spec.knowledge.reranking is True
        assert spec.evaluation.metrics == ["factuality", "citation_correctness"]

    def test_nl_to_spec_chat_only(self):
        provider = FakeIntelligenceProvider(
            structured_responses={
                "ParsedSpecification": {
                    "name": "Simple Chatbot",
                    "purpose": "Answer questions",
                    "capabilities": ["chat"],
                    "knowledge_mode": "none",
                    "reranking": False,
                    "evaluation_metrics": ["response_quality"],
                }
            }
        )
        router = ModelRouter(default_provider=provider)
        spec, result = generate_specification("Build a simple chatbot", router)

        assert spec is not None
        assert len(spec.capabilities) == 1
        assert spec.capabilities[0].name == "chat"
        assert spec.knowledge is None  # No retrieval

    def test_nl_to_spec_failure(self):
        provider = FakeIntelligenceProvider()  # No structured response configured
        router = ModelRouter(default_provider=provider)
        spec, result = generate_specification("Build me an AI", router)
        assert spec is None
        assert result.success is False

    def test_spec_serialization(self):
        provider = FakeIntelligenceProvider(
            structured_responses={
                "ParsedSpecification": {
                    "name": "Test System",
                    "purpose": "Testing",
                    "capabilities": ["chat", "retrieval"],
                    "knowledge_mode": "dense",
                    "reranking": False,
                    "evaluation_metrics": [],
                }
            }
        )
        router = ModelRouter(default_provider=provider)
        spec, _ = generate_specification("test", router)

        # Verify round-trip serialization
        json_str = spec.model_dump_json()
        from core.creation.domain import AISystemSpecification
        restored = AISystemSpecification.model_validate_json(json_str)
        assert restored.identity.name == "Test System"
        assert restored.purpose == "Testing"


# ── Evaluation Framework Tests ────────────────────────────

class TestEvaluation:
    def test_success_rate_metric(self):
        predictions = [
            EvalPrediction(sample_id="1", output="a", success=True),
            EvalPrediction(sample_id="2", output="b", success=True),
            EvalPrediction(sample_id="3", output="", success=False, error="failed"),
        ]
        metric = SuccessRateMetric()
        result = metric.compute(predictions, [])
        assert result.value == pytest.approx(2.0 / 3.0, abs=0.01)

    def test_exact_match_metric(self):
        samples = [
            EvalSample(id="1", input="q1", reference="answer1"),
            EvalSample(id="2", input="q2", reference="answer2"),
        ]
        predictions = [
            EvalPrediction(sample_id="1", output="answer1", success=True),
            EvalPrediction(sample_id="2", output="wrong", success=True),
        ]
        metric = ExactMatchMetric()
        result = metric.compute(predictions, samples)
        assert result.value == 0.5

    def test_contains_reference_metric(self):
        samples = [
            EvalSample(id="1", input="q", reference="Paris"),
        ]
        predictions = [
            EvalPrediction(sample_id="1", output="The capital of France is Paris.", success=True),
        ]
        metric = ContainsReferenceMetric()
        result = metric.compute(predictions, samples)
        assert result.value == 1.0

    def test_structured_output_validity(self):
        predictions = [
            EvalPrediction(sample_id="1", output='{"key": "value"}', success=True),
            EvalPrediction(sample_id="2", output='not json', success=True),
        ]
        metric = StructuredOutputValidityMetric()
        result = metric.compute(predictions, [])
        assert result.value == 0.5

    def test_evaluation_runner(self):
        dataset = EvalDataset(
            name="test_dataset",
            samples=[
                EvalSample(id="1", input="2+2", reference="4"),
                EvalSample(id="2", input="3+3", reference="6"),
            ],
        )

        def fake_system(input_str: str) -> str:
            # Simple calculator
            try:
                return str(eval(input_str))
            except Exception:
                return "error"

        runner = EvaluationRunner(metrics=[
            SuccessRateMetric(),
            MeanLatencyMetric(),
            ExactMatchMetric(),
        ])
        result = runner.run(dataset, fake_system, system_name="calculator")

        assert result.total_samples == 2
        assert result.successful_samples == 2
        assert result.system_name == "calculator"
        assert any(m.name == "success_rate" and m.value == 1.0 for m in result.metrics)
        assert any(m.name == "exact_match" and m.value == 1.0 for m in result.metrics)

    def test_evaluation_with_failures(self):
        dataset = EvalDataset(
            name="failing_dataset",
            samples=[EvalSample(id="1", input="test")],
        )

        def failing_system(input_str: str) -> str:
            raise RuntimeError("System crash")

        runner = EvaluationRunner()
        result = runner.run(dataset, failing_system)
        assert result.successful_samples == 0
        assert result.predictions[0].success is False
        assert "System crash" in result.predictions[0].error

    def test_eval_result_serialization(self):
        result = EvalRunResult(
            dataset_name="test",
            system_name="test",
            metrics=[MetricResult(name="acc", value=0.9)],
            total_samples=10,
            successful_samples=9,
        )
        json_str = result.model_dump_json()
        restored = EvalRunResult.model_validate_json(json_str)
        assert restored.metrics[0].value == 0.9


# Import for serialization test
from intelligence.evaluation import MetricResult
