# Syntera Intelligence Architecture (Phase 8)

## 1. Intelligence Core

Phase 8 introduces a **model-agnostic Intelligence layer** (`backend/intelligence/`) that
provides typed AI operations (generate, understand, reason, plan, structured_generate) to
the rest of Syntera without any direct SDK coupling.

```
    IntelligenceCore
         │
    ModelRouter
         │
    ┌────┴────┐
    │         │
  Ollama   [Future]
  Provider  OpenAI/Gemini
```

### Key Contracts
- `IntelligenceRequest` / `IntelligenceResponse` — typed request/response for all generation.
- `StructuredGenerationRequest` / `StructuredGenerationResult` — schema-validated output.
- `BaseIntelligenceProvider` — abstract provider interface.
- `ModelProfile` — describes a model's capabilities, cost tier, locality.

## 2. Model Providers

### Current: OllamaProvider
- Connects to local Ollama instance (`http://localhost:11434`)
- Supports `generate()` via `/api/chat` endpoint
- Supports `structured_generate()` via JSON mode + Pydantic validation + retry/repair
- Records token usage (input_tokens, output_tokens, tokens_per_sec)

### Future Extension Points
- `OpenAIProvider` — for GPT-4o/4.1 frontier models
- `GeminiProvider` — for Google Gemini models
- `HuggingFaceProvider` — for local transformers inference
- Any provider implementing `BaseIntelligenceProvider`

## 3. Model Routing

`ModelRouter` implements deterministic routing:
1. **Explicit override** — `metadata.provider` selects a specific provider
2. **Complexity-based** — COMPLEX tasks prefer frontier/high-tier models
3. **Default fallback** — routes to the registered default provider

The router is configurable, testable, and provider-independent.

## 4. Structured Output

The structured output pipeline:
```
    Prompt
      ↓
    LLM (JSON mode)
      ↓
    JSON parse
      ↓
    Pydantic validation
      ↓
    ┌──── valid? ────┐
    │ YES            │ NO
    ↓                ↓
    Return         Repair prompt
    typed obj      + retry (up to N)
```

This replaces the brittle `json.loads()` approach from Phase 6.

## 5. Natural Language → Specification

The most important intelligence capability:

```
    "Build me a research AI that can analyze PDFs,
     retrieve evidence, and produce verified reports."
              ↓
    IntelligenceCore.structured_generate()
              ↓
    ParsedSpecification (intermediate schema)
              ↓
    AISystemSpecification (full domain model)
              ↓
    Validation
```

- Uses an intermediate `ParsedSpecification` schema (simpler than full spec)
- LLM fills the intermediate schema via structured output
- Code transforms it into the full `AISystemSpecification`
- Avoids asking the LLM to fill deeply nested structures

## 6. Agentic Integration

The Intelligence Core integrates with the Phase 6 agentic graph:
- `PlannerNode` can use `IntelligenceCore.plan()` for structured planning
- `CriticNode` can use `IntelligenceCore.reason()` for evaluation
- Tool selection uses `IntelligenceCore.understand()` for intent classification

## 7. Neural Modular RAG

The existing Modular RAG pipeline uses neural components:
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (CPU)
- **Reranking**: Cross-encoder reranker (CPU)
- **Generation**: Via Intelligence Core → Ollama → local model
- **Verification**: Citation validation

Each component remains independently replaceable.

## 8. GPU Runtime

### Current Status
- PyTorch 2.13.0+cpu installed (CPU-only build)
- CUDA: NOT AVAILABLE in current environment
- Ollama: Running on CPU with qwen3:1.7b

### GPU Readiness
The architecture supports GPU acceleration when available:
- Ollama automatically uses GPU if CUDA drivers are present
- `sentence-transformers` supports `.to("cuda")` for embeddings
- Cross-encoder reranker supports GPU inference
- No code changes needed — just install CUDA-enabled PyTorch

### Honest Assessment
GPU is NOT being used in this phase because the environment has CPU-only PyTorch.
This is not a limitation of the architecture — it's an environment constraint.

## 9. Evaluation

### Framework
`backend/intelligence/evaluation.py` provides:
- `EvalDataset` / `EvalSample` — typed evaluation data
- `EvalPrediction` — system output with latency/usage tracking
- `EvalRunResult` — complete evaluation run with metrics
- `EvaluationRunner` — generic runner that feeds samples through any system function

### Built-in Metrics
- `SuccessRateMetric` — % of successful predictions
- `MeanLatencyMetric` — average response time
- `ExactMatchMetric` — exact string match against reference
- `ContainsReferenceMetric` — fuzzy containment check
- `StructuredOutputValidityMetric` — JSON validity rate

### Extensibility
Custom metrics implement `BaseMetric.compute(predictions, samples) → MetricResult`.

## 10. Benchmarking

The live test suite (`test_intelligence_live.py`) runs an actual benchmark:
- 5-sample knowledge QA dataset
- Measures success rate, latency, and answer containment
- Records per-sample latency and token usage
- Results are structured and serializable

## 11. Cost / Latency Tracking

Every `IntelligenceResponse` records:
- `latency_ms` — wall-clock time
- `usage.input_tokens` — prompt tokens
- `usage.output_tokens` — completion tokens
- `usage.tokens_per_sec` — inference throughput
- `model` / `provider` — which model served the request

For local inference: API cost = $0, but latency/throughput are tracked.

## 12. Failure Handling

Structured failure taxonomy:
- **Provider failure** → `IntelligenceResponse(success=False, error="...")`
- **JSON parse failure** → retry with repair prompt
- **Schema validation failure** → retry with error context
- **Timeout** → `requests.Timeout` caught and reported
- **Model unavailable** → graceful error in response

No silent error swallowing. All failures are typed and observable.

## 13. Framework Decision

LangChain/LangGraph remain **REJECTED**.
The native `ExecutionGraph` + `IntelligenceCore` combination provides:
- Structured output with Pydantic validation
- Model routing
- Agentic loops
- Dynamic graph construction

No framework dependency needed.

## 14. Current Limitations

1. **Single provider**: Only Ollama implemented. OpenAI/Gemini providers are future work.
2. **CPU inference**: No GPU acceleration in current environment.
3. **Small model**: qwen3:1.7b is capable but not frontier-class.
4. **No persistent evaluation**: Results are in-memory; future phases should persist to disk.
5. **IEG legacy**: The IEG orchestrator remains procedural.

## 15. Future Evolution Engine Integration

The evaluation framework produces serializable `EvalRunResult` artifacts.
The future Evolution Engine can:
1. Run evaluations on AI System v1
2. Identify weak metrics
3. Mutate the `AISystemSpecification`
4. Build AI System v2 via the Creation Engine
5. Run identical evaluations
6. Compare results
7. Select the better version

All artifacts (`AISystemSpecification`, `GeneratedAISystem`, `EvalRunResult`) are
Pydantic models with full serialization support.
