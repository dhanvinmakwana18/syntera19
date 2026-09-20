import time
from pprint import pprint

# Container & Setup
from core.container import ApplicationContainer
from core.creation.builder import AISystemBuilder
from core.creation.capabilities import CapabilityRegistry
from core.creation.standard_capabilities import RetrievalCapability, ChatCapability, VerificationCapability, PlanningCapability
from core.creation.domain import AISystemSpecification, SystemIdentity

from intelligence.evaluation import EvalSample, EvalDataset, EvaluationRunner, SuccessRateMetric, MeanLatencyMetric
from intelligence.core import IntelligenceCore
from intelligence.router import ModelRouter
from intelligence.providers.hf_provider import HuggingFaceProvider

# Evolution Components
from evolution.contracts import SelectionPolicy, SelectionRule, SelectionObjective, EvolutionEvent
from evolution.engine import EvolutionEngine

def demo_evolution():
    print("============================================================")
    print(" PHASE 9: SYNTERA EVOLUTION ENGINE — LIVE DEMONSTRATION")
    print("============================================================\n")
    
    # 1. Setup DI Container and Intelligence
    print("[1] Initializing Infrastructure...")
    container = ApplicationContainer()
    intelligence = container.get_intelligence()
    
    # Register capabilities
    registry = CapabilityRegistry()
    registry.register(RetrievalCapability())
    registry.register(ChatCapability())
    registry.register(VerificationCapability())
    registry.register(PlanningCapability())
    
    # Register core retrieval components (as done in phase 8)
    from providers.embeddings import EmbeddingProvider
    from retrieval.assembler import PipelineContextAssembler
    from core.contracts import BaseQueryProcessor, Query
    from vectorstore.qdrant_client import create_qdrant_retriever
    from retrieval.reranker import CrossEncoderReranker
    from verification.grounding import CitationVerifier
    
    class PassthroughQueryProcessor(BaseQueryProcessor):
        def process(self, query: Query) -> list[Query]:
            return [query]
            
    container.registry.register_embedding_provider("sentence_transformers", lambda: EmbeddingProvider())
    container.registry.register_retriever("qdrant", lambda embedding_provider: create_qdrant_retriever(embedding_provider))
    container.registry.register_reranker("cross_encoder", lambda: CrossEncoderReranker())
    container.registry.register_context_assembler("default", lambda: PipelineContextAssembler())
    container.registry.register_query_processor("passthrough", lambda: PassthroughQueryProcessor())
    container.registry.register_verifier("citation", lambda: CitationVerifier())
    
    builder = AISystemBuilder(registry, container)
    
    # 2. Setup Evaluation Dataset
    # We want a dataset that tests deep reasoning and retrieval (or tools). 
    # A simple chat model will fail to answer the complex internal spec question or multi-step logic.
    dataset = EvalDataset(
        name="complex_reasoning_benchmark_v1",
        samples=[
            EvalSample(id="1", input="What is 53 * 19? Use tools to calculate if needed. Output just the number.", reference="1007"),
            EvalSample(id="2", input="Find the name of the capability used for generating plans in Syntera.", reference="PlanningCapability")
        ]
    )
    
    # 3. Create Baseline System v1 (A naive Chatbot)
    from core.creation.domain import CapabilityRequirement
    spec_v1 = AISystemSpecification(
        identity=SystemIdentity(id="sys-v1", name="NaiveChatbot", version="1.0"),
        purpose="A simple chatbot that answers questions off the top of its head.",
        capabilities=[CapabilityRequirement(name="chat")]
    )
    
    print("[2] Building Baseline System (v1)...")
    system_v1 = builder.build(spec_v1)
    
    # 4. Setup Evolution Engine
    policy = SelectionPolicy(
        rules=[
            SelectionRule(objective=SelectionObjective.MAXIMIZE_SUCCESS, weight=10.0),
            SelectionRule(objective=SelectionObjective.MINIMIZE_LATENCY, weight=1.0)
        ]
    )
    
    eval_runner = EvaluationRunner(metrics=[SuccessRateMetric(), MeanLatencyMetric()])
    engine = EvolutionEngine(intelligence, builder, eval_runner, policy)
    
    # Add an observer for logging
    def on_event(event: EvolutionEvent, data: dict):
        print(f"   [EVENT] {event.value}: {data}")
    engine.add_hook(on_event)
    
    print("\n[3] STARTING EVOLUTION LOOP")
    print("    - Goal: Evolve system v1 to pass the complex reasoning benchmark.")
    print("    - Policy: Maximize Success Rate, Minimize Latency tiebreaker.")
    
    start_time = time.time()
    
    # Run Evolution (max 1 iteration for demo to save time, but could be N)
    system_v2 = engine.evolve(system_v1, dataset, max_iterations=1, num_candidates=1)
    
    elapsed = time.time() - start_time
    
    print("\n[4] EVOLUTION COMPLETE")
    print(f"    - Total Evolution Time: {elapsed:.2f}s")
    
    # Results
    res_v1 = system_v1.metadata.get("eval_result")
    res_v2 = system_v2.metadata.get("eval_result")
    
    if res_v1:
        metrics1 = {m.name: m.value for m in res_v1.metrics}
        print(f"    - Baseline (v1) Success Rate: {metrics1.get('success_rate', 0):.2f}")
    if res_v2:
        metrics2 = {m.name: m.value for m in res_v2.metrics}
        print(f"    - Evolved (v2) Success Rate:  {metrics2.get('success_rate', 0):.2f}")
        
    print(f"\n    - Final System: {system_v2.identity.name} v{system_v2.identity.version}")
    print(f"    - Final Capabilities: {system_v2.resolved_capabilities}")

if __name__ == "__main__":
    demo_evolution()
