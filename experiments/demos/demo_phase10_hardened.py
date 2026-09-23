import time
from pprint import pprint

from core.container import ApplicationContainer
from core.creation.builder import AISystemBuilder
from core.creation.capabilities import CapabilityRegistry
from core.creation.domain import AISystemSpecification, SystemIdentity, CapabilityRequirement
from intelligence.evaluation import EvalSample, EvalDataset, EvaluationRunner, ExactMatchMetric

from swarm.contracts import AgentSpec, SwarmSpec
from swarm.capabilities import MultiAgentCapability
from core.domain import Query
from core.graph.contracts import FailurePolicy

def run_benchmark():
    print("============================================================")
    print("============================================================\n")
    
    from core.container import build_container
    container = build_container()
    
    registry = CapabilityRegistry()
    registry.register(MultiAgentCapability())
    
    # Also register ChatCapability
    from core.creation.standard_capabilities import ChatCapability
    registry.register(ChatCapability())
    
    builder = AISystemBuilder(registry, container)
    
    dataset = EvalDataset(
        name="swarm_benchmark_hardened",
        samples=[
            EvalSample(id="1", input="Investigate the impact of multi-agent architectures on AI performance. Provide a final verified report.", reference="multi-agent architectures"),
            EvalSample(id="2", input="Analyze the impact of quantum computing on modern cryptography and summarize.", reference="quantum computing"),
            EvalSample(id="3", input="Compare RNNs and Transformers for sequence tasks and conclude which is better for parallelization.", reference="Transformers")
        ]
    )
    runner = EvaluationRunner(metrics=[ExactMatchMetric()])
    
    # 1. Single Agent Baseline
    print("\n[1] Running Single Agent Baseline...")
    
    single_spec = AISystemSpecification(
        identity=SystemIdentity(id="single-1", name="SingleAgent", version="1.0"),
        purpose="Do research and write a report.",
        capabilities=[CapabilityRequirement(name="chat")]
    )
    
    single_sys = builder.build(single_spec)
    
    start = time.time()
    from core.domain import GenerationContext
    # Chat capability entry point is usually generation which takes query and generationcontext
    # But graph expects state to have query text. Wait, ChatCapability uses orchestration basic nodes.
    # We will just evaluate it through evaluation runner.
    # Actually for evaluation runner we need to provide a function.
    def eval_single(text: str) -> str:
        # ChatCapability expects Query object and GenerationContext
        from core.domain import Query, GenerationContext
        class SingleState:
            query = Query(text=text)
            context = GenerationContext(texts=[])
            final_answer = ""
        st = SingleState()
        res = single_sys.execution_graph.run(st)
        return getattr(res.final_state, "final_answer", getattr(res.final_state, "answer", ""))

    try:
        single_res = runner.run(dataset, eval_single, system_name="SingleAgent")
        print(f"Single Agent Latency: {single_res.metrics[0].dict().get('details', {}).get('latency_ms', 0)} ms")
    except Exception as e:
        print(f"Single Agent failed: {e}")
        
    # 2. Sequential Swarm
    print("\n[2] Running Sequential Swarm Baseline...")
    researcher = {
        "name": "Researcher", "role": "Data Retriever", "system_prompt": "You are a Researcher. Find facts.", 
        "model": {"task": "generate", "structured_output": False, "context_window": "default", "local_preferred": True, "reasoning_required": "standard"},
        "tools": [{"name": "retrieval", "config": {}}]
    }
    analyst = {
        "name": "Analyst", "role": "Data Analyst", "system_prompt": "You are an Analyst. Analyze the data.", 
        "model": {"task": "generate", "structured_output": False, "context_window": "default", "local_preferred": True, "reasoning_required": "standard"},
        "tools": []
    }
    writer = {
        "name": "Writer", "role": "Report Writer", "system_prompt": "You are a Writer. Write the final report.", 
        "model": {"task": "generate", "structured_output": False, "context_window": "default", "local_preferred": True, "reasoning_required": "standard"},
        "tools": []
    }
    
    seq_workflow = {
        "agents": [researcher, analyst, writer],
        "workflow_type": "sequential",
        "entry_point": "Researcher",
        "edges": [],
        "failure_policy": "FAIL_FAST"
    }
    
    seq_spec = AISystemSpecification(
        identity=SystemIdentity(id="seq-swarm", name="SeqSwarm", version="1.0"),
        purpose="Sequential swarm",
        capabilities=[CapabilityRequirement(name="multi_agent")],
        workflow=seq_workflow
    )
    
    seq_sys = builder.build(seq_spec)
    
    def eval_swarm(sys):
        def _eval(text: str) -> str:
            from swarm.contracts import SwarmState
            st = SwarmState(query=Query(text=text))
            res = sys.execution_graph.run(st)
            return res.final_state.final_answer or ""
        return _eval
        
    start = time.time()
    seq_res = runner.run(dataset, eval_swarm(seq_sys), system_name="SeqSwarm")
    seq_latency = time.time() - start
    print(f"Sequential Swarm Success: {seq_res.successful_samples}/{seq_res.total_samples}")
    print(f"Sequential Swarm Latency: {seq_latency:.2f}s")
    
    # 3. Parallel Swarm with Dependencies
    print("\n[3] Running Parallel Swarm with True Dependencies...")
    verifier = {
        "name": "Verifier", "role": "Fact Checker", "system_prompt": "You are a Verifier. Check the facts.", 
        "model": {"task": "generate", "structured_output": False, "context_window": "default", "local_preferred": True, "reasoning_required": "standard"},
        "tools": []
    }
    
    par_workflow = {
        "agents": [researcher, analyst, verifier, writer],
        "workflow_type": "custom",
        "entry_point": "Researcher",
        "edges": [
            ["Researcher", "Analyst"],
            ["Researcher", "Verifier"],
            ["Analyst", "Writer"],
            ["Verifier", "Writer"]
        ],
        "failure_policy": "SKIP_DEPENDENTS"
    }
    
    par_spec = AISystemSpecification(
        identity=SystemIdentity(id="par-swarm", name="ParSwarm", version="1.0"),
        purpose="Parallel swarm",
        capabilities=[CapabilityRequirement(name="multi_agent")],
        workflow=par_workflow
    )
    
    par_sys = builder.build(par_spec)
    
    start = time.time()
    par_res = runner.run(dataset, eval_swarm(par_sys), system_name="ParSwarm")
    par_latency = time.time() - start
    print(f"Parallel Swarm Success: {par_res.successful_samples}/{par_res.total_samples}")
    print(f"Parallel Swarm Latency: {par_latency:.2f}s")
    
if __name__ == "__main__":
    run_benchmark()
