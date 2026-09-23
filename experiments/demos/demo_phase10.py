import time
from pprint import pprint

# Container & Setup
from core.container import ApplicationContainer
from core.creation.builder import AISystemBuilder
from core.creation.capabilities import CapabilityRegistry
from core.creation.domain import AISystemSpecification, SystemIdentity, CapabilityRequirement
from intelligence.evaluation import EvalSample, EvalDataset, EvaluationRunner, ExactMatchMetric

from swarm.contracts import AgentSpec, SwarmSpec
from swarm.capabilities import MultiAgentCapability
from core.domain import Query

def demo_multi_agent():
    print("============================================================")
    print(" PHASE 10: SYNTERA MULTI-AGENT INTELLIGENCE ENGINE")
    print("============================================================\n")
    
    print("[1] Initializing Multi-Agent Infrastructure...")
    container = ApplicationContainer()
    intelligence = container.get_intelligence()
    
    # Register capabilities
    registry = CapabilityRegistry()
    registry.register(MultiAgentCapability())
    
    builder = AISystemBuilder(registry, container)
    
    # 2. Define Multi-Agent Spec
    print("[2] Creating Multi-Agent Specification...")
    
    researcher = {
        "name": "Researcher",
        "role": "Data Retriever",
        "system_prompt": "You are a Researcher. You look for facts. Output: 'Found data on X.'",
        "model": {"task": "generate", "structured_output": False, "context_window": "default", "local_preferred": True, "reasoning_required": "standard"},
        "tools": []
    }
    
    analyst = {
        "name": "Analyst",
        "role": "Data Analyst",
        "system_prompt": "You are an Analyst. You analyze data. Output: 'Analysis of data.'",
        "model": {"task": "generate", "structured_output": False, "context_window": "default", "local_preferred": True, "reasoning_required": "standard"},
        "tools": []
    }
    
    verifier = {
        "name": "Verifier",
        "role": "Fact Checker",
        "system_prompt": "You are a Verifier. You check facts. Output: 'Facts verified.'",
        "model": {"task": "generate", "structured_output": False, "context_window": "default", "local_preferred": True, "reasoning_required": "standard"},
        "tools": []
    }
    
    writer = {
        "name": "Writer",
        "role": "Report Writer",
        "system_prompt": "You are a Writer. You compile the final report. Read the message history and output a final summary.",
        "model": {"task": "generate", "structured_output": False, "context_window": "default", "local_preferred": True, "reasoning_required": "standard"},
        "tools": []
    }
    
    workflow = {
        "agents": [researcher, analyst, verifier, writer],
        "workflow_type": "custom",
        "entry_point": "Researcher",
        "edges": [
            # Researcher forks to Analyst and Verifier (Parallel Execution)
            ["Researcher", "Analyst"],
            ["Researcher", "Verifier"],
            
            # Analyst and Verifier join at Writer
            ["Analyst", "Writer"],
            ["Verifier", "Writer"],
            
            # Writer ends execution
            ["Writer", "END"]
        ]
    }
    
    spec = AISystemSpecification(
        identity=SystemIdentity(id="swarm-1", name="ResearchSwarm", version="1.0"),
        purpose="A specialized multi-agent swarm for researching, analyzing, verifying, and writing.",
        capabilities=[CapabilityRequirement(name="multi_agent")],
        workflow=workflow
    )
    
    # 3. Build Swarm
    print("[3] Building Multi-Agent System (ExecutionGraph)...")
    swarm_system = builder.build(spec)
    
    print("\n   => Swarm Topology Built:")
    for node_name in swarm_system.execution_graph.nodes:
        print(f"      Node: {node_name}")
    print("\n   => Edges:")
    for edge in swarm_system.execution_graph.edges.items():
        print(f"      {edge}")
        
    # 4. Execute Swarm
    print("\n[4] Executing Swarm System (Parallelism Enabled)...")
    from swarm.contracts import SwarmState
    
    initial_state = SwarmState(query=Query(text="Investigate the impact of multi-agent architectures on AI performance."))
    
    start_time = time.time()
    result = swarm_system.execution_graph.run(initial_state)
    latency = time.time() - start_time
    
    print("\n   => Execution Trace:")
    for trace_item in result.trace:
        print(f"      - {trace_item['node']} ({trace_item['latency']:.2f}s)")
        
    print(f"\n   => Total Execution Time: {latency:.2f}s")
    print(f"   => Final Answer: {result.final_state.final_answer}")
    
    # 5. Evaluate Swarm
    print("\n[5] Evaluating Multi-Agent System...")
    dataset = EvalDataset(
        name="swarm_benchmark_v1",
        samples=[
            EvalSample(id="1", input="Test prompt", reference=result.final_state.final_answer or "")
        ]
    )
    
    def evaluate_swarm(input_text: str) -> str:
        st = SwarmState(query=Query(text=input_text))
        res = swarm_system.execution_graph.run(st)
        return res.final_state.final_answer or ""
        
    runner = EvaluationRunner(metrics=[ExactMatchMetric()])
    eval_res = runner.run(dataset, evaluate_swarm, system_name="ResearchSwarm")
    
    print(f"   => Swarm Success Rate: {eval_res.successful_samples}/{eval_res.total_samples}")

if __name__ == "__main__":
    demo_multi_agent()
