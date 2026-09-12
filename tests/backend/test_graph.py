import pytest
from core.graph import ExecutionGraph
from orchestration.state import RAGState
from core.domain import Query, RetrievalResult, Node, GenerationContext, GenerationResult, VerificationResult
from orchestration.nodes.basic_nodes import QueryNode, RetrieveNode, ContextNode, GenerateNode, VerifyNode

class FakeQueryProcessor:
    def process(self, query):
        return [Query(text=query.text + " processed")]

class FakePipeline:
    def run(self, query, limit=5):
        from retrieval.engine import PipelineResult
        nodes = [RetrievalResult(node=Node(id="1", text="Fake evidence"), score=0.9)]
        return PipelineResult(candidates=nodes, trace=[])

class FakeContextAssembler:
    def assemble(self, results, **kwargs):
        return GenerationContext(text="Fake context", sources=[{"id": 1, "text": "Fake evidence"}])

class FakeGenerator:
    def generate(self, query, context, **kwargs):
        return GenerationResult(answer="Fake answer [Source 1]")

class FakeVerifier:
    def verify(self, query, context, response, **kwargs):
        return VerificationResult(passed=True, reason="Valid")

def test_graph_execution_success():
    graph = ExecutionGraph()
    
    q_node = QueryNode(FakeQueryProcessor())
    r_node = RetrieveNode(FakePipeline())
    c_node = ContextNode(FakeContextAssembler())
    g_node = GenerateNode(FakeGenerator())
    v_node = VerifyNode(FakeVerifier())
    
    graph.add_node(q_node)
    graph.add_node(r_node)
    graph.add_node(c_node)
    graph.add_node(g_node)
    graph.add_node(v_node)
    
    graph.set_entry_point("query_processing")
    graph.add_edge("query_processing", "retrieval")
    graph.add_edge("retrieval", "context_assembly")
    graph.add_edge("context_assembly", "generation")
    graph.add_edge("generation", "verification")
    graph.add_edge("verification", "END")
    
    initial_state = RAGState(query=Query(text="What is fake?"))
    exec_result = graph.run(initial_state)
    
    final_state = exec_result.final_state
    
    assert len(final_state.processed_queries) == 1
    assert final_state.processed_queries[0].text == "What is fake? processed"
    assert len(final_state.retrieval_results) == 1
    assert final_state.context is not None
    assert final_state.generation_result is not None
    assert final_state.generation_result.answer == "Fake answer [Source 1]"
    assert final_state.verification_result is not None
    assert final_state.verification_result.passed is True
    assert len(exec_result.trace) >= 5
    
def test_graph_conditional_routing():
    graph = ExecutionGraph()
    
    q_node = QueryNode(FakeQueryProcessor())
    g_node = GenerateNode(FakeGenerator())
    
    graph.add_node(q_node)
    graph.add_node(g_node)
    
    graph.set_entry_point("query_processing")
    
    def route(state):
        if len(state.processed_queries) > 0:
            return ["generation"]
        return ["END"]
        
    graph.add_conditional_edge("query_processing", route)
    graph.add_edge("generation", "END")
    
    initial_state = RAGState(query=Query(text="Test condition"))
    exec_result = graph.run(initial_state)
    
    final_state = exec_result.final_state
    
    assert final_state.generation_result is not None
    assert "I cannot find sufficient evidence" in final_state.generation_result.answer
