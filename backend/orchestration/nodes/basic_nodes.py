from typing import Dict, Any, List
import time
from orchestration.state import RAGState
from core.graph.contracts import GraphNode, NodeResult, RoutingDecision
from core.contracts import BaseQueryProcessor, BaseContextAssembler, BaseGenerator, BaseVerifier
from retrieval.engine import RetrievalPipeline

class QueryNode(GraphNode):
    def __init__(self, processor: BaseQueryProcessor):
        self.processor = processor
        
    @property
    def name(self) -> str:
        return "query_processing"
        
    def execute(self, state: RAGState) -> NodeResult:
        processed = self.processor.process(state.query)
        return NodeResult(state_updates={"processed_queries": processed})

class RetrieveNode(GraphNode):
    def __init__(self, pipeline: RetrievalPipeline, limit: int = 5):
        self.pipeline = pipeline
        self.limit = limit
        
    @property
    def name(self) -> str:
        return "retrieval"
        
    def execute(self, state: RAGState) -> NodeResult:
        target_query = state.processed_queries[0] if state.processed_queries else state.query
        result = self.pipeline.run(target_query, limit=self.limit)
        
        return NodeResult(
            state_updates={"retrieval_results": result.candidates},
            metrics={"retrieved_count": len(result.candidates)}
        )

class ContextNode(GraphNode):
    def __init__(self, assembler: BaseContextAssembler):
        self.assembler = assembler
        
    @property
    def name(self) -> str:
        return "context_assembly"
        
    def execute(self, state: RAGState) -> NodeResult:
        context = self.assembler.assemble(state.retrieval_results)
        return NodeResult(
            state_updates={"context": context},
            metrics={"sources_count": len(context.sources)}
        )

class GenerateNode(GraphNode):
    def __init__(self, generator: BaseGenerator):
        self.generator = generator
        
    @property
    def name(self) -> str:
        return "generation"
        
    def execute(self, state: RAGState) -> NodeResult:
        if not state.context or not state.context.sources:
            from core.domain import GenerationResult
            res = GenerationResult(answer="I cannot find sufficient evidence in the knowledge base to answer your question. Please upload relevant documents first.")
            return NodeResult(state_updates={"generation_result": res})
            
        target_query = state.processed_queries[0] if state.processed_queries else state.query
        result = self.generator.generate(target_query, state.context)
        return NodeResult(state_updates={"generation_result": result})

class VerifyNode(GraphNode):
    def __init__(self, verifier: BaseVerifier):
        self.verifier = verifier
        
    @property
    def name(self) -> str:
        return "verification"
        
    def execute(self, state: RAGState) -> NodeResult:
        if not state.generation_result or not state.context or not state.context.sources:
            from core.domain import VerificationResult
            res = VerificationResult(passed=False, reason="No generation or context to verify")
            return NodeResult(state_updates={"verification_result": res})
            
        target_query = state.processed_queries[0] if state.processed_queries else state.query
        result = self.verifier.verify(target_query, state.context, state.generation_result)
        return NodeResult(
            state_updates={"verification_result": result},
            metrics={"passed": result.passed}
        )
