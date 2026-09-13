import copy
from typing import Dict, Any, List
from core.registry import ComponentRegistry
from retrieval.engine import RetrievalPipeline
from core.graph import ExecutionGraph

class ApplicationContainer:
    def __init__(self, config: Dict[str, Any] = None, registry: ComponentRegistry = None):
        self.config = config or {}
        self._instances = {}
        self.registry = registry if registry else ComponentRegistry()
        
    def get_embedding_provider(self, name: str = "sentence_transformers"):
        if "embedding_provider" not in self._instances:
            self._instances["embedding_provider"] = self.registry.get_embedding_provider(name)
        return self._instances["embedding_provider"]

    def get_vector_store(self, name: str = "qdrant"):
        if "vector_store" not in self._instances:
            emb = self.get_embedding_provider()
            self._instances["vector_store"] = self.registry.get_retriever(name, embedding_provider=emb)
        return self._instances["vector_store"]
        
    def get_bm25_store(self, name: str = "bm25"):
        if "bm25_store" not in self._instances:
            self._instances["bm25_store"] = self.registry.get_retriever(name)
        return self._instances["bm25_store"]

    def get_llm(self, name: str = "default"):
        if "llm" not in self._instances:
            self._instances["llm"] = self.registry.get_llm(name)
        return self._instances["llm"]

    def get_reranker(self, name: str = "cross_encoder"):
        if "reranker" not in self._instances:
            self._instances["reranker"] = self.registry.get_reranker(name)
        return self._instances["reranker"]

    def get_intelligence(self):
        if "intelligence" not in self._instances:
            from intelligence.core import IntelligenceCore
            from intelligence.router import ModelRouter
            from intelligence.providers.hf_provider import HuggingFaceProvider
            
            # Since Ollama might be unstable on this CPU-only VM, we register HuggingFace as default
            # You could also add OllamaProvider here if desired.
            provider = HuggingFaceProvider()
            router = ModelRouter(default_provider=provider)
            
            # try:
            #     # Try to use Ollama if available
            #     from intelligence.providers.ollama_provider import OllamaProvider
            #     import requests
            #     if requests.get("http://localhost:11434/api/tags", timeout=1).status_code == 200:
            #         ollama = OllamaProvider(model="qwen3:1.7b")
            #         router.register_provider(ollama)
            #         router.set_default("qwen3:1.7b")
            # except Exception:
            #     pass
                
            self._instances["intelligence"] = IntelligenceCore(router)
        return self._instances["intelligence"]

    def build_pipeline(self, retrieval_mode: str = "rerank", expand_neighbors: bool = False, dense_weight: float = 1.0, sparse_weight: float = 1.0) -> RetrievalPipeline:
        retrievers = []
        if retrieval_mode in ["dense", "hybrid", "rerank"]:
            retrievers.append(self.get_vector_store())
        if retrieval_mode in ["sparse", "hybrid", "rerank"]:
            retrievers.append(self.get_bm25_store())

        fusion_strategy = None
        if len(retrievers) > 1:
            fusion_strategy = self.registry.get_fusion("rrf", weights=[dense_weight, sparse_weight])

        reranker = None
        if retrieval_mode == "rerank":
            reranker = self.get_reranker()

        post_processors = []
        if expand_neighbors:
            dense_r = self.get_vector_store()
            pp = self.registry.get_post_processor("neighbor_expansion", vector_store=dense_r.vector_store)
            post_processors.append(pp)

        return RetrievalPipeline(
            retrievers=retrievers,
            fusion_strategy=fusion_strategy,
            reranker=reranker,
            post_processors=post_processors
        )

    def build_graph(self, retrieval_mode: str = "rerank", expand_neighbors: bool = False) -> ExecutionGraph:
        from orchestration.nodes.basic_nodes import QueryNode, RetrieveNode, ContextNode, GenerateNode, VerifyNode
        
        query_proc = self.registry.get_query_processor("passthrough")
        retrieval_pipe = self.build_pipeline(retrieval_mode=retrieval_mode, expand_neighbors=expand_neighbors)
        context_assembler = self.registry.get_context_assembler("default")
        generator = self.registry.get_generator("standard", llm_provider=self.get_llm())
        verifier = self.registry.get_verifier("citation")
        
        q_node = QueryNode(query_proc)
        r_node = RetrieveNode(retrieval_pipe)
        c_node = ContextNode(context_assembler)
        g_node = GenerateNode(generator)
        v_node = VerifyNode(verifier)
        
        graph = ExecutionGraph()
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
        
        return graph


    def build_agentic_graph(self) -> ExecutionGraph:
        from orchestration.agentic.nodes import PlannerNode, DecisionNode, ToolExecutionNode, CriticNode
        from orchestration.agentic.tools import RAGTool
        
        llm = self.get_llm()
        generator = self.registry.get_generator("standard", llm_provider=llm)
        pipeline = self.build_pipeline(retrieval_mode="rerank")
        assembler = self.registry.get_context_assembler("default")
        
        rag_tool = RAGTool(pipeline, assembler)
        
        planner = PlannerNode(llm)
        decision = DecisionNode()
        tool_exec = ToolExecutionNode(tools=[rag_tool])
        critic = CriticNode(generator)
        
        graph = ExecutionGraph()
        graph.add_node(planner)
        graph.add_node(decision)
        graph.add_node(tool_exec)
        graph.add_node(critic)
        
        graph.set_entry_point("planner")
        graph.add_edge("planner", "decision")
        graph.add_edge("critic", "END")
        # decision routes conditionally, we already built that in DecisionNode routing_decision
        
        return graph

def build_container() -> ApplicationContainer:
    import vectorstore.qdrant_client
    import vectorstore.bm25_store
    import providers.llm
    import providers.embeddings
    import retrieval.fusion
    import retrieval.post_processors
    import retrieval.reranker
    import retrieval.assembler
    import verification.grounding
    import orchestration.nodes.processors
    
    registry = ComponentRegistry()
    
    vectorstore.qdrant_client.register(registry)
    vectorstore.bm25_store.register(registry)
    providers.llm.register(registry)
    providers.embeddings.register(registry)
    retrieval.fusion.register(registry)
    retrieval.post_processors.register(registry)
    retrieval.reranker.register(registry)
    retrieval.assembler.register(registry)
    verification.grounding.register(registry)
    orchestration.nodes.processors.register(registry)
    
    return ApplicationContainer(registry=registry)
