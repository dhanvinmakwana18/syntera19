from typing import List, Any
from core.creation.capabilities import BaseCapability, GraphBlueprint
from core.creation.domain import AISystemSpecification
from orchestration.nodes.basic_nodes import QueryNode, RetrieveNode, ContextNode, GenerateNode, VerifyNode

class RetrievalCapability(BaseCapability):
    @property
    def name(self) -> str:
        return "retrieval"
        
    @property
    def dependencies(self) -> List[str]:
        return []
        
    def apply(self, blueprint: GraphBlueprint, container: Any, spec: AISystemSpecification) -> None:
        # Determine configuration from spec.knowledge if present
        expand = False
        mode = "rerank"
        if spec.knowledge:
            expand = spec.knowledge.expand_neighbors
            mode = spec.knowledge.mode
            
        pipeline = container.build_pipeline(retrieval_mode=mode, expand_neighbors=expand)
        assembler = container.registry.get_context_assembler("default")
        query_proc = container.registry.get_query_processor("passthrough")
        
        q_node = QueryNode(query_proc)
        r_node = RetrieveNode(pipeline)
        c_node = ContextNode(assembler)
        
        blueprint.nodes.extend([q_node, r_node, c_node])
        
        # Link internal nodes
        blueprint.edges.append(("query_processing", "retrieval"))
        blueprint.edges.append(("retrieval", "context_assembly"))
        
        if not blueprint.entry_point:
            blueprint.entry_point = "query_processing"

class ChatCapability(BaseCapability):
    @property
    def name(self) -> str:
        return "chat"
        
    @property
    def dependencies(self) -> List[str]:
        return []
        
    def apply(self, blueprint: GraphBlueprint, container: Any, spec: AISystemSpecification) -> None:
        intelligence = container.get_intelligence()
        
        from core.contracts import BaseGenerator
        from core.domain import Query, GenerationContext, GenerationResult
        
        class IntelligenceGeneratorAdapter(BaseGenerator):
            def __init__(self, core):
                self.core = core
                
            def generate(self, query: Query, context: GenerationContext) -> GenerationResult:
                ctx_text = "\n".join(context.texts) if context else ""
                prompt = f"Context:\n{ctx_text}\n\nQuestion: {query.text}"
                res = self.core.generate(prompt, system_prompt="Answer the question using ONLY the provided context.")
                return GenerationResult(answer=res.content)
                
        generator = IntelligenceGeneratorAdapter(intelligence)
        
        from orchestration.nodes.basic_nodes import GenerateNode
        g_node = GenerateNode(generator)
        blueprint.nodes.append(g_node)
        
        # Check if retrieval is present in blueprint nodes
        has_retrieval = any(n.name == "context_assembly" for n in blueprint.nodes)
        
        if has_retrieval:
            blueprint.edges.append(("context_assembly", "generation"))
        else:
            if not blueprint.entry_point:
                blueprint.entry_point = "generation"

class VerificationCapability(BaseCapability):
    @property
    def name(self) -> str:
        return "verification"
        
    @property
    def dependencies(self) -> List[str]:
        return [] # Can depend on chat, but we keep it decoupled and link structurally
        
    def apply(self, blueprint: GraphBlueprint, container: Any, spec: AISystemSpecification) -> None:
        verifier = container.registry.get_verifier("citation")
        v_node = VerifyNode(verifier)
        blueprint.nodes.append(v_node)
        
        has_generation = any(n.name == "generation" for n in blueprint.nodes)
        if has_generation:
            blueprint.edges.append(("generation", "verification"))
            blueprint.edges.append(("verification", "END"))
        else:
            blueprint.edges.append(("verification", "END"))
            if not blueprint.entry_point:
                blueprint.entry_point = "verification"

class PlanningCapability(BaseCapability):
    @property
    def name(self) -> str:
        return "planning"
        
    @property
    def dependencies(self) -> List[str]:
        return []
        
    def apply(self, blueprint: GraphBlueprint, container: Any, spec: AISystemSpecification) -> None:
        from orchestration.agentic.nodes import PlannerNode, DecisionNode, ToolExecutionNode, CriticNode
        from orchestration.agentic.tools import RAGTool
        
        intelligence = container.get_intelligence()
        planner = PlannerNode(intelligence)
        decision = DecisionNode()
        
        # Build tools
        tools = []
        if any(req.name == "retrieval" for req in spec.capabilities):
            # Convert retrieval to a tool if requested in an agentic loop
            pipe = container.build_pipeline(retrieval_mode="rerank")
            assembler = container.registry.get_context_assembler("default")
            tools.append(RAGTool(pipe, assembler))
            
        tool_exec = ToolExecutionNode(tools=tools)
        critic = CriticNode(intelligence)
        
        blueprint.nodes.extend([planner, decision, tool_exec, critic])
        
        # In a planning paradigm, planner is always entry point
        blueprint.entry_point = "planner"
        
        blueprint.edges.append(("planner", "decision"))
        blueprint.edges.append(("critic", "END"))
