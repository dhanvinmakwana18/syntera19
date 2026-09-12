import pytest
from core.creation.domain import (
    AISystemSpecification, SystemIdentity, CapabilityRequirement,
    KnowledgeRequirement
)
from core.creation.capabilities import CapabilityRegistry
from core.creation.standard_capabilities import ChatCapability, RetrievalCapability, VerificationCapability, PlanningCapability
from core.creation.builder import AISystemBuilder
from core.container import build_container

@pytest.fixture
def creation_env():
    container = build_container()
    registry = CapabilityRegistry()
    registry.register(ChatCapability())
    registry.register(RetrievalCapability())
    registry.register(VerificationCapability())
    registry.register(PlanningCapability())
    builder = AISystemBuilder(registry, container)
    return builder, container

def test_create_simple_chat(creation_env):
    builder, _ = creation_env
    spec = AISystemSpecification(
        identity=SystemIdentity(id="chat1", name="Basic Chat"),
        purpose="Simple conversation",
        capabilities=[CapabilityRequirement(name="chat")]
    )
    
    sys = builder.build(spec)
    assert sys.specification.identity.name == "Basic Chat"
    assert "chat" in sys.resolved_capabilities
    
    graph = sys.execution_graph
    assert "generation" in graph.nodes
    assert graph.entry_point == "generation"

def test_create_rag_system(creation_env):
    builder, _ = creation_env
    spec = AISystemSpecification(
        identity=SystemIdentity(id="rag1", name="RAG Agent"),
        purpose="Answer from docs",
        capabilities=[
            CapabilityRequirement(name="retrieval"),
            CapabilityRequirement(name="chat"),
            CapabilityRequirement(name="verification")
        ]
    )
    
    sys = builder.build(spec)
    graph = sys.execution_graph
    
    # Verify RAG topology
    assert "query_processing" in graph.nodes
    assert "retrieval" in graph.nodes
    assert "context_assembly" in graph.nodes
    assert "generation" in graph.nodes
    assert "verification" in graph.nodes
    
    assert graph.entry_point == "query_processing"
    assert graph.edges["query_processing"] == "retrieval"
    assert graph.edges["retrieval"] == "context_assembly"
    assert graph.edges["context_assembly"] == "generation"
    assert graph.edges["generation"] == "verification"

def test_create_research_agent(creation_env):
    builder, _ = creation_env
    spec = AISystemSpecification(
        identity=SystemIdentity(id="res1", name="Researcher"),
        purpose="Autonomous research",
        capabilities=[
            CapabilityRequirement(name="planning"),
            CapabilityRequirement(name="retrieval") # Requested as a capability to populate tools
        ]
    )
    
    sys = builder.build(spec)
    graph = sys.execution_graph
    
    assert "planner" in graph.nodes
    assert "decision" in graph.nodes
    assert "tool_execution" in graph.nodes
    assert "critic" in graph.nodes
    
    assert graph.entry_point == "planner"
    assert graph.edges["planner"] == "decision"
    
    # tool_exec -> decision loop is handled dynamically by RoutingDecision in agentic workflow
    
def test_validation_missing_capability(creation_env):
    builder, _ = creation_env
    spec = AISystemSpecification(
        identity=SystemIdentity(id="fail1", name="Fail"),
        purpose="Should fail",
        capabilities=[CapabilityRequirement(name="telepathy")]
    )
    
    with pytest.raises(ValueError, match="not found in registry"):
        builder.build(spec)
