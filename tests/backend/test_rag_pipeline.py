import pytest
from core.domain import Query, RetrievalResult, Node
from retrieval.query_transform import transform_query
from retrieval.fusion import RRFFusionStrategy

def test_query_transform():
    query_str = transform_query("What is X?")
    assert query_str == "What is X?"

def test_fusion_rrf():
    q = Query(text="test")
    c1 = [
        RetrievalResult(node=Node(id="1", text="A"), score=0.9),
        RetrievalResult(node=Node(id="2", text="B"), score=0.8)
    ]
    c2 = [
        RetrievalResult(node=Node(id="2", text="B"), score=0.85),
        RetrievalResult(node=Node(id="3", text="C"), score=0.7)
    ]
    
    rrf = RRFFusionStrategy(weights=[1.0, 1.0])
    fused = rrf.fuse([c1, c2])
    
    assert len(fused) == 3
    assert fused[0].node.id == "2"

def test_assembler():
    candidates = [
        RetrievalResult(node=Node(id="1", text="Duplicate", metadata={"source": "f1.pdf", "page": 1}), score=0.9),
        RetrievalResult(node=Node(id="2", text="Duplicate", metadata={"source": "f1.pdf", "page": 2}), score=0.8),
        RetrievalResult(node=Node(id="3", text="Unique", metadata={"source": "f2.txt", "page": 1}), score=0.5)
    ]
    
    from retrieval.assembler import PipelineContextAssembler
    result = PipelineContextAssembler().assemble(candidates)
    
    assert len(result.sources) == 3 
    assert "Duplicate" in result.text
