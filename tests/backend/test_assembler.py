import pytest
from core.domain import Query, RetrievalResult, Node, GenerationContext
from retrieval.assembler import PipelineContextAssembler

def test_assembler():
    candidates = [
        RetrievalResult(node=Node(id="1", text="Duplicate", metadata={"source": "f1.pdf", "page": 1}), score=0.9),
        RetrievalResult(node=Node(id="2", text="Duplicate", metadata={"source": "f1.pdf", "page": 2}), score=0.8),
        RetrievalResult(node=Node(id="3", text="Unique", metadata={"source": "f2.txt", "page": 1}), score=0.5)
    ]
    
    result = PipelineContextAssembler().assemble(candidates)
    
    assert len(result.sources) == 3
    assert "Duplicate" in result.text
