import pytest
from retrieval.query_transform import transform_query
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.assembler import assemble_context
from verification.grounding import validate_citations
from retrieval.pipeline import retrieve_documents

def test_query_transform():
    query = "  What   is   SYNTERA?  "
    assert transform_query(query) == "What is SYNTERA?"
    
def test_fusion():
    dense = [
        {"id": "A", "score": 0.9, "payload": {"text": "Alpha"}},
        {"id": "B", "score": 0.8, "payload": {"text": "Beta"}}
    ]
    sparse = [
        {"id": "B", "score": 2.5, "payload": {"text": "Beta"}},
        {"id": "C", "score": 1.5, "payload": {"text": "Gamma"}}
    ]
    
    fused = reciprocal_rank_fusion(dense, sparse, k=60, limit=5)
    
    # B is in both at rank 1 and 0 (dense rank 1, sparse rank 0)
    # A is dense rank 0
    # C is sparse rank 1
    # B should be rank 1 overall
    assert len(fused) == 3
    assert fused[0]["id"] == "B"
    
def test_assembler():
    candidates = [
        {"rerank_score": 0.9, "payload": {"text": "Duplicate", "source": "f1.pdf", "page": 1}},
        {"rerank_score": 0.8, "payload": {"text": "Duplicate", "source": "f1.pdf", "page": 2}},
        {"rerank_score": 0.5, "payload": {"text": "Unique", "source": "f2.txt", "page": 1}}
    ]
    
    context, sources = assemble_context(candidates)
    
    # Duplicate should be removed
    assert len(sources) == 2
    assert sources[0]["text"] == "Duplicate"
    assert sources[1]["text"] == "Unique"
    assert "[Source 1]" in context
    assert "[Source 2]" in context

def test_validate_citations():
    sources = [{"id": 1, "text": "Something"}]
    
    valid_response = "The answer is something [Source 1]."
    assert validate_citations(valid_response, sources) == valid_response
    
    invalid_response = "The answer is hallucinated [Source 2]."
    validated = validate_citations(invalid_response, sources)
    assert "[SYSTEM WARNING" in validated
