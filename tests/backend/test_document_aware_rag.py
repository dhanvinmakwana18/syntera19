import pytest
from ingestion.parser import update_heading_stack, format_section_path, chunk_text
from retrieval.assembler import expand_and_organize_context

def test_heading_stack():
    stack = []
    stack = update_heading_stack(stack, "# Chapter 1\nSome text")
    assert format_section_path(stack) == "Chapter 1"
    
    stack = update_heading_stack(stack, "## 1.1 Intro\nMore text")
    assert format_section_path(stack) == "Chapter 1 > 1.1 Intro"
    
    stack = update_heading_stack(stack, "### Details\nHello")
    assert format_section_path(stack) == "Chapter 1 > 1.1 Intro > Details"
    
    stack = update_heading_stack(stack, "## 1.2 Outro\nBye")
    assert format_section_path(stack) == "Chapter 1 > 1.2 Outro"
    
    stack = update_heading_stack(stack, "# Chapter 2\nEnd")
    assert format_section_path(stack) == "Chapter 2"

def test_context_assembly_grouping():
    # Mock chunks from Qdrant
    candidates = [
        {"id": "doc1_chunk_5", "score": 0.9, "payload": {"text": "chunk 5", "source": "doc1", "chunk_index": 5, "section_path": "Chapter 1"}},
        {"id": "doc2_chunk_1", "score": 0.8, "payload": {"text": "doc 2 chunk 1", "source": "doc2", "chunk_index": 1, "section_path": "Root"}},
        {"id": "doc1_chunk_2", "score": 0.7, "payload": {"text": "chunk 2", "source": "doc1", "chunk_index": 2, "section_path": "Intro"}},
    ]
    
    context, sources = expand_and_organize_context(candidates, expand_neighbors=False)
    
    # doc1 has highest score (0.9), so doc1 chunks should appear first.
    # Within doc1, chunk 2 should appear before chunk 5 because of index ordering.
    assert len(sources) == 3
    assert sources[0]["filename"] == "doc1"
    assert sources[0]["chunk_index"] == 2
    assert sources[1]["filename"] == "doc1"
    assert sources[1]["chunk_index"] == 5
    assert sources[2]["filename"] == "doc2"
    assert sources[2]["chunk_index"] == 1

def test_context_expansion_mock(monkeypatch):
    import uuid
    from vectorstore.qdrant_client import vector_store
    
    candidates = [
        {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, "doc1_chunk_2")), "score": 0.9, "payload": {"text": "target chunk", "source": "doc1", "chunk_index": 2, "section_path": "Root"}}
    ]
    
    def mock_get_points(ids):
        # Return a mock chunk 1 (predecessor) and chunk 3 (successor)
        results = []
        for i in ids:
            if str(uuid.uuid5(uuid.NAMESPACE_URL, "doc1_chunk_1")) == i:
                results.append({"id": i, "payload": {"text": "predecessor chunk", "source": "doc1", "chunk_index": 1, "section_path": "Root"}})
            elif str(uuid.uuid5(uuid.NAMESPACE_URL, "doc1_chunk_3")) == i:
                results.append({"id": i, "payload": {"text": "successor chunk", "source": "doc1", "chunk_index": 3, "section_path": "Root"}})
        return results
        
    monkeypatch.setattr(vector_store, "get_points_by_ids", mock_get_points)
    
    context, sources = expand_and_organize_context(candidates, expand_neighbors=True)
    
    assert len(sources) == 3
    assert sources[0]["chunk_index"] == 1
    assert sources[0]["is_expanded"] == True
    assert sources[1]["chunk_index"] == 2
    assert sources[1]["is_expanded"] == False
    assert sources[2]["chunk_index"] == 3
    assert sources[2]["is_expanded"] == True
