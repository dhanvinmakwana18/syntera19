import time
import pytest
from typing import List, Dict, Any

from backend.retrieval.chunking import BaseChunker, FixedTokenChunker, SemanticChunker, StructureAwareChunker

def test_fixed_token_chunker():
    chunker = FixedTokenChunker(chunk_size=10, overlap=2)
    text = "0123456789abcdefghij"
    chunks = chunker.chunk(text, {"source": "test", "chunk_index": 0})
    assert len(chunks) == 3
    assert chunks[0]["text"] == "0123456789"
    assert chunks[1]["text"] == "89abcdefgh"
    assert chunks[2]["text"] == "ghij"
    assert chunks[0]["metadata"]["next_chunk_id"] == chunks[1]["metadata"]["chunk_id"]
    assert chunks[1]["metadata"]["previous_chunk_id"] == chunks[0]["metadata"]["chunk_id"]

def test_semantic_chunker():
    chunker = SemanticChunker(chunk_size=30, overlap=0)
    text = "Hello world. This is a test. Short! Extra."
    chunks = chunker.chunk(text, {"source": "test"})
    assert chunks[0]["text"] == "Hello world. This is a test."
    assert chunks[1]["text"] == "Short! Extra."

def test_structure_aware_chunker():
    chunker = StructureAwareChunker(chunk_size=50, overlap=0)
    text = "# Heading 1\nSome text.\n## Heading 2\nMore text."
    chunks = chunker.chunk(text, {"source": "test"})
    assert len(chunks) > 0
    
    # We should have section metadata preserved
    has_h1 = any("Heading 1" in c["metadata"].get("section_path", "") for c in chunks)
    assert has_h1

# Simplified Benchmark
def simple_retrieval(query: str, chunks: List[Dict[str, Any]], k: int = 3):
    # Dummy retrieval using word overlap
    query_words = set(query.lower().split())
    scored = []
    for i, chunk in enumerate(chunks):
        chunk_words = set(chunk["text"].lower().split())
        score = len(query_words.intersection(chunk_words))
        scored.append((score, i, chunk))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c[2] for c in scored[:k]]

@pytest.mark.benchmark
def test_chunking_benchmark():
    document_text = """
# Advanced Chunking
Chunking is a critical step in RAG systems. It splits large documents into smaller pieces.

## Fixed Chunking
Fixed chunking uses a character limit. It is fast but can break semantics.

## Semantic Chunking
Semantic chunking uses natural boundaries like sentences. This improves coherence.

## Structure Aware
Structure aware chunking looks at markdown headings. It preserves the hierarchy of the document.
""" * 10  # Make it a bit longer

    eval_queries = [
        {"query": "What is fixed chunking?", "expected_in_text": "character limit"},
        {"query": "What does semantic chunking use?", "expected_in_text": "natural boundaries"},
        {"query": "How does structure aware chunking work?", "expected_in_text": "markdown headings"}
    ]

    strategies = {
        "Fixed": FixedTokenChunker(chunk_size=50, overlap=10),
        "Semantic": SemanticChunker(chunk_size=50, overlap=0),
        "Structure": StructureAwareChunker(chunk_size=50, overlap=10)
    }

    results = {}

    for name, chunker in strategies.items():
        start_time = time.time()
        chunks = chunker.chunk(document_text, {"source": "benchmark"})
        latency = time.time() - start_time
        
        chunk_count = len(chunks)
        avg_context_size = sum(len(c["text"]) for c in chunks) / max(chunk_count, 1)

        # Evaluate Retrieval
        recall_at_3 = 0
        mrr = 0.0
        precision_at_3 = 0
        
        for q in eval_queries:
            retrieved = simple_retrieval(q["query"], chunks, k=3)
            
            hit_rank = -1
            for rank, c in enumerate(retrieved):
                if q["expected_in_text"].lower() in c["text"].lower():
                    hit_rank = rank + 1
                    break
            
            if hit_rank != -1:
                recall_at_3 += 1
                mrr += 1.0 / hit_rank
                precision_at_3 += 1.0 / 3.0 # Simplified precision 

        results[name] = {
            "Recall@3": recall_at_3 / len(eval_queries),
            "MRR": mrr / len(eval_queries),
            "Precision": precision_at_3 / len(eval_queries),
            "Latency": latency,
            "ChunkCount": chunk_count,
            "AvgContextSize": avg_context_size
        }

    # Print or log the results
    print("\n--- Chunking Benchmark Results ---")
    for name, metrics in results.items():
        print(f"Strategy: {name}")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    
    # Assert benchmark ran successfully
    assert len(results) == 3
