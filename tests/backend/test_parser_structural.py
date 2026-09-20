import pytest
from backend.retrieval.chunking import StructureAwareChunker

def test_heading_propagation():
    chunker = StructureAwareChunker()
    stack = []
    stack = chunker._update_heading_stack(stack, "# Main Title")
    assert chunker._format_section_path(stack) == "Main Title"
    
    stack = chunker._update_heading_stack(stack, "## Subtitle")
    assert chunker._format_section_path(stack) == "Main Title > Subtitle"
    
    stack = chunker._update_heading_stack(stack, "# New Title")
    assert chunker._format_section_path(stack) == "New Title"

def test_chunk_text_basic():
    chunker = StructureAwareChunker(chunk_size=1000, overlap=200)
    text = "A" * 1500
    chunks = chunker.chunk(text, {})
    assert len(chunks) == 2
    assert len(chunks[0]["text"]) == 1000
    # overlap logic ensures next chunk has previous overlap
    assert len(chunks[1]["text"]) <= 1200 

def test_chunk_text_small():
    chunker = StructureAwareChunker(chunk_size=1000, overlap=200)
    text = "Short text"
    chunks = chunker.chunk(text, {})
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Short text"

import pytest
import os
from backend.ingestion.parser import parse_pdf, ingest_document

def test_parse_pdf_extracts_blocks():
    pdf_path = os.path.join("..", "data", "documents", "NexusLLM_RAG_Upgrade_and_AI-V_AI-D_Postponement_Plan.pdf")
    if not os.path.exists(pdf_path):
        pytest.skip("PDF file not found for testing")
        
    pages = parse_pdf(pdf_path)
    assert len(pages) > 0
    
    # Page 0 has a table
    blocks = pages[0]["blocks"]
    assert len(blocks) > 0
    
    # Check block types
    types = [b["type"] for b in blocks]
    assert "text" in types
    assert "table" in types
    
    # Check table formatting
    table_block = next(b for b in blocks if b["type"] == "table")
    assert "|" in table_block["text"]
    assert "bbox" in table_block

def test_parse_pdf_ordering():
    pdf_path = os.path.join("..", "data", "documents", "NexusLLM_RAG_Upgrade_and_AI-V_AI-D_Postponement_Plan.pdf")
    if not os.path.exists(pdf_path):
        pytest.skip("PDF file not found for testing")
        
    pages = parse_pdf(pdf_path)
    blocks = pages[0]["blocks"]
    
    # Verify vertical Y0 sorting
    for i in range(1, len(blocks)):
        prev_y0 = blocks[i-1]["bbox"][1]
        curr_y0 = blocks[i]["bbox"][1]
        assert curr_y0 >= prev_y0 or abs(curr_y0 - prev_y0) < 50

