import pytest
from backend.retrieval.chunking import StructureAwareChunker


def test_heading_stack():
    chunker = StructureAwareChunker()
    stack = []
    stack = chunker._update_heading_stack(stack, "# Chapter 1\nSome text")
    assert chunker._format_section_path(stack) == "Chapter 1"
    
    stack = chunker._update_heading_stack(stack, "## 1.1 Intro\nMore text")
    assert chunker._format_section_path(stack) == "Chapter 1 > 1.1 Intro"
    
    stack = chunker._update_heading_stack(stack, "### Details\nHello")
    assert chunker._format_section_path(stack) == "Chapter 1 > 1.1 Intro > Details"
    
    stack = chunker._update_heading_stack(stack, "## 1.2 Outro\nBye")
    assert chunker._format_section_path(stack) == "Chapter 1 > 1.2 Outro"
    
    stack = chunker._update_heading_stack(stack, "# Chapter 2\nEnd")
    assert chunker._format_section_path(stack) == "Chapter 2"
