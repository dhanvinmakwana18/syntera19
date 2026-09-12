import pytest
from ingestion.parser import update_heading_stack, format_section_path, chunk_text


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
