import pytest
from datetime import datetime, timedelta
from memory.contracts import MemoryType, MemoryRecord
from memory.store import InMemoryDictStore
from memory.manager import MemoryManager
from retrieval.memory_retriever import MemoryRetriever
from core.domain import Query

def test_memory_lifecycle():
    store = InMemoryDictStore()
    manager = MemoryManager(store)
    
    # 1. Observation
    record = manager.observe("User prefers Python", source="user_input")
    assert record.type == MemoryType.WORKING
    assert record.content == "User prefers Python"
    
    # 2. Validation
    assert manager.validate(record) is True
    invalid_record = manager.observe("   ")
    assert manager.validate(invalid_record) is False
    
    # 3. Storage & Deduplication
    manager.store(record)
    assert len(store.get_all()) == 1
    
    # Try duplicate
    dup_record = manager.observe("User prefers Python", source="user_input")
    manager.store(dup_record)
    assert len(store.get_all()) == 1  # Deduplicated
    
    # 4. Retrieval
    results = manager.retrieve("Python")
    assert len(results) == 1
    assert results[0].content == "User prefers Python"
    
    # 5. Consolidation
    manager.consolidate()
    promoted = store.get(record.id)
    assert promoted.type == MemoryType.EPISODIC
    
    # 6. Forgetting
    # Set expiration to the past
    promoted.expiration = datetime.utcnow() - timedelta(days=1)
    store.update(promoted)
    
    manager.forget()
    assert len(store.get_all()) == 0

def test_memory_retriever():
    store = InMemoryDictStore()
    manager = MemoryManager(store)
    
    record1 = manager.observe("Learning AI", source="chat")
    record2 = manager.observe("Building a memory system", source="chat")
    manager.store(record1)
    manager.store(record2)
    
    retriever = MemoryRetriever(store)
    query = Query(text="memory")
    results = retriever.retrieve(query)
    
    assert len(results) >= 1
    
    # Check if the result has correctly populated Node fields
    retrieved_text = [r.node.text for r in results]
    assert "Building a memory system" in retrieved_text
    
    # Check metadata
    node = results[0].node
    assert "type" in node.metadata
    assert node.metadata["type"] == MemoryType.WORKING.value
