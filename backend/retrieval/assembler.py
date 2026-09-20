def expand_and_organize_context(candidates, expand_neighbors=False):
    from vectorstore.qdrant_client import vector_store
    import uuid
    
    selected = []
    seen_texts = set()
    for c in candidates:
        text = c.get("payload", {}).get("text", "")
        if text and text not in seen_texts:
            seen_texts.add(text)
            selected.append(c)
            
    expanded_chunks = []
    if expand_neighbors:
        ids_to_fetch = set()
        existing_ids = {str(c.get("id")) for c in selected if c.get("id")}
        
        for c in selected:
            payload = c.get("payload", {})
            source = payload.get("source")
            chunk_index = payload.get("chunk_index")
            
            if source and chunk_index is not None:
                prev_id = f"{source}_chunk_{chunk_index - 1}"
                next_id = f"{source}_chunk_{chunk_index + 1}"
                
                prev_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, prev_id))
                next_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, next_id))
                
                if prev_uuid not in existing_ids: ids_to_fetch.add(prev_uuid)
                if next_uuid not in existing_ids: ids_to_fetch.add(next_uuid)
                
        if ids_to_fetch:
            extra_points = vector_store.get_points_by_ids(list(ids_to_fetch))
            for ep in extra_points:
                text = ep.get("payload", {}).get("text", "")
                # Only add if it belongs to a valid section in the same document
                # (Qdrant filters strictly by ID, so it's guaranteed to be the right doc if it exists)
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    ep["is_expanded"] = True
                    ep["score"] = 0.0
                    expanded_chunks.append(ep)
                    
    all_chunks = selected + expanded_chunks
    
    grouped = {}
    for c in all_chunks:
        payload = c.get("payload", {})
        source = payload.get("source", "Unknown")
        chunk_index = payload.get("chunk_index", 999999)
        
        if source not in grouped:
            grouped[source] = []
        grouped[source].append({"chunk": c, "index": chunk_index})
        
    for source in grouped:
        grouped[source] = sorted(grouped[source], key=lambda x: x["index"])
        
    context = ""
    sources = []
    
    def max_score(group):
        return max(item["chunk"].get("rerank_score", item["chunk"].get("score", 0.0)) for item in group)
        
    sorted_sources = sorted(grouped.keys(), key=lambda s: max_score(grouped[s]), reverse=True)
    
    for source in sorted_sources:
        group = grouped[source]
        
        for item in group:
            c = item["chunk"]
            payload = c.get("payload", {})
            text = payload.get("text", "")
            page = payload.get("page", "?")
            section_path = payload.get("section_path", "Root")
            chunk_index = payload.get("chunk_index", "?")
            score = c.get("rerank_score", c.get("score", 0.0))
            is_expanded = c.get("is_expanded", False)
            
            source_idx = len(sources) + 1
            
            sources.append({
                "id": source_idx,
                "filename": source,
                "page": page,
                "section": section_path,
                "chunk_index": chunk_index,
                "text": text,
                "score": score,
                "is_expanded": is_expanded, "block_type": payload.get("block_type", "text"), "bbox": payload.get("bbox", None)
            })
            
            context += f"[Source {source_idx}] (File: {source}, Section: {section_path}, Page: {page}):\n{text}\n\n"
            
    return context.strip(), sources

def assemble_context(candidates, relevance_threshold=None, expand_neighbors=False):
    valid_candidates = []
    for c in candidates:
        score = c.get("rerank_score", c.get("score", 0.0))
        if relevance_threshold is None or score >= relevance_threshold:
            valid_candidates.append(c)
            
    return expand_and_organize_context(valid_candidates, expand_neighbors)

