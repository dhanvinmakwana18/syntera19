from typing import List, Tuple, Dict, Any
from core.domain import RetrievalResult

class ContextBuilder:
    """Builds formatted string context from RetrievalResults without performing database I/O."""
    
    def build(self, results: List[RetrievalResult], relevance_threshold: float = None) -> Tuple[str, List[Dict[str, Any]]]:
        valid_results = []
        for r in results:
            if relevance_threshold is None or r.score >= relevance_threshold:
                valid_results.append(r)
                
        # Group by source
        grouped: Dict[str, List[RetrievalResult]] = {}
        for r in valid_results:
            source = r.node.metadata.get("source", "Unknown")
            if source not in grouped:
                grouped[source] = []
            grouped[source].append(r)
            
        # Sort each group by chunk_index
        for source in grouped:
            grouped[source] = sorted(grouped[source], key=lambda x: x.node.metadata.get("chunk_index", 999999))
            
        # Sort groups by max score
        def max_score(group: List[RetrievalResult]) -> float:
            return max(item.score for item in group)
            
        sorted_sources = sorted(grouped.keys(), key=lambda s: max_score(grouped[s]), reverse=True)
        
        context = ""
        sources = []
        
        for source in sorted_sources:
            group = grouped[source]
            for r in group:
                node = r.node
                text = node.text
                page = node.metadata.get("page", "?")
                section_path = node.metadata.get("section_path", "Root")
                chunk_index = node.metadata.get("chunk_index", "?")
                is_expanded = node.metadata.get("is_expanded", False)
                
                source_idx = len(sources) + 1
                
                sources.append({
                    "id": source_idx,
                    "filename": source,
                    "page": page,
                    "section": section_path,
                    "chunk_index": chunk_index,
                    "text": text,
                    "score": r.score,
                    "is_expanded": is_expanded, 
                    "block_type": node.metadata.get("block_type", "text"), 
                    "bbox": node.metadata.get("bbox", None)
                })
                
                context += f"[Source {source_idx}] (File: {source}, Section: {section_path}, Page: {page}):\n{text}\n\n"
                
        return context.strip(), sources
