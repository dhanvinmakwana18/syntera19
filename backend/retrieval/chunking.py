import re
import uuid
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod

class BaseChunker(ABC):
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    @abstractmethod
    def chunk(self, text: str, base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Splits text into chunks and attaches metadata.
        Returns a list of dicts with 'text' and 'metadata'.
        """
        pass

    def _link_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Adds previous and next chunk IDs to preserve relationships."""
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk['metadata']['previous_chunk_id'] = chunks[i-1]['metadata'].get('chunk_id')
            else:
                chunk['metadata']['previous_chunk_id'] = None
                
            if i < len(chunks) - 1:
                chunk['metadata']['next_chunk_id'] = chunks[i+1]['metadata'].get('chunk_id')
            else:
                chunk['metadata']['next_chunk_id'] = None
        return chunks

class FixedTokenChunker(BaseChunker):
    """Chunks text strictly by character length with a sliding window."""
    def chunk(self, text: str, base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not text:
            return []
            
        chunks_data = []
        start = 0
        text_len = len(text)
        chunk_idx = base_metadata.get('chunk_index', 0)
        source = base_metadata.get('source', 'doc')

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end]
            
            meta = base_metadata.copy()
            meta['chunk_id'] = f"{source}_chunk_{chunk_idx}"
            meta['chunk_index'] = chunk_idx
            
            chunks_data.append({
                'text': chunk_text,
                'metadata': meta
            })
            
            chunk_idx += 1
            if self.chunk_size <= self.overlap:
                start += self.chunk_size
            else:
                start += (self.chunk_size - self.overlap)
            
        return self._link_chunks(chunks_data)

class SemanticChunker(BaseChunker):
    """Chunks text by sentence boundaries to preserve semantic meaning."""
    def chunk(self, text: str, base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not text:
            return []
            
        # Basic sentence splitting heuristic
        sentences = re.split(r'(?<=[.?!])\s+', text)
        
        chunks_data = []
        current_chunk = ""
        chunk_idx = base_metadata.get('chunk_index', 0)
        source = base_metadata.get('source', 'doc')
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                current_chunk += (" " + sentence) if current_chunk else sentence
            else:
                if current_chunk:
                    meta = base_metadata.copy()
                    meta['chunk_id'] = f"{source}_chunk_{chunk_idx}"
                    meta['chunk_index'] = chunk_idx
                    chunks_data.append({'text': current_chunk.strip(), 'metadata': meta})
                    chunk_idx += 1
                
                # Handle oversized sentences
                if len(sentence) > self.chunk_size:
                    # Fallback to fixed size for this giant sentence
                    sub_chunks = [sentence[i:i+self.chunk_size] for i in range(0, len(sentence), self.chunk_size)]
                    for sc in sub_chunks:
                        meta = base_metadata.copy()
                        meta['chunk_id'] = f"{source}_chunk_{chunk_idx}"
                        meta['chunk_index'] = chunk_idx
                        chunks_data.append({'text': sc, 'metadata': meta})
                        chunk_idx += 1
                    current_chunk = ""
                else:
                    current_chunk = sentence
                    
        if current_chunk:
            meta = base_metadata.copy()
            meta['chunk_id'] = f"{source}_chunk_{chunk_idx}"
            meta['chunk_index'] = chunk_idx
            chunks_data.append({'text': current_chunk.strip(), 'metadata': meta})
            
        return self._link_chunks(chunks_data)

class StructureAwareChunker(BaseChunker):
    """Chunks text preserving Markdown headings and document hierarchy."""
    
    def _update_heading_stack(self, current_stack: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        matches = re.finditer(r'(?:^|\n)(#{1,6})\s+(.*)', text)
        new_stack = list(current_stack)
        for match in matches:
            level = len(match.group(1))
            title = match.group(2).strip()
            new_stack = [h for h in new_stack if h['level'] < level]
            new_stack.append({'level': level, 'title': title})
        return new_stack
        
    def _format_section_path(self, stack: List[Dict[str, Any]]) -> str:
        if not stack:
            return "Root"
        return " > ".join([h['title'] for h in stack])
        
    def chunk(self, text: str, base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not text:
            return []
            
        # First use a separator-based split
        separators = ["\n# ", "\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""]
        
        def split_with_separator(text_to_split, sep):
            if sep == "": return list(text_to_split)
            parts = text_to_split.split(sep)
            result = []
            for i, part in enumerate(parts):
                if i > 0 and sep.startswith("\n"): 
                    result.append(sep + part)
                elif i < len(parts) - 1 and not sep.startswith("\n"):
                    result.append(part + sep)
                else:
                    result.append(part)
            return [r for r in result if r]

        def recursive_split(text_to_split, current_sep_index):
            if len(text_to_split) <= self.chunk_size:
                return [text_to_split]
            if current_sep_index >= len(separators):
                step = max(1, self.chunk_size - self.overlap)
                return [text_to_split[i:i+self.chunk_size] for i in range(0, len(text_to_split), step)]
                
            sep = separators[current_sep_index]
            splits = split_with_separator(text_to_split, sep)
            
            if len(splits) == 1:
                return recursive_split(text_to_split, current_sep_index + 1)
                
            merged = []
            current_chunk = ""
            for s in splits:
                if len(current_chunk) + len(s) <= self.chunk_size:
                    current_chunk += s
                else:
                    if current_chunk: merged.append(current_chunk)
                    if len(s) > self.chunk_size:
                        merged.extend(recursive_split(s, current_sep_index + 1))
                        current_chunk = ""
                    else:
                        current_chunk = s
            if current_chunk:
                merged.append(current_chunk)
            return merged

        raw_chunks = recursive_split(text, 0)
        
        # Apply overlap
        overlapped_chunks = []
        if self.overlap > 0:
            for i, c in enumerate(raw_chunks):
                if i > 0 and len(raw_chunks[i-1]) > self.overlap:
                    prefix = raw_chunks[i-1][-self.overlap:]
                    space_idx = prefix.find(" ")
                    if space_idx != -1 and space_idx < len(prefix) // 2:
                        prefix = prefix[space_idx:]
                    c = prefix + c
                if len(c) > self.chunk_size + self.overlap:
                    c = c[:self.chunk_size + self.overlap]
                overlapped_chunks.append(c)
        else:
            overlapped_chunks = raw_chunks
            
        chunks_data = []
        chunk_idx = base_metadata.get('chunk_index', 0)
        source = base_metadata.get('source', 'doc')
        current_heading_stack = base_metadata.get('heading_stack', [])
        
        for c in overlapped_chunks:
            current_heading_stack = self._update_heading_stack(current_heading_stack, c)
            section_path = self._format_section_path(current_heading_stack)
            section = current_heading_stack[-1]['title'] if current_heading_stack else "Root"
            
            meta = base_metadata.copy()
            meta['chunk_id'] = f"{source}_chunk_{chunk_idx}"
            meta['chunk_index'] = chunk_idx
            meta['section'] = section
            meta['section_path'] = section_path
            
            chunks_data.append({'text': c, 'metadata': meta})
            chunk_idx += 1
            
        # Write back the final heading stack to base_metadata if needed
        base_metadata['heading_stack'] = current_heading_stack
        
        return self._link_chunks(chunks_data)
