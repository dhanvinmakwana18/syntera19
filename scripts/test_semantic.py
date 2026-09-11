import re

def semantic_chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    # Separators from largest semantic boundary to smallest
    separators = [
        "\n# ",
        "\n## ",
        "\n### ",
        "\n#### ",
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
    
    def split_with_separator(text_to_split, sep):
        if sep == "":
            return list(text_to_split)
        
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
        if len(text_to_split) <= chunk_size:
            return [text_to_split]
            
        if current_sep_index >= len(separators):
            return [text_to_split[i:i+chunk_size] for i in range(0, len(text_to_split), chunk_size - overlap)]
            
        sep = separators[current_sep_index]
        splits = split_with_separator(text_to_split, sep)
        
        if len(splits) == 1:
            return recursive_split(text_to_split, current_sep_index + 1)
            
        merged = []
        current_chunk = ""
        
        for s in splits:
            if len(current_chunk) + len(s) <= chunk_size:
                current_chunk += s
            else:
                if current_chunk:
                    merged.append(current_chunk)
                if len(s) > chunk_size:
                    merged.extend(recursive_split(s, current_sep_index + 1))
                    current_chunk = ""
                else:
                    current_chunk = s
                    
        if current_chunk:
            merged.append(current_chunk)
            
        return merged

    chunks = recursive_split(text, 0)
    
    if overlap > 0:
        overlapped_chunks = []
        for i, c in enumerate(chunks):
            if i > 0 and len(chunks[i-1]) > overlap:
                prefix = chunks[i-1][-overlap:]
                space_idx = prefix.find(" ")
                if space_idx != -1 and space_idx < len(prefix) // 2:
                    prefix = prefix[space_idx:]
                c = prefix + c
            if len(c) > chunk_size + overlap:
                c = c[:chunk_size + overlap]
            overlapped_chunks.append(c)
        return overlapped_chunks
        
    return chunks

from ingestion.parser import chunk_text as baseline_chunk_text

with open('../data/documents/langchain_readme.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
c_base = baseline_chunk_text(text, 1000, 200)
c_sem = semantic_chunk_text(text, 1000, 200)

print(f"Base chunks: {len(c_base)}")
print(f"Semantic chunks: {len(c_sem)}")
print(f"Avg Base: {sum(len(c) for c in c_base)/len(c_base):.1f}")
print(f"Avg Sem: {sum(len(c) for c in c_sem)/len(c_sem):.1f}")
