import fitz  # PyMuPDF
import os
import re
import uuid

def parse_pdf(file_path: str):
    """Extracts text and page metadata from a PDF file."""
    doc = fitz.open(file_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        tables = page.find_tables()
        table_bboxes = []
        table_blocks = []
        
        if tables and tables.tables:
            for i, tab in enumerate(tables.tables):
                bbox = tab.bbox
                table_bboxes.append(bbox)
                table_blocks.append({
                    "type": "table",
                    "bbox": [round(c, 2) for c in bbox],
                    "text": tab.to_markdown()
                })
                
        blocks = page.get_text("blocks")
        text_blocks = []
        for b in blocks:
            if b[6] == 0:  # Text block
                bbox = (b[0], b[1], b[2], b[3])
                text = b[4].strip()
                if not text: continue
                
                is_in_table = False
                for t_bbox in table_bboxes:
                    # Check if text block overlaps significantly with a table
                    r1 = fitz.Rect(bbox)
                    r2 = fitz.Rect(t_bbox)
                    intersect = r1.intersect(r2)
                    if intersect.get_area() > r1.get_area() * 0.5:
                        is_in_table = True
                        break
                
                if not is_in_table:
                    text_blocks.append({
                        "type": "text",
                        "bbox": [round(c, 2) for c in bbox],
                        "text": text
                    })
                    
        all_blocks = table_blocks + text_blocks
        all_blocks.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
        
        if all_blocks:
            pages.append({"page": page_num + 1, "blocks": all_blocks})
            
    return pages

def parse_text(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return [{"page": 1, "blocks": [{"type": "text", "bbox": None, "text": f.read()}]}]

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    separators = [
        "\n# ", "\n## ", "\n### ", "\n#### ",
        "\n\n", "\n", ". ", " ", ""
    ]
    
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
                if current_chunk: merged.append(current_chunk)
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

def update_heading_stack(current_stack, chunk_text):
    matches = re.finditer(r'(?:^|\n)(#{1,6})\s+(.*)', chunk_text)
    new_stack = list(current_stack)
    for match in matches:
        level = len(match.group(1))
        title = match.group(2).strip()
        new_stack = [h for h in new_stack if h['level'] < level]
        new_stack.append({'level': level, 'title': title})
    return new_stack

def format_section_path(stack):
    if not stack:
        return "Root"
    return " > ".join([h['title'] for h in stack])

def ingest_document(file_path: str, filename: str):
    from vectorstore.qdrant_client import vector_store
    
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        pages = parse_pdf(file_path)
    else:
        pages = parse_text(file_path)
        
    all_chunks = []
    all_metadatas = []
    
    global_chunk_index = 0
    current_heading_stack = []
    
    for page_data in pages:
        page_num = page_data["page"]
        current_chunk_text = ""
        current_chunk_bbox = None
        
        # Helper to flush accumulated text chunks
        def flush_text():
            nonlocal current_chunk_text, current_chunk_bbox, global_chunk_index, current_heading_stack
            if not current_chunk_text:
                return
            
            text_chunks = chunk_text(current_chunk_text)
            for chunk in text_chunks:
                current_heading_stack = update_heading_stack(current_heading_stack, chunk)
                section_path = format_section_path(current_heading_stack)
                section = current_heading_stack[-1]['title'] if current_heading_stack else "Root"
                
                chunk_id = f"{filename}_chunk_{global_chunk_index}"
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source": filename,
                    "page": page_num,
                    "section": section,
                    "section_path": section_path,
                    "chunk_id": chunk_id,
                    "chunk_index": global_chunk_index,
                    "type": "document",
                    "block_type": "text",
                    "bbox": current_chunk_bbox
                })
                global_chunk_index += 1
            current_chunk_text = ""
            current_chunk_bbox = None

        for block in page_data["blocks"]:
            if block["type"] == "table":
                flush_text()
                
                # Treat the table as a single intact chunk
                chunk = block["text"]
                current_heading_stack = update_heading_stack(current_heading_stack, chunk)
                section_path = format_section_path(current_heading_stack)
                section = current_heading_stack[-1]['title'] if current_heading_stack else "Root"
                
                chunk_id = f"{filename}_chunk_{global_chunk_index}"
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source": filename,
                    "page": page_num,
                    "section": section,
                    "section_path": section_path,
                    "chunk_id": chunk_id,
                    "chunk_index": global_chunk_index,
                    "type": "document",
                    "block_type": "table",
                    "bbox": block["bbox"]
                })
                global_chunk_index += 1
            else:
                # Accumulate text
                if current_chunk_text:
                    current_chunk_text += "\n" + block["text"]
                else:
                    current_chunk_text = block["text"]
                    current_chunk_bbox = block["bbox"]
                    
        flush_text()
            
    if all_chunks:
        doc_ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, meta["chunk_id"])) for meta in all_metadatas]
        vector_store.add_texts(all_chunks, all_metadatas, ids=doc_ids)
        from vectorstore.bm25_store import bm25_store
        if not bm25_store._is_synced:
            bm25_store.sync_from_qdrant(vector_store)
        else:
            bm25_store.add_texts(all_chunks, all_metadatas, doc_ids)
        
    return len(all_chunks)
