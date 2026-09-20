import fitz  # PyMuPDF
import os
import re
import uuid
from backend.retrieval.chunking import StructureAwareChunker, BaseChunker

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

def ingest_document(file_path: str, filename: str, chunker: BaseChunker = None):
    from vectorstore.qdrant_client import vector_store
    
    if chunker is None:
        chunker = StructureAwareChunker(chunk_size=1000, overlap=200)
        
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
            
            base_metadata = {
                "source": filename,
                "page": page_num,
                "type": "document",
                "block_type": "text",
                "bbox": current_chunk_bbox,
                "chunk_index": global_chunk_index,
                "heading_stack": current_heading_stack
            }
            
            chunk_results = chunker.chunk(current_chunk_text, base_metadata)
            
            for res in chunk_results:
                all_chunks.append(res["text"])
                # We don't want to store heading_stack in the final metadata as it's an internal list
                meta = res["metadata"]
                if "heading_stack" in meta:
                    del meta["heading_stack"]
                all_metadatas.append(meta)
                global_chunk_index += 1
                
            current_heading_stack = base_metadata.get("heading_stack", current_heading_stack)
            current_chunk_text = ""
            current_chunk_bbox = None

        for block in page_data["blocks"]:
            if block["type"] == "table":
                flush_text()
                
                # Treat the table as a single intact chunk
                chunk = block["text"]
                base_metadata = {
                    "source": filename,
                    "page": page_num,
                    "type": "document",
                    "block_type": "table",
                    "bbox": block["bbox"],
                    "chunk_index": global_chunk_index,
                    "heading_stack": current_heading_stack
                }
                
                # Use chunker just to format metadata and link, but ensure it doesn't split tables by using a huge limit temporarily
                # Or just manually append it to avoid splitting tables:
                if isinstance(chunker, StructureAwareChunker):
                    current_heading_stack = chunker._update_heading_stack(current_heading_stack, chunk)
                    section_path = chunker._format_section_path(current_heading_stack)
                    section = current_heading_stack[-1]['title'] if current_heading_stack else "Root"
                else:
                    section = "Root"
                    section_path = "Root"
                    
                chunk_id = f"{filename}_chunk_{global_chunk_index}"
                all_chunks.append(chunk)
                meta = base_metadata.copy()
                meta.update({
                    "section": section,
                    "section_path": section_path,
                    "chunk_id": chunk_id,
                    "previous_chunk_id": f"{filename}_chunk_{global_chunk_index-1}" if global_chunk_index > 0 else None,
                    "next_chunk_id": f"{filename}_chunk_{global_chunk_index+1}"
                })
                if "heading_stack" in meta:
                    del meta["heading_stack"]
                    
                all_metadatas.append(meta)
                global_chunk_index += 1
            else:
                # Accumulate text
                if current_chunk_text:
                    current_chunk_text += "\n" + block["text"]
                else:
                    current_chunk_text = block["text"]
                    current_chunk_bbox = block["bbox"]
                    
        flush_text()
            
    # Fix the last chunk's next_chunk_id
    if all_metadatas:
        all_metadatas[-1]["next_chunk_id"] = None

    if all_chunks:
        doc_ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, meta["chunk_id"])) for meta in all_metadatas]
        vector_store.add_texts(all_chunks, all_metadatas, ids=doc_ids)
        from vectorstore.bm25_store import bm25_store
        bm25_store.add_texts(all_chunks, all_metadatas, doc_ids)
        
    return len(all_chunks)
