from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
import shutil
import os
from core.config import settings
from ingestion.parser import ingest_document

kb_router = APIRouter()

class IngestResponse(BaseModel):
    status: str
    filename: str
    chunks: int

@kb_router.post("/upload", response_model=IngestResponse)
async def upload_document(file: UploadFile = File(...)):
    # Save file
    file_path = os.path.join(settings.DOCUMENTS_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Process and ingest
    try:
        num_chunks = ingest_document(file_path, file.filename)
        return IngestResponse(status="success", filename=file.filename, chunks=num_chunks)
    except Exception as e:
        return IngestResponse(status="error", filename=file.filename, chunks=0)
        
@kb_router.get("/documents")
def list_documents():
    docs = []
    if os.path.exists(settings.DOCUMENTS_DIR):
        for filename in os.listdir(settings.DOCUMENTS_DIR):
            path = os.path.join(settings.DOCUMENTS_DIR, filename)
            docs.append({
                "name": filename,
                "size_kb": round(os.path.getsize(path) / 1024, 2)
            })
    return {"documents": docs}
