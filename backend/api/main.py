from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from core.config import settings
import time

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    description="Autonomous Agentic RAG & Multi-Modal AI Engine API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Structured error categories
ERROR_CODES = {
    "BACKEND_OFFLINE": "The backend engine is not reachable.",
    "INVALID_REQUEST": "The request payload is invalid.",
    "ROUTING_FAILURE": "Failed to classify the query intent.",
    "RETRIEVAL_FAILURE": "Failed to retrieve documents from the vector store.",
    "VECTOR_STORE_FAILURE": "The vector database is unavailable.",
    "EMBEDDING_FAILURE": "Failed to generate embeddings.",
    "RERANKING_FAILURE": "The reranking stage failed.",
    "MODEL_FAILURE": "The LLM provider returned an error.",
    "GROUNDING_FAILURE": "Failed to verify answer grounding.",
}

# Global exception handler with structured errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_type = "INTERNAL_ERROR"
    error_msg = str(exc)
    
    # Classify known error patterns
    err_lower = error_msg.lower()
    if "connection refused" in err_lower or "connection error" in err_lower:
        error_type = "MODEL_FAILURE"
    elif "qdrant" in err_lower or "vector" in err_lower:
        error_type = "VECTOR_STORE_FAILURE"
    elif "embedding" in err_lower:
        error_type = "EMBEDDING_FAILURE"
        
    return JSONResponse(
        status_code=500,
        content={
            "error_code": error_type,
            "detail": ERROR_CODES.get(error_type, "An unexpected error occurred."),
            "message": error_msg,
        },
    )

from api.router import api_router
from api.kb_router import kb_router
from api.multimodal_router import mm_router

app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(kb_router, prefix=f"{settings.API_V1_STR}/kb")
app.include_router(mm_router, prefix=settings.API_V1_STR)

@app.get("/health")
def health_check():
    """Comprehensive health check for Syntera engine."""
    status = "ONLINE"
    components = {}
    
    # Check LLM provider
    try:
        from providers.llm import llm_provider
        if llm_provider.gemini_api_key:
            components["llm"] = {"status": "ok", "provider": "gemini"}
        else:
            components["llm"] = {"status": "ok", "provider": "ollama (local)"}
    except Exception as e:
        components["llm"] = {"status": "error", "detail": str(e)}
        status = "DEGRADED"
    
    # Check vector store
    try:
        from vectorstore.qdrant_client import vector_store
        info = vector_store.client.get_collection(vector_store.collection_name)
        count = getattr(info, 'points_count', getattr(info, 'vectors_count', 0))
        components["vector_store"] = {
            "status": "ok",
            "collection": vector_store.collection_name,
            "points_count": count
        }
    except Exception as e:
        components["vector_store"] = {"status": "error", "detail": str(e)}
        status = "DEGRADED"
    
    # Check embedding model
    try:
        from providers.embeddings import embedding_provider
        components["embeddings"] = {
            "status": "ok",
            "model": settings.EMBEDDING_MODEL,
            "vector_size": embedding_provider.vector_size
        }
    except Exception as e:
        components["embeddings"] = {"status": "error", "detail": str(e)}
        status = "DEGRADED"
    
    # Check reranker
    try:
        from retrieval.reranker import reranker_service
        if reranker_service.model is not None:
            components["reranker"] = {"status": "ok", "model": reranker_service.model_name}
        else:
            components["reranker"] = {"status": "degraded", "detail": "Model not loaded, fallback active"}
            if status == "ONLINE":
                status = "DEGRADED"
    except Exception as e:
        components["reranker"] = {"status": "error", "detail": str(e)}
    
    return {
        "status": status,
        "service": settings.PROJECT_NAME,
        "version": "2.0.0",
        "components": components
    }
