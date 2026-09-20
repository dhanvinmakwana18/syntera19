import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "Syntera"
    API_V1_STR: str = "/api/v1"
    
    TESTING: bool = os.getenv("TESTING", "False").lower() in ("true", "1", "yes", "t")
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(os.path.dirname(BASE_DIR), "test_data" if TESTING else "data")
    QDRANT_STORAGE_PATH: str = ":memory:" if TESTING else os.path.join(DATA_DIR, "qdrant")
    SQLITE_DB_PATH: str = os.path.join(DATA_DIR, "sqlite", "nexus.db")
    DOCUMENTS_DIR: str = os.path.join(DATA_DIR, "documents")
    
    # LLM Provider Abstraction
    LLM_PROVIDER: str = "ollama" # ollama, gemini, astra, openai
    LLM_BASE_URL: str = ""
    LLM_MODEL: str = "llama3" # Default local model for ollama
    
    OPENAI_API_KEY: Optional[str] = None
    ASTRA_API_KEY: Optional[str] = None
    ASTRA_MODEL: str = "gpt-6-astra"
    ASTRA_TEMPERATURE: float = 0.2
    ASTRA_MAX_RETRIES: int = 3
    
    # Retrieval configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    QDRANT_COLLECTION_NAME: str = "nexus_knowledge"
    MAX_UPLOAD_SIZE_MB: int = 50
    
    # RAG Tuning Parameters (Configurable)
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    RETRIEVAL_K: int = 5
    DENSE_WEIGHT: float = 1.0
    SPARSE_WEIGHT: float = 1.0
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

# Ensure directories exist
if settings.QDRANT_STORAGE_PATH != ":memory:":
    os.makedirs(settings.QDRANT_STORAGE_PATH, exist_ok=True)
os.makedirs(os.path.dirname(settings.SQLITE_DB_PATH), exist_ok=True)
os.makedirs(settings.DOCUMENTS_DIR, exist_ok=True)
