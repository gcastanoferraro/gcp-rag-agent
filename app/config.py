from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    groq_api_key: str
    groq_model: str = "llama-3.1-70b-versatile"

    # Embeddings
    voyage_api_key: str
    voyage_model: str = "voyage-3"

    # Weaviate
    weaviate_url: str = "http://localhost:8080"
    weaviate_collection: str = "GCPDocs"

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # RAG params
    chunk_size: int = 1500
    chunk_overlap: int = 150
    retrieval_top_k: int = 5

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()