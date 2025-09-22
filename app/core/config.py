from typing import List, Dict
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Mistral API Configuration
    mistral_api_key: str = Field(..., description="Mistral API key")
    mistral_base_url: str = Field(default="https://api.mistral.ai", description="Mistral API base URL")
    mistral_chat_model: str = Field(default="mistral-large-latest", description="Mistral chat model")
    mistral_embed_model: str = Field(default="mistral-embed", description="Mistral embedding model")
    
    # CORS Configuration
    frontend_origin: List[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"], 
        description="Allowed frontend origins for CORS"
    )
    
    # RAG Configuration
    top_k: int = Field(default=5, description="Number of top chunks to retrieve")
    chunk_size: int = Field(default=800, description="Target chunk size in tokens")
    chunk_overlap: int = Field(default=150, description="Chunk overlap in tokens")
    
    # Source prioritization
    source_priority_boost: Dict[str, float] = Field(
        default={
            "experience.json": 1.3,
            "projects.json": 1.1,
            "about.md": 1.05,
            "education.json": 1.0,
            "extracurricular.json": 0.95,
            "projects_social.json": 0.9
        }, 
        description="Multiplier to boost similarity scores by source file"
    )
    
    # Storage paths
    faiss_index_path: str = Field(default="storage/faiss.index", description="Path to FAISS index file")
    chunks_path: str = Field(default="storage/chunks.jsonl", description="Path to chunks metadata file")
    knowledge_dir: str = Field(default="knowledge", description="Directory containing knowledge files")
    
    # API Configuration
    max_tokens: int = Field(default=1024, description="Maximum tokens for chat completion")
    temperature: float = Field(default=0.2, description="Temperature for chat completion")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()