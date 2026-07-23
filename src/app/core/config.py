from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings, loaded from environment variables and .env file."""
    
    qdrant_url: str = Field(default="http://localhost:6333")
    collection_name: str = "kairos_knowledge"
    groq_api_key: Optional[str] = Field(default=None)
    groq_model: str = "openai/gpt-oss-20b"
    top_k: int = 5
    temperature: float = 0.2
    a2a_api_key: Optional[str] = Field(default=None)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
