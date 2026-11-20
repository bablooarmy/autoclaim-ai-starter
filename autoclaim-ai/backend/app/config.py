from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AutoClaim AI"
    app_env: str = "development"
    api_prefix: str = "/api/v1"

    # Local Ollama endpoints & models
    ollama_base_url: str = "http://localhost:11434"
    ollama_vision_model: str = "qwen2.5vl:7b"
    ollama_text_model: str = "qwen2.5:7b"
    ollama_embed_model: str = "nomic-embed-text"

    # Local Qdrant Vector DB
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "insurance_policies"
    qdrant_in_memory: bool = False

    # Database (SQLite default for zero-config local run, PostgreSQL in Docker Compose)
    database_url: str = "sqlite:///./autoclaim.db"

    # Governance & Evaluation thresholds
    min_confidence_for_auto_approve: float = 0.75
    max_auto_approve_estimate_usd: float = 3500.0
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]


settings = Settings()
