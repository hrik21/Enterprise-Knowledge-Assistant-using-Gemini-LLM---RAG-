"""
Configuration management for RAG Enterprise Assistant.
Handles environment-specific settings and validation.
"""

from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
import os
from pathlib import Path


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    name: str = Field(default="rag_assistant", env="DB_NAME")
    user: str = Field(default="postgres", env="DB_USER")
    password: str = Field(env="DB_PASSWORD")
    
    @property
    def url(self) -> str:
        """Generate database URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class VectorStoreSettings(BaseSettings):
    """FAISS vector store configuration."""
    
    index_type: str = Field(default="HNSW", env="FAISS_INDEX_TYPE")
    embedding_dimension: int = Field(default=768, env="EMBEDDING_DIMENSION")
    distance_metric: str = Field(default="cosine", env="DISTANCE_METRIC")
    index_path: str = Field(default="./data/faiss_index", env="FAISS_INDEX_PATH")
    
    @field_validator("index_type")
    @classmethod
    def validate_index_type(cls, v):
        allowed_types = ["HNSW", "IVF", "Flat"]
        if v not in allowed_types:
            raise ValueError(f"Index type must be one of {allowed_types}")
        return v


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""
    
    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    batch_size: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")
    max_length: int = Field(default=512, env="EMBEDDING_MAX_LENGTH")


class LLMSettings(BaseSettings):
    """Language model configuration."""
    
    provider: str = Field(default="gemini", env="LLM_PROVIDER")
    api_key: str = Field(env="GEMINI_API_KEY")
    model_name: str = Field(default="gemini-pro", env="GEMINI_MODEL")
    max_tokens: int = Field(default=1024, env="LLM_MAX_TOKENS")
    temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")
    
    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v


class DocumentProcessingSettings(BaseSettings):
    """Document processing configuration."""
    
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    supported_formats: List[str] = Field(default=["pdf", "txt", "docx"], env="SUPPORTED_FORMATS")
    input_directory: str = Field(default="./data/input", env="INPUT_DIRECTORY")
    processed_directory: str = Field(default="./data/processed", env="PROCESSED_DIRECTORY")
    
    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v, info):
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        return v


class APISettings(BaseSettings):
    """API server configuration."""
    
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    workers: int = Field(default=1, env="API_WORKERS")
    reload: bool = Field(default=False, env="API_RELOAD")
    log_level: str = Field(default="info", env="LOG_LEVEL")
    
    # Security settings
    secret_key: str = Field(env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")


class AirflowSettings(BaseSettings):
    """Apache Airflow configuration."""
    
    dags_folder: str = Field(default="./airflow/dags", env="AIRFLOW_DAGS_FOLDER")
    executor: str = Field(default="LocalExecutor", env="AIRFLOW_EXECUTOR")
    schedule_interval: str = Field(default="@hourly", env="INGESTION_SCHEDULE")
    max_active_runs: int = Field(default=1, env="AIRFLOW_MAX_ACTIVE_RUNS")


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration."""
    
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    correlation_id_header: str = Field(default="X-Correlation-ID", env="CORRELATION_ID_HEADER")


class Settings(BaseSettings):
    """Main application settings."""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Component settings
    database: DatabaseSettings = DatabaseSettings()
    vector_store: VectorStoreSettings = VectorStoreSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    llm: LLMSettings = LLMSettings()
    document_processing: DocumentProcessingSettings = DocumentProcessingSettings()
    api: APISettings = APISettings()
    airflow: AirflowSettings = AirflowSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.document_processing.input_directory,
            self.document_processing.processed_directory,
            os.path.dirname(self.vector_store.index_path),
            self.airflow.dags_folder,
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        allowed_envs = ["development", "staging", "production"]
        if v not in allowed_envs:
            raise ValueError(f"Environment must be one of {allowed_envs}")
        return v


# Global settings instance
settings = Settings()