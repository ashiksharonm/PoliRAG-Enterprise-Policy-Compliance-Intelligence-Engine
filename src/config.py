"""Configuration management for PoliRAG."""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Configuration
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8001, alias="API_PORT")
    api_workers: int = Field(default=4, alias="API_WORKERS")
    environment: str = Field(default="development", alias="ENVIRONMENT")

    # OpenAI Configuration
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_embedding_model: str = Field(
        default="text-embedding-3-large", alias="OPENAI_EMBEDDING_MODEL"
    )
    openai_embedding_dimensions: int = Field(default=3072, alias="OPENAI_EMBEDDING_DIMENSIONS")
    openai_llm_model: str = Field(default="gpt-4-turbo-preview", alias="OPENAI_LLM_MODEL")
    openai_max_retries: int = Field(default=3, alias="OPENAI_MAX_RETRIES")
    openai_timeout: int = Field(default=60, alias="OPENAI_TIMEOUT")

    # Chunking Configuration
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=77, alias="CHUNK_OVERLAP")
    chunk_min_size: int = Field(default=300, alias="CHUNK_MIN_SIZE")
    chunk_max_size: int = Field(default=600, alias="CHUNK_MAX_SIZE")

    # Vector Store Configuration
    faiss_index_path: str = Field(default="./indexes/faiss", alias="FAISS_INDEX_PATH")
    faiss_metadata_path: str = Field(default="./indexes/metadata", alias="FAISS_METADATA_PATH")
    faiss_index_type: str = Field(default="IndexFlatIP", alias="FAISS_INDEX_TYPE")
    vector_dimension: int = Field(default=3072, alias="VECTOR_DIMENSION")

    # Retrieval Configuration
    retrieval_top_k: int = Field(default=20, alias="RETRIEVAL_TOP_K")
    retrieval_rerank_top_k: int = Field(default=5, alias="RETRIEVAL_RERANK_TOP_K")
    retrieval_bm25_weight: float = Field(default=0.3, alias="RETRIEVAL_BM25_WEIGHT")
    retrieval_semantic_weight: float = Field(default=0.7, alias="RETRIEVAL_SEMANTIC_WEIGHT")
    retrieval_min_confidence: float = Field(default=0.65, alias="RETRIEVAL_MIN_CONFIDENCE")

    # Reranking Configuration
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANK_MODEL"
    )
    rerank_batch_size: int = Field(default=32, alias="RERANK_BATCH_SIZE")

    # Generation Configuration
    generation_max_tokens: int = Field(default=1024, alias="GENERATION_MAX_TOKENS")
    generation_temperature: float = Field(default=0.1, alias="GENERATION_TEMPERATURE")
    generation_require_citation: bool = Field(default=True, alias="GENERATION_REQUIRE_CITATION")
    generation_confidence_threshold: float = Field(
        default=0.65, alias="GENERATION_CONFIDENCE_THRESHOLD"
    )

    # Guardrails Configuration
    enable_pii_redaction: bool = Field(default=True, alias="ENABLE_PII_REDACTION")
    enable_rbac: bool = Field(default=True, alias="ENABLE_RBAC")
    enable_rate_limiting: bool = Field(default=True, alias="ENABLE_RATE_LIMITING")
    rate_limit_requests_per_minute: int = Field(
        default=60, alias="RATE_LIMIT_REQUESTS_PER_MINUTE"
    )
    pii_detection_model: str = Field(default="en_core_web_sm", alias="PII_DETECTION_MODEL")

    # Data Paths
    data_raw_path: str = Field(default="./data/raw", alias="DATA_RAW_PATH")
    data_staged_path: str = Field(default="./data/staged", alias="DATA_STAGED_PATH")
    data_manifests_path: str = Field(default="./data/manifests", alias="DATA_MANIFESTS_PATH")

    # Caching Configuration
    enable_embedding_cache: bool = Field(default=True, alias="ENABLE_EMBEDDING_CACHE")
    enable_query_cache: bool = Field(default=True, alias="ENABLE_QUERY_CACHE")
    cache_ttl_seconds: int = Field(default=3600, alias="CACHE_TTL_SECONDS")

    # Observability
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="json", alias="LOG_FORMAT")
    enable_metrics: bool = Field(default=True, alias="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, alias="METRICS_PORT")

    # Evaluation
    eval_dataset_path: str = Field(
        default="./data/eval/golden_qa.json", alias="EVAL_DATASET_PATH"
    )
    eval_min_recall_at_k: float = Field(default=0.85, alias="EVAL_MIN_RECALL_AT_K")
    eval_min_mrr: float = Field(default=0.75, alias="EVAL_MIN_MRR")
    eval_max_hallucination_rate: float = Field(default=0.05, alias="EVAL_MAX_HALLUCINATION_RATE")

    # Multi-tenancy
    default_tenant_id: str = Field(default="default", alias="DEFAULT_TENANT_ID")
    enable_tenant_isolation: bool = Field(default=True, alias="ENABLE_TENANT_ISOLATION")

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"

    def get_absolute_path(self, path: str) -> Path:
        """Convert relative path to absolute path."""
        p = Path(path)
        if p.is_absolute():
            return p
        return Path.cwd() / p


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings instance (useful for testing)."""
    global _settings
    _settings = None