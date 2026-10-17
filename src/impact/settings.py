"""Environment-driven configuration for the impact service (no secrets required)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    embedding_backend: str = Field(
        "sentence-transformers", description="hashing | sentence-transformers"
    )
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    api_key: str | None = Field(None, description="When set, requests need X-API-Key")
    log_level: str = "INFO"
    max_request_bytes: int = 1_000_000
    max_questions: int = 200
    v1_dir: Path = ROOT / "data" / "raw"
    v2_dir: Path = ROOT / "data" / "versions" / "v2"
    gold_path: Path = ROOT / "data" / "eval" / "retrieval_gold.json"
