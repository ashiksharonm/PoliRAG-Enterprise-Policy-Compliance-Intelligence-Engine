"""Shared data models for PoliRAG."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DocumentFormat(str, Enum):
    """Supported document formats."""

    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"
    JSON = "json"
    EMAIL = "email"
    TEXT = "text"


class Role(str, Enum):
    """User roles for RBAC."""

    ADMIN = "admin"
    LEGAL = "legal"
    AUDIT = "audit"
    COMPLIANCE = "compliance"
    READ_ONLY = "read_only"


class Document(BaseModel):
    """Document model with versioning and metadata."""

    id: UUID = Field(default_factory=uuid4)
    filename: str
    format: DocumentFormat
    content_hash: str  # SHA-256
    size_bytes: int
    tenant_id: str
    role_scope: List[Role] = Field(default_factory=list)
    version: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_path: Optional[str] = None
    staged_path: Optional[str] = None

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


class Chunk(BaseModel):
    """Text chunk model."""

    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    content: str
    content_hash: str  # SHA-256 of chunk content
    chunk_index: int
    token_count: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RetrievedChunk(BaseModel):
    """Retrieved chunk with score."""

    chunk: Chunk
    score: float
    rank: int
    retrieval_method: str  # "semantic", "bm25", "hybrid", "reranked"


class Query(BaseModel):
    """User query model."""

    id: UUID = Field(default_factory=uuid4)
    text: str
    tenant_id: str
    user_role: Role
    filters: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class GeneratedResponse(BaseModel):
    """Generated response with citations."""

    query_id: UUID
    answer: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    confidence: float
    tokens_used: int
    latency_ms: float
    model: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestionManifest(BaseModel):
    """Document ingestion manifest."""

    document_id: UUID
    filename: str
    content_hash: str
    status: str  # "pending", "processing", "completed", "failed"
    chunks_created: int = 0
    embeddings_generated: int = 0
    error_message: Optional[str] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    processing_time_seconds: Optional[float] = None


class EvaluationResult(BaseModel):
    """Evaluation metrics result."""

    test_set_name: str
    recall_at_k: float
    mean_reciprocal_rank: float
    hallucination_rate: float
    avg_confidence: float
    avg_latency_ms: float
    total_queries: int
    passed: bool
    created_at: datetime = Field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = Field(default_factory=dict)